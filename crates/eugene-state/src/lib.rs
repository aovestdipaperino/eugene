//! Graph runner with checkpoints and human-in-the-loop interrupts.
//!
//! An agent loop is sufficient when the model can plan and execute a task
//! in one breath. As soon as the work splits into phases — plan, execute,
//! verify — and you want any of: per-step retries, an approval gate before
//! a destructive operation, or the ability to resume after a crash, the
//! shape that survives is a state machine.
//!
//! This crate provides three things. A `Node<S>` trait whose `run` method
//! takes a mutable typed state and returns the next step. A `Graph<S>`
//! that walks nodes by name. A `Checkpointer<S>` trait with an
//! in-memory implementation in the default build and a `rusqlite`-backed
//! implementation behind the `sqlite` feature.
//!
//! ```
//! use async_trait::async_trait;
//! use eugene_state::{Graph, GraphError, NextStep, Node, Outcome};
//!
//! #[derive(Default)]
//! struct State { count: u32 }
//!
//! struct Increment;
//! #[async_trait]
//! impl Node<State> for Increment {
//!     fn name(&self) -> &'static str { "inc" }
//!     async fn run(&self, s: &mut State) -> Result<NextStep, GraphError> {
//!         s.count += 1;
//!         if s.count >= 3 { Ok(NextStep::Halt) } else { Ok(NextStep::Goto("inc")) }
//!     }
//! }
//!
//! # #[tokio::main]
//! # async fn main() {
//! let mut graph = Graph::<State>::new("inc");
//! graph.add(Increment);
//! let Outcome::Done(state) = graph.run(State::default()).await.unwrap() else {
//!     panic!("expected done");
//! };
//! assert_eq!(state.count, 3);
//! # }
//! ```

use std::collections::HashMap;

use async_trait::async_trait;
use thiserror::Error;

#[cfg(feature = "sqlite")]
pub mod sqlite;

mod memory;
pub use memory::InMemoryCheckpointer;

#[derive(Debug, Error)]
pub enum GraphError {
    #[error("unknown node: {0}")]
    UnknownNode(String),

    #[error("max steps exceeded ({0})")]
    MaxSteps(usize),

    #[error("transient failure (worth retrying): {0}")]
    Transient(String),

    #[error("checkpoint error: {0}")]
    Checkpoint(String),

    #[error("node failed: {0}")]
    Node(String),
}

impl GraphError {
    pub fn is_transient(&self) -> bool {
        matches!(self, GraphError::Transient(_))
    }
}

/// What a node tells the runner to do next.
#[derive(Debug, Clone)]
pub enum NextStep {
    /// Continue at the named node.
    Goto(&'static str),
    /// Stop. The state at this point is the result.
    Halt,
    /// Pause and surface a reason to the caller. The graph saves a
    /// checkpoint at the current node so the caller can resume after
    /// review.
    Interrupt { reason: String },
}

/// Implemented by each phase of the graph. The state is mutable; nodes
/// edit it in place and return `NextStep`.
#[async_trait]
pub trait Node<S>: Send + Sync + 'static
where
    S: Send + Sync + 'static,
{
    /// Stable, unique node name. Lowercase, kebab- or snake-case.
    fn name(&self) -> &'static str;

    async fn run(&self, state: &mut S) -> Result<NextStep, GraphError>;
}

/// The result of running (or resuming) a graph.
#[derive(Debug)]
pub enum Outcome<S> {
    Done(S),
    Interrupted {
        state: S,
        next_node: String,
        reason: String,
    },
}

/// A persisted snapshot of a paused graph.
#[derive(Debug, Clone)]
pub struct Checkpoint<S> {
    pub next_node: String,
    pub state: S,
}

#[async_trait]
pub trait Checkpointer<S>: Send + Sync
where
    S: Send + Sync + 'static,
{
    async fn save(&self, run_id: &str, next_node: &str, state: &S) -> Result<(), GraphError>;

    async fn load(&self, run_id: &str) -> Result<Option<Checkpoint<S>>, GraphError>;

    async fn delete(&self, run_id: &str) -> Result<(), GraphError>;
}

/// A graph of named nodes. `S` is the typed state every node shares.
pub struct Graph<S>
where
    S: Send + Sync + 'static,
{
    nodes: HashMap<&'static str, Box<dyn Node<S>>>,
    entry: &'static str,
    max_steps: usize,
    retry_attempts: u32,
}

impl<S> Graph<S>
where
    S: Send + Sync + 'static,
{
    pub fn new(entry: &'static str) -> Self {
        Self {
            nodes: HashMap::new(),
            entry,
            max_steps: 32,
            retry_attempts: 1,
        }
    }

    /// Cap on the number of node transitions before the graph errors out.
    /// Prevents an infinite loop on a buggy goto.
    pub fn max_steps(mut self, n: usize) -> Self {
        self.max_steps = n;
        self
    }

    /// Number of times to re-run a node when it returns
    /// `GraphError::Transient`. Default is 1 (no retries).
    pub fn retry_attempts(mut self, n: u32) -> Self {
        self.retry_attempts = n.max(1);
        self
    }

    pub fn add(&mut self, node: impl Node<S>) -> &mut Self {
        self.nodes.insert(node.name(), Box::new(node));
        self
    }

    /// Run the graph from the entry node with an initial state.
    pub async fn run(&self, initial: S) -> Result<Outcome<S>, GraphError> {
        self.run_from(initial, self.entry).await
    }

    /// Run the graph starting at a named node.
    pub async fn run_from(&self, initial: S, start: &str) -> Result<Outcome<S>, GraphError> {
        let mut current = self.lookup(start)?;
        let mut state = initial;

        for _ in 0..self.max_steps {
            let node = &self.nodes[current];
            let next = self.run_node_with_retry(node.as_ref(), &mut state).await?;
            match next {
                NextStep::Halt => return Ok(Outcome::Done(state)),
                NextStep::Interrupt { reason } => {
                    return Ok(Outcome::Interrupted {
                        state,
                        next_node: current.to_string(),
                        reason,
                    });
                }
                NextStep::Goto(next_name) => {
                    current = self.lookup(next_name)?;
                }
            }
        }
        Err(GraphError::MaxSteps(self.max_steps))
    }

    /// Run with checkpoint persistence. On entry, looks for a saved
    /// checkpoint under `run_id` and resumes from it; otherwise starts
    /// from the entry node. Saves a checkpoint after every node so a
    /// crash mid-graph loses at most one node's work.
    pub async fn run_with_checkpoint<C>(
        &self,
        run_id: &str,
        initial: S,
        ckpt: &C,
    ) -> Result<Outcome<S>, GraphError>
    where
        C: Checkpointer<S>,
    {
        let (mut state, mut current) = match ckpt.load(run_id).await? {
            Some(cp) => (cp.state, self.lookup(&cp.next_node)?),
            None => (initial, self.entry),
        };

        for _ in 0..self.max_steps {
            let node = &self.nodes[current];
            let next = self.run_node_with_retry(node.as_ref(), &mut state).await?;
            match next {
                NextStep::Halt => {
                    ckpt.delete(run_id).await?;
                    return Ok(Outcome::Done(state));
                }
                NextStep::Interrupt { reason } => {
                    ckpt.save(run_id, current, &state).await?;
                    return Ok(Outcome::Interrupted {
                        state,
                        next_node: current.to_string(),
                        reason,
                    });
                }
                NextStep::Goto(next_name) => {
                    current = self.lookup(next_name)?;
                    ckpt.save(run_id, current, &state).await?;
                }
            }
        }
        Err(GraphError::MaxSteps(self.max_steps))
    }

    async fn run_node_with_retry(
        &self,
        node: &dyn Node<S>,
        state: &mut S,
    ) -> Result<NextStep, GraphError> {
        let mut last: Option<GraphError> = None;
        for attempt in 0..self.retry_attempts {
            match node.run(state).await {
                Ok(next) => return Ok(next),
                Err(e) if e.is_transient() && attempt + 1 < self.retry_attempts => {
                    last = Some(e);
                }
                Err(e) => return Err(e),
            }
        }
        Err(last.unwrap_or_else(|| GraphError::Node("retry exhausted".into())))
    }

    fn lookup(&self, name: &str) -> Result<&'static str, GraphError> {
        self.nodes
            .keys()
            .find(|&&k| k == name)
            .copied()
            .ok_or_else(|| GraphError::UnknownNode(name.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};
    use std::sync::atomic::{AtomicU32, Ordering};

    #[derive(Default, Clone, Serialize, Deserialize, Debug, PartialEq)]
    struct State {
        count: u32,
        log: Vec<String>,
    }

    struct Inc;
    #[async_trait]
    impl Node<State> for Inc {
        fn name(&self) -> &'static str {
            "inc"
        }
        async fn run(&self, s: &mut State) -> Result<NextStep, GraphError> {
            s.count += 1;
            s.log.push("inc".into());
            if s.count >= 3 {
                Ok(NextStep::Goto("done"))
            } else {
                Ok(NextStep::Goto("inc"))
            }
        }
    }

    struct Done;
    #[async_trait]
    impl Node<State> for Done {
        fn name(&self) -> &'static str {
            "done"
        }
        async fn run(&self, s: &mut State) -> Result<NextStep, GraphError> {
            s.log.push("done".into());
            Ok(NextStep::Halt)
        }
    }

    #[tokio::test]
    async fn loop_terminates_via_halt() {
        let mut g = Graph::<State>::new("inc");
        g.add(Inc).add(Done);
        let Outcome::Done(state) = g.run(State::default()).await.unwrap() else {
            panic!("expected done");
        };
        assert_eq!(state.count, 3);
        assert_eq!(state.log, vec!["inc", "inc", "inc", "done"]);
    }

    #[tokio::test]
    async fn unknown_node_errors() {
        struct WrongGoto;
        #[async_trait]
        impl Node<State> for WrongGoto {
            fn name(&self) -> &'static str {
                "start"
            }
            async fn run(&self, _: &mut State) -> Result<NextStep, GraphError> {
                Ok(NextStep::Goto("nope"))
            }
        }
        let mut g = Graph::<State>::new("start");
        g.add(WrongGoto);
        let err = g.run(State::default()).await.unwrap_err();
        assert!(matches!(err, GraphError::UnknownNode(_)));
    }

    #[tokio::test]
    async fn max_steps_terminates_runaway_loop() {
        struct Spin;
        #[async_trait]
        impl Node<State> for Spin {
            fn name(&self) -> &'static str {
                "spin"
            }
            async fn run(&self, _: &mut State) -> Result<NextStep, GraphError> {
                Ok(NextStep::Goto("spin"))
            }
        }
        let mut g = Graph::<State>::new("spin").max_steps(5);
        g.add(Spin);
        let err = g.run(State::default()).await.unwrap_err();
        assert!(matches!(err, GraphError::MaxSteps(5)));
    }

    #[tokio::test]
    async fn interrupt_returns_state_and_next_node() {
        struct Pause;
        #[async_trait]
        impl Node<State> for Pause {
            fn name(&self) -> &'static str {
                "pause"
            }
            async fn run(&self, s: &mut State) -> Result<NextStep, GraphError> {
                s.log.push("paused".into());
                Ok(NextStep::Interrupt {
                    reason: "need approval".into(),
                })
            }
        }
        let mut g = Graph::<State>::new("pause");
        g.add(Pause);
        match g.run(State::default()).await.unwrap() {
            Outcome::Interrupted {
                state,
                next_node,
                reason,
            } => {
                assert_eq!(next_node, "pause");
                assert_eq!(reason, "need approval");
                assert_eq!(state.log, vec!["paused"]);
            }
            other => panic!("expected interrupt, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn retry_recovers_from_transient() {
        struct Flaky(AtomicU32);
        #[async_trait]
        impl Node<State> for Flaky {
            fn name(&self) -> &'static str {
                "flaky"
            }
            async fn run(&self, _: &mut State) -> Result<NextStep, GraphError> {
                let n = self.0.fetch_add(1, Ordering::SeqCst);
                if n < 2 {
                    Err(GraphError::Transient(format!("attempt {n}")))
                } else {
                    Ok(NextStep::Halt)
                }
            }
        }
        let mut g = Graph::<State>::new("flaky").retry_attempts(3);
        g.add(Flaky(AtomicU32::new(0)));
        let outcome = g.run(State::default()).await.unwrap();
        assert!(matches!(outcome, Outcome::Done(_)));
    }

    #[tokio::test]
    async fn checkpoint_survives_interrupt_and_resumes() {
        struct StepOne;
        #[async_trait]
        impl Node<State> for StepOne {
            fn name(&self) -> &'static str {
                "step1"
            }
            async fn run(&self, s: &mut State) -> Result<NextStep, GraphError> {
                s.log.push("step1".into());
                Ok(NextStep::Interrupt {
                    reason: "approve".into(),
                })
            }
        }
        struct StepTwo;
        #[async_trait]
        impl Node<State> for StepTwo {
            fn name(&self) -> &'static str {
                "step2"
            }
            async fn run(&self, s: &mut State) -> Result<NextStep, GraphError> {
                s.log.push("step2".into());
                Ok(NextStep::Halt)
            }
        }

        let mut g = Graph::<State>::new("step1");
        g.add(StepOne).add(StepTwo);
        let ckpt: InMemoryCheckpointer<State> = InMemoryCheckpointer::new();

        // First run interrupts.
        let outcome = g
            .run_with_checkpoint("run-1", State::default(), &ckpt)
            .await
            .unwrap();
        assert!(matches!(outcome, Outcome::Interrupted { .. }));

        // We've decided to approve. Bump the next node manually to step2.
        ckpt.save(
            "run-1",
            "step2",
            &State {
                count: 0,
                log: vec!["step1".into()],
            },
        )
        .await
        .unwrap();

        // Resume: the checkpoint says step2.
        let outcome = g
            .run_with_checkpoint("run-1", State::default(), &ckpt)
            .await
            .unwrap();
        match outcome {
            Outcome::Done(s) => {
                assert_eq!(s.log, vec!["step1", "step2"]);
            }
            other => panic!("expected done, got {other:?}"),
        }
        assert!(ckpt.load("run-1").await.unwrap().is_none());
    }
}
