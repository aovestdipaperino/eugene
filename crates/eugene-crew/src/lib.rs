//! Multi-agent crew runner.
//!
//! When a task is genuinely better served by several specialists than one
//! generalist — researcher, critic, synthesizer; planner, executor,
//! verifier; for-the-motion, against-the-motion, judge — the shape that
//! works is a *crew*: a collection of agents that the runtime composes.
//!
//! This crate provides three pieces. An [`Agent`] trait whose handle
//! method takes a free-form query and returns a free-form response. A
//! [`Crew`] struct that holds agents by name and offers parallel,
//! sequential, and routed dispatch. A [`debate`] function that runs two
//! specialists against each other for a fixed number of rounds with an
//! optional judge.
//!
//! ```
//! use async_trait::async_trait;
//! use eugene_crew::{Agent, AgentDescription, Crew, CrewError};
//!
//! struct Echo(&'static str);
//! #[async_trait]
//! impl Agent for Echo {
//!     fn description(&self) -> AgentDescription {
//!         AgentDescription { name: self.0.into(), summary: "echo".into() }
//!     }
//!     async fn handle(&self, query: &str) -> Result<String, CrewError> {
//!         Ok(format!("{}: {}", self.0, query))
//!     }
//! }
//!
//! # #[tokio::main] async fn main() {
//! let mut crew = Crew::new();
//! crew.add(Echo("researcher")).add(Echo("critic"));
//! let results = crew.run_parallel(&[
//!     ("researcher", "what is rust?"),
//!     ("critic", "what is rust?"),
//! ]).await;
//! assert_eq!(results.len(), 2);
//! # }
//! ```

use std::collections::HashMap;

use async_trait::async_trait;
use futures::future::join_all;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CrewError {
    #[error("unknown agent: {0}")]
    UnknownAgent(String),

    #[error("router failed: {0}")]
    Router(String),

    #[error("agent failed: {0}")]
    Agent(String),

    #[error("debate failed: {0}")]
    Debate(String),
}

/// A short, structured description of an agent. Surfaced to routers and
/// to the user. The same shape Claude Code's `AgentDefinition` carries.
#[derive(Debug, Clone)]
pub struct AgentDescription {
    pub name: String,
    pub summary: String,
}

/// Implemented by each specialist. The `handle` method takes a free-form
/// query and returns a free-form response. How the agent produces that
/// response (LLM call, tool registry, rule-based logic) is its own
/// business.
#[async_trait]
pub trait Agent: Send + Sync + 'static {
    fn description(&self) -> AgentDescription;

    fn name(&self) -> String {
        self.description().name
    }

    async fn handle(&self, query: &str) -> Result<String, CrewError>;
}

/// Selects which agent (or agents) should handle a query.
#[async_trait]
pub trait Router: Send + Sync + 'static {
    async fn pick(&self, query: &str, candidates: &[AgentDescription])
    -> Result<String, CrewError>;
}

/// Crew owns a collection of named agents and exposes the dispatch
/// patterns (run-one, run-parallel, run-sequential, route).
#[derive(Default)]
pub struct Crew {
    agents: HashMap<String, Box<dyn Agent>>,
    order: Vec<String>,
}

impl Crew {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an agent. Calling with the same name replaces the
    /// existing one.
    pub fn add(&mut self, agent: impl Agent) -> &mut Self {
        let name = agent.name();
        if !self.agents.contains_key(&name) {
            self.order.push(name.clone());
        }
        self.agents.insert(name, Box::new(agent));
        self
    }

    pub fn descriptions(&self) -> Vec<AgentDescription> {
        self.order
            .iter()
            .filter_map(|n| self.agents.get(n))
            .map(|a| a.description())
            .collect()
    }

    /// Run one agent by name.
    pub async fn run(&self, name: &str, query: &str) -> Result<String, CrewError> {
        let agent = self
            .agents
            .get(name)
            .ok_or_else(|| CrewError::UnknownAgent(name.to_string()))?;
        agent.handle(query).await
    }

    /// Run several agents in parallel. Each entry is (agent_name, query).
    /// Returns results in the same order, with each `Result` carrying the
    /// success or per-agent failure.
    pub async fn run_parallel(
        &self,
        calls: &[(impl AsRef<str> + Sync, impl AsRef<str> + Sync)],
    ) -> Vec<(String, Result<String, CrewError>)> {
        let futs = calls.iter().map(|(name, query)| async move {
            let name = name.as_ref().to_string();
            let query = query.as_ref();
            let result = match self.agents.get(&name) {
                Some(a) => a.handle(query).await,
                None => Err(CrewError::UnknownAgent(name.clone())),
            };
            (name, result)
        });
        join_all(futs).await
    }

    /// Run agents one after another. Each agent sees the previous
    /// agent's output as its query. Returns the entire chain of outputs.
    pub async fn run_sequential(
        &self,
        names: &[&str],
        initial: &str,
    ) -> Result<Vec<String>, CrewError> {
        let mut outputs = Vec::with_capacity(names.len());
        let mut next_input = initial.to_string();
        for name in names {
            let agent = self
                .agents
                .get(*name)
                .ok_or_else(|| CrewError::UnknownAgent((*name).to_string()))?;
            let out = agent.handle(&next_input).await?;
            next_input = out.clone();
            outputs.push(out);
        }
        Ok(outputs)
    }

    /// Ask the router to pick one agent, then run it.
    pub async fn route(&self, query: &str, router: &dyn Router) -> Result<String, CrewError> {
        let chosen = router.pick(query, &self.descriptions()).await?;
        self.run(&chosen, query).await
    }
}

/// A minimal router that picks the first agent whose name appears as a
/// case-insensitive substring of the query. Cheap, deterministic, and
/// useful for tests and CLI triage.
pub struct KeywordRouter;

#[async_trait]
impl Router for KeywordRouter {
    async fn pick(
        &self,
        query: &str,
        candidates: &[AgentDescription],
    ) -> Result<String, CrewError> {
        let q = query.to_ascii_lowercase();
        for c in candidates {
            if q.contains(&c.name.to_ascii_lowercase()) {
                return Ok(c.name.clone());
            }
        }
        candidates
            .first()
            .map(|c| c.name.clone())
            .ok_or_else(|| CrewError::Router("no agents to route to".into()))
    }
}

/// Two-agent debate. `pro` defends a position, `con` argues against,
/// and `judge` (when supplied) picks a winner after `rounds`. Returns
/// the judge's verdict, or the final pro/con exchange when no judge is
/// given.
pub async fn debate(
    pro: &dyn Agent,
    con: &dyn Agent,
    judge: Option<&dyn Agent>,
    topic: &str,
    rounds: u32,
) -> Result<String, CrewError> {
    let mut transcript = format!("Topic: {topic}\n");
    let mut last_con: String = String::new();

    for round in 1..=rounds {
        let pro_prompt = if last_con.is_empty() {
            format!("Argue FOR: {topic}")
        } else {
            format!(
                "Argue FOR: {topic}\n\nOpponent's last argument:\n{last_con}\n\nRebut and strengthen your case."
            )
        };
        let last_pro = pro.handle(&pro_prompt).await?;
        transcript.push_str(&format!(
            "\n--- Round {round}, PRO ({}) ---\n{last_pro}\n",
            pro.name()
        ));

        let con_prompt = format!(
            "Argue AGAINST: {topic}\n\nOpponent's last argument:\n{last_pro}\n\nRebut and strengthen your case."
        );
        last_con = con.handle(&con_prompt).await?;
        transcript.push_str(&format!(
            "\n--- Round {round}, CON ({}) ---\n{last_con}\n",
            con.name()
        ));
    }
    // Silence the unused-assignment warning on the final round's last_con
    // while keeping the variable visible for any judge that wants the full
    // transcript that already includes it.
    let _ = &last_con;

    match judge {
        Some(j) => {
            let verdict_prompt = format!(
                "Read the following debate transcript and pick a winner. \
                 Give a one-paragraph verdict.\n\n{transcript}"
            );
            j.handle(&verdict_prompt).await
        }
        None => Ok(transcript),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::time::{Duration, Instant};

    struct Stub {
        name: &'static str,
        delay: Duration,
        reply: String,
    }

    impl Stub {
        fn new(name: &'static str, reply: impl Into<String>) -> Self {
            Self {
                name,
                delay: Duration::ZERO,
                reply: reply.into(),
            }
        }
        fn with_delay(mut self, delay: Duration) -> Self {
            self.delay = delay;
            self
        }
    }

    #[async_trait]
    impl Agent for Stub {
        fn description(&self) -> AgentDescription {
            AgentDescription {
                name: self.name.into(),
                summary: format!("stub agent {}", self.name),
            }
        }
        async fn handle(&self, query: &str) -> Result<String, CrewError> {
            tokio::time::sleep(self.delay).await;
            Ok(format!("{}({query})={}", self.name, self.reply))
        }
    }

    #[tokio::test]
    async fn run_dispatches_by_name() {
        let mut crew = Crew::new();
        crew.add(Stub::new("alice", "ok"));
        let out = crew.run("alice", "hi").await.unwrap();
        assert_eq!(out, "alice(hi)=ok");
    }

    #[tokio::test]
    async fn unknown_agent_is_an_error() {
        let crew = Crew::new();
        let err = crew.run("nobody", "hi").await.unwrap_err();
        assert!(matches!(err, CrewError::UnknownAgent(_)));
    }

    #[tokio::test]
    async fn parallel_runs_concurrently() {
        let mut crew = Crew::new();
        crew.add(Stub::new("a", "1").with_delay(Duration::from_millis(80)));
        crew.add(Stub::new("b", "2").with_delay(Duration::from_millis(80)));
        crew.add(Stub::new("c", "3").with_delay(Duration::from_millis(80)));

        let calls = vec![("a", "x"), ("b", "x"), ("c", "x")];
        let start = Instant::now();
        let results = crew.run_parallel(&calls).await;
        let elapsed = start.elapsed();

        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|(_, r)| r.is_ok()));
        // Parallel: ~80ms; serial would be ~240ms.
        assert!(
            elapsed < Duration::from_millis(200),
            "expected parallel, took {elapsed:?}"
        );
    }

    #[tokio::test]
    async fn sequential_threads_outputs_through() {
        let mut crew = Crew::new();
        crew.add(Stub::new("first", "OUT1"));
        crew.add(Stub::new("second", "OUT2"));

        let outputs = crew
            .run_sequential(&["first", "second"], "start")
            .await
            .unwrap();
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0], "first(start)=OUT1");
        // Second sees first's output as its query.
        assert_eq!(outputs[1], "second(first(start)=OUT1)=OUT2");
    }

    #[tokio::test]
    async fn keyword_router_matches_name_in_query() {
        let mut crew = Crew::new();
        crew.add(Stub::new("coder", "code"));
        crew.add(Stub::new("writer", "prose"));
        let out = crew
            .route("please coder, refactor this", &KeywordRouter)
            .await
            .unwrap();
        assert!(out.starts_with("coder("));
    }

    #[tokio::test]
    async fn debate_runs_pro_and_con_for_rounds() {
        let counter = Arc::new(AtomicU32::new(0));

        struct Counting {
            counter: Arc<AtomicU32>,
            name: &'static str,
        }
        #[async_trait]
        impl Agent for Counting {
            fn description(&self) -> AgentDescription {
                AgentDescription {
                    name: self.name.into(),
                    summary: String::new(),
                }
            }
            async fn handle(&self, _: &str) -> Result<String, CrewError> {
                self.counter.fetch_add(1, Ordering::SeqCst);
                Ok(format!("{} speaks", self.name))
            }
        }

        let pro = Counting {
            counter: counter.clone(),
            name: "pro",
        };
        let con = Counting {
            counter: counter.clone(),
            name: "con",
        };
        let result = debate(&pro, &con, None, "is rust fun?", 2).await.unwrap();
        // 2 rounds * 2 sides = 4 calls.
        assert_eq!(counter.load(Ordering::SeqCst), 4);
        assert!(result.contains("Round 2, CON"));
    }
}
