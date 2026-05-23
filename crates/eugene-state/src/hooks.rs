//! Pre/post node hooks and permission modes.
//!
//! The graph runs registered hooks in order before and after every node.
//! A pre-hook can pass through, gate the node (surface an interrupt to
//! the caller), or skip it entirely. A post-hook observes the node's
//! result and can log, measure, or mutate the next step.
//!
//! The pattern is borrowed directly from Claude Code, whose tool
//! executor runs `executePreToolHooks` and `executePostToolHooks` around
//! every dispatch. The most common cross-cutting concern those hooks
//! handle is permission gating, which this module supports through the
//! [`PermissionMode`] enum and the [`ModeGate`] hook.
//!
//! ```
//! use async_trait::async_trait;
//! use eugene_state::{
//!     Graph, GraphError, NextStep, Node, Outcome,
//!     hooks::{HookOutcome, ModeGate, NodeHook, PermissionMode},
//! };
//!
//! #[derive(Default)]
//! struct State { mode: PermissionMode, applied: bool }
//!
//! struct ApplyEdit;
//! #[async_trait]
//! impl Node<State> for ApplyEdit {
//!     fn name(&self) -> &'static str { "apply" }
//!     fn requires_mode(&self) -> PermissionMode { PermissionMode::AcceptEdits }
//!     async fn run(&self, s: &mut State) -> Result<NextStep, GraphError> {
//!         s.applied = true;
//!         Ok(NextStep::Halt)
//!     }
//! }
//!
//! # #[tokio::main] async fn main() {
//! let mut graph = Graph::<State>::new("apply");
//! graph
//!     .add(ApplyEdit)
//!     .with_hook(ModeGate::new(|s: &State| s.mode));
//!
//! // In Plan mode the node is gated by the hook and the graph interrupts.
//! let outcome = graph
//!     .run(State { mode: PermissionMode::Plan, applied: false })
//!     .await
//!     .unwrap();
//! assert!(matches!(outcome, Outcome::Interrupted { .. }));
//! # }
//! ```

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::{GraphError, NextStep};

/// Permission modes mirror the set Claude Code's CLI carries on its
/// `ToolPermissionContext`. The runtime checks the node's
/// `requires_mode()` against the current mode through [`ModeGate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum PermissionMode {
    /// Read-only exploration; destructive nodes are gated.
    Plan,
    /// The standard mode: destructive nodes interrupt for confirmation.
    Default,
    /// Destructive edits are pre-approved; the graph runs through.
    AcceptEdits,
    /// All gates are bypassed. Use with care.
    BypassPermissions,
}

impl Default for PermissionMode {
    fn default() -> Self {
        Self::Default
    }
}

impl PermissionMode {
    /// Numeric capability level. Higher = more permissive.
    pub fn level(self) -> u8 {
        match self {
            PermissionMode::Plan => 0,
            PermissionMode::Default => 1,
            PermissionMode::AcceptEdits => 2,
            PermissionMode::BypassPermissions => 3,
        }
    }

    /// True when this mode allows operations that require `required`.
    pub fn permits(self, required: PermissionMode) -> bool {
        self.level() >= required.level()
    }
}

/// What a pre-hook tells the runner to do about the upcoming node.
#[derive(Debug, Clone)]
pub enum HookOutcome {
    /// Run the node as normal.
    Proceed,
    /// Skip the node and continue at the named successor.
    Skip(&'static str),
    /// Interrupt the graph. The caller sees the reason and the saved
    /// state can be resumed once the obstacle is cleared.
    Interrupt(String),
}

/// Hook invoked around every node transition. Implementations cover
/// permission gating, logging, metrics, and cache eviction.
#[async_trait]
pub trait NodeHook<S>: Send + Sync + 'static
where
    S: Send + Sync + 'static,
{
    /// Before the node runs. Default impl proceeds without comment.
    async fn before(
        &self,
        _node_name: &str,
        _required: PermissionMode,
        _state: &S,
    ) -> Result<HookOutcome, GraphError> {
        Ok(HookOutcome::Proceed)
    }

    /// After the node returns. Default impl ignores the outcome. The
    /// hook is free to mutate the next-step decision by returning a new
    /// `NextStep`; returning the original `result` is the no-op.
    async fn after(
        &self,
        _node_name: &str,
        _state: &S,
        result: NextStep,
    ) -> Result<NextStep, GraphError> {
        Ok(result)
    }
}

/// Built-in hook that interrupts whenever a node's required mode is
/// not satisfied by the current permission mode.
///
/// Construct with a closure that extracts the mode from your state:
///
/// ```ignore
/// graph.with_hook(ModeGate::new(|s: &MyState| s.mode));
/// ```
pub struct ModeGate<S, F>
where
    F: Fn(&S) -> PermissionMode + Send + Sync + 'static,
    S: Send + Sync + 'static,
{
    extractor: F,
    _marker: std::marker::PhantomData<fn(&S)>,
}

impl<S, F> ModeGate<S, F>
where
    F: Fn(&S) -> PermissionMode + Send + Sync + 'static,
    S: Send + Sync + 'static,
{
    pub fn new(extractor: F) -> Self {
        Self {
            extractor,
            _marker: std::marker::PhantomData,
        }
    }
}

#[async_trait]
impl<S, F> NodeHook<S> for ModeGate<S, F>
where
    F: Fn(&S) -> PermissionMode + Send + Sync + 'static,
    S: Send + Sync + 'static,
{
    async fn before(
        &self,
        node_name: &str,
        required: PermissionMode,
        state: &S,
    ) -> Result<HookOutcome, GraphError> {
        let mode = (self.extractor)(state);
        if mode.permits(required) {
            Ok(HookOutcome::Proceed)
        } else {
            Ok(HookOutcome::Interrupt(format!(
                "node `{node_name}` needs {required:?} but the current mode is {mode:?}"
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permits_is_monotone_in_level() {
        assert!(PermissionMode::BypassPermissions.permits(PermissionMode::AcceptEdits));
        assert!(PermissionMode::AcceptEdits.permits(PermissionMode::Default));
        assert!(PermissionMode::Default.permits(PermissionMode::Default));
        assert!(!PermissionMode::Plan.permits(PermissionMode::Default));
        assert!(!PermissionMode::Default.permits(PermissionMode::AcceptEdits));
    }
}
