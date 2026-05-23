//! Pre/post skill hooks.
//!
//! The registry runs hooks before and after every tool dispatch. A
//! pre-hook can pass through, deny the call (the model sees an
//! `is_error` tool result), replace the dispatch with a synthetic
//! output (for cached or stubbed tools), or rewrite the arguments
//! before the skill sees them.
//!
//! The pattern is borrowed from Claude Code, whose tool executor runs
//! `executePreToolHooks` and `executePostToolHooks` around every call.
//! That is where their `CanUseToolFn` permission gate lives, alongside
//! analytics, redaction, and user-defined shell hooks.

use async_trait::async_trait;
use serde_json::Value;

use crate::{SkillError, SkillMetadata};

/// What a pre-hook tells the registry to do about the upcoming dispatch.
#[derive(Debug, Clone)]
pub enum HookOutcome {
    /// Run the skill with the original arguments.
    Proceed,
    /// Rewrite the arguments before dispatch. Useful for redaction or
    /// injecting context the model did not supply.
    Modify(Value),
    /// Block the dispatch. The model sees an error result with `reason`.
    Deny(String),
    /// Skip the dispatch and return `output` as if the skill had run.
    /// Useful for caches and pre-baked replies.
    Replace(String),
}

/// Hook invoked around every dispatch. Implementations cover permission
/// gating, logging, metrics, and result rewriting.
#[async_trait]
pub trait SkillHook: Send + Sync + 'static {
    async fn before(
        &self,
        _skill_name: &str,
        _metadata: SkillMetadata,
        _args: &Value,
    ) -> Result<HookOutcome, SkillError> {
        Ok(HookOutcome::Proceed)
    }

    async fn after(
        &self,
        _skill_name: &str,
        _args: &Value,
        result: Result<String, SkillError>,
    ) -> Result<String, SkillError> {
        result
    }
}

/// A built-in hook that denies any call to a destructive skill unless
/// `is_destructive_allowed` returns true. Construct with a closure or
/// any callable that decides per-skill at runtime.
///
/// This is the smallest useful version of Claude Code's `CanUseToolFn`:
/// the gate looks at the metadata Eugene already carries on each skill
/// and refuses the destructive ones in modes that have not pre-approved
/// edits.
pub struct PermissionGate<F>
where
    F: Fn(&str, SkillMetadata) -> bool + Send + Sync + 'static,
{
    is_allowed: F,
}

impl<F> PermissionGate<F>
where
    F: Fn(&str, SkillMetadata) -> bool + Send + Sync + 'static,
{
    pub fn new(is_allowed: F) -> Self {
        Self { is_allowed }
    }
}

#[async_trait]
impl<F> SkillHook for PermissionGate<F>
where
    F: Fn(&str, SkillMetadata) -> bool + Send + Sync + 'static,
{
    async fn before(
        &self,
        skill_name: &str,
        metadata: SkillMetadata,
        _args: &Value,
    ) -> Result<HookOutcome, SkillError> {
        if (self.is_allowed)(skill_name, metadata) {
            Ok(HookOutcome::Proceed)
        } else {
            Ok(HookOutcome::Deny(format!(
                "permission denied: skill `{skill_name}` requires explicit approval \
                 (read_only={}, destructive={})",
                metadata.is_read_only, metadata.is_destructive
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn permission_gate_denies_destructive_by_default() {
        let gate = PermissionGate::new(|_, md: SkillMetadata| !md.is_destructive);
        let md = SkillMetadata {
            is_read_only: false,
            is_destructive: true,
            is_concurrency_safe: false,
        };
        let outcome = gate.before("rm", md, &Value::Null).await.unwrap();
        assert!(matches!(outcome, HookOutcome::Deny(_)));
    }

    #[tokio::test]
    async fn permission_gate_lets_read_only_through() {
        let gate = PermissionGate::new(|_, md: SkillMetadata| !md.is_destructive);
        let md = SkillMetadata {
            is_read_only: true,
            is_destructive: false,
            is_concurrency_safe: true,
        };
        let outcome = gate.before("read", md, &Value::Null).await.unwrap();
        assert!(matches!(outcome, HookOutcome::Proceed));
    }
}
