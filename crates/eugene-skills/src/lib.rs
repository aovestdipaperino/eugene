//! Skill trait, registry, and retry helpers.
//!
//! A `Skill` is one tool the agent can call: a name, a description, a
//! typed input, and an async `run` method. The trait derives its JSON
//! Schema from the input type via `schemars`, so a Rust struct is the
//! single source of truth for the wire format.
//!
//! Skills land in a `Registry`. The registry owns the dispatch table,
//! produces provider-neutral `ToolDefinition`s for the prompt builder,
//! and runs concurrency-safe calls in parallel when the model issues
//! several in one turn.
//!
//! ```no_run
//! use async_trait::async_trait;
//! use eugene_skills::{Registry, Skill, SkillError};
//! use schemars::JsonSchema;
//! use serde::Deserialize;
//!
//! #[derive(Deserialize, JsonSchema)]
//! struct EchoArgs { message: String }
//!
//! struct Echo;
//!
//! #[async_trait]
//! impl Skill for Echo {
//!     type Input = EchoArgs;
//!     fn name(&self) -> &str { "echo" }
//!     fn description(&self) -> &str { "Echo the message back to the user." }
//!     async fn run(&self, input: EchoArgs) -> Result<String, SkillError> {
//!         Ok(input.message)
//!     }
//! }
//!
//! # async fn run() {
//! let mut registry = Registry::new();
//! registry.add(Echo);
//! let definitions = registry.definitions();
//! assert_eq!(definitions.len(), 1);
//! # }
//! ```

use std::collections::HashMap;
use std::time::Duration;

use async_trait::async_trait;
use eugene_core::ToolDefinition;
use futures::future::join_all;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde_json::Value;
use thiserror::Error;

pub mod hooks;
pub use hooks::{HookOutcome, PermissionGate, SkillHook};

/// Errors a skill can return. The agent can distinguish transient failures
/// (worth retrying) from terminal ones (give up and report to the model).
#[derive(Debug, Error)]
pub enum SkillError {
    #[error("invalid arguments: {0}")]
    BadArgs(#[from] serde_json::Error),

    #[error("safety check failed: {0}")]
    Safety(String),

    #[error("transient failure (worth retrying): {0}")]
    Transient(String),

    #[error("tool failed: {0}")]
    Tool(String),
}

impl SkillError {
    pub fn is_transient(&self) -> bool {
        matches!(self, SkillError::Transient(_))
    }
}

/// Metadata the agent uses to decide whether a tool call is safe to run
/// in parallel, requires user confirmation, or can be auto-approved.
/// Modelled after the same flags Claude Code carries on each Tool.
#[derive(Debug, Clone, Copy)]
pub struct SkillMetadata {
    pub is_read_only: bool,
    pub is_destructive: bool,
    pub is_concurrency_safe: bool,
}

/// The skill author's view. Implement this with a typed `Input` and
/// schemars takes care of the schema.
#[async_trait]
pub trait Skill: Send + Sync + 'static {
    /// Input type. Must deserialize from JSON and derive `JsonSchema`.
    type Input: DeserializeOwned + JsonSchema + Send + 'static;

    /// Unique tool name shown to the model. Lowercase, snake_case.
    fn name(&self) -> &str;

    /// One-paragraph description the model uses to choose this tool.
    /// Be specific. Hedging language leads to hedging behaviour.
    fn description(&self) -> &str;

    /// True when running the skill does not modify external state.
    /// Defaults to true: most tools you write first are reads.
    fn is_read_only(&self) -> bool {
        true
    }

    /// True when the skill performs irreversible operations. The agent
    /// surfaces these for confirmation in interactive use.
    fn is_destructive(&self) -> bool {
        false
    }

    /// True when two calls in the same turn can run in parallel without
    /// interfering. Read-only tools are concurrency-safe by default.
    fn is_concurrency_safe(&self) -> bool {
        self.is_read_only()
    }

    /// Alternative names the model can use to call this skill. Useful
    /// for backwards compatibility when a skill is renamed: keep the
    /// old name as an alias for a release or two before retiring it.
    fn aliases(&self) -> &[&'static str] {
        &[]
    }

    /// One-line capability phrase (3-10 words, no trailing period) used
    /// by `Registry::search` to surface this skill on keyword match.
    /// Prefer terms that are NOT already in the skill name. The same
    /// convention Claude Code's `searchHint` follows for `ToolSearch`.
    fn search_hint(&self) -> Option<&str> {
        None
    }

    /// True when the schema for this skill should be omitted from the
    /// initial tool list and only surfaced after a `Registry::search`
    /// hit. Use for agents with many skills whose prompt would otherwise
    /// balloon: hide the long tail, keep the core always-loaded.
    fn should_defer(&self) -> bool {
        false
    }

    /// Run the tool. The output is whatever string the model should see.
    /// Structured results travel as JSON inside this string.
    async fn run(&self, input: Self::Input) -> Result<String, SkillError>;
}

/// Object-safe wrapper so the registry can hold heterogeneous skills.
#[async_trait]
pub trait DynSkill: Send + Sync {
    fn definition(&self) -> ToolDefinition;
    fn metadata(&self) -> SkillMetadata;
    fn aliases(&self) -> Vec<&'static str>;
    fn search_hint(&self) -> Option<&str>;
    fn should_defer(&self) -> bool;
    async fn run_json(&self, args: &Value) -> Result<String, SkillError>;
}

#[async_trait]
impl<S: Skill> DynSkill for S {
    fn definition(&self) -> ToolDefinition {
        let schema = schemars::schema_for!(S::Input);
        let mut schema_value = serde_json::to_value(&schema).unwrap_or(Value::Null);
        // Anthropic ignores `$schema` and `title`, but stripping them keeps
        // the input_schema visually compact and matches Claude Code's
        // generated tool definitions.
        if let Value::Object(map) = &mut schema_value {
            map.remove("$schema");
            map.remove("title");
        }
        ToolDefinition {
            name: self.name().to_string(),
            description: self.description().to_string(),
            input_schema: schema_value,
        }
    }

    fn metadata(&self) -> SkillMetadata {
        SkillMetadata {
            is_read_only: self.is_read_only(),
            is_destructive: self.is_destructive(),
            is_concurrency_safe: self.is_concurrency_safe(),
        }
    }

    fn aliases(&self) -> Vec<&'static str> {
        Skill::aliases(self).to_vec()
    }

    fn search_hint(&self) -> Option<&str> {
        Skill::search_hint(self)
    }

    fn should_defer(&self) -> bool {
        Skill::should_defer(self)
    }

    async fn run_json(&self, args: &Value) -> Result<String, SkillError> {
        let input: S::Input = serde_json::from_value(args.clone())?;
        self.run(input).await
    }
}

/// Owns the set of skills available to an agent.
#[derive(Default)]
pub struct Registry {
    skills: Vec<Box<dyn DynSkill>>,
    /// Maps both primary names and aliases to the skill index.
    index: HashMap<String, usize>,
    hooks: Vec<Box<dyn SkillHook>>,
}

impl Registry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a skill. Later additions with the same name replace earlier ones.
    pub fn add<S: Skill>(&mut self, skill: S) -> &mut Self {
        let boxed: Box<dyn DynSkill> = Box::new(skill);
        let name = boxed.definition().name;
        let aliases = boxed.aliases();
        if let Some(&idx) = self.index.get(&name) {
            self.skills[idx] = boxed;
        } else {
            let new_idx = self.skills.len();
            self.index.insert(name, new_idx);
            for alias in &aliases {
                self.index.insert(alias.to_string(), new_idx);
            }
            self.skills.push(boxed);
        }
        self
    }

    /// Register a hook. Hooks run in registration order around every
    /// dispatch. The first hook that denies short-circuits the chain.
    pub fn with_hook(&mut self, hook: impl SkillHook) -> &mut Self {
        self.hooks.push(Box::new(hook));
        self
    }

    /// Tool definitions, ready for the system prompt or the provider call.
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.skills.iter().map(|s| s.definition()).collect()
    }

    /// Non-deferred tool definitions only. Use this when you support
    /// keyword discovery via `search`: the always-loaded set goes in
    /// the initial prompt, the deferred long tail surfaces on demand.
    pub fn loaded_definitions(&self) -> Vec<ToolDefinition> {
        self.skills
            .iter()
            .filter(|s| !s.should_defer())
            .map(|s| s.definition())
            .collect()
    }

    /// Look up skills by keyword. Matches the skill name, its aliases,
    /// and its `search_hint`. The result is the same `ToolDefinition`
    /// shape so the caller can hand it straight to the provider.
    pub fn search(&self, query: &str) -> Vec<ToolDefinition> {
        let q = query.to_ascii_lowercase();
        self.skills
            .iter()
            .filter(|s| {
                let def = s.definition();
                def.name.to_ascii_lowercase().contains(&q)
                    || s.aliases()
                        .iter()
                        .any(|a| a.to_ascii_lowercase().contains(&q))
                    || s.search_hint()
                        .map(|h| h.to_ascii_lowercase().contains(&q))
                        .unwrap_or(false)
            })
            .map(|s| s.definition())
            .collect()
    }

    pub fn names(&self) -> Vec<String> {
        self.skills.iter().map(|s| s.definition().name).collect()
    }

    pub fn metadata(&self, name: &str) -> Option<SkillMetadata> {
        self.index.get(name).map(|&i| self.skills[i].metadata())
    }

    /// Run one tool call. Returns the model-facing string on success,
    /// or a `SkillError` on failure (the agent typically renders the
    /// error back to the model as a `tool_result` with `is_error: true`).
    /// Pre/post hooks fire around the dispatch.
    pub async fn run(&self, name: &str, args: &Value) -> Result<String, SkillError> {
        let idx = self
            .index
            .get(name)
            .copied()
            .ok_or_else(|| SkillError::Tool(format!("unknown tool: {name}")))?;
        let skill = self.skills[idx].as_ref();
        let metadata = skill.metadata();

        let mut effective_args = args.clone();
        for hook in &self.hooks {
            match hook.before(name, metadata, &effective_args).await? {
                HookOutcome::Proceed => {}
                HookOutcome::Modify(v) => effective_args = v,
                HookOutcome::Deny(reason) => {
                    let denied: Result<String, SkillError> = Err(SkillError::Safety(reason));
                    return self.run_after_hooks(name, &effective_args, denied).await;
                }
                HookOutcome::Replace(out) => {
                    return self.run_after_hooks(name, &effective_args, Ok(out)).await;
                }
            }
        }
        let result = skill.run_json(&effective_args).await;
        self.run_after_hooks(name, &effective_args, result).await
    }

    async fn run_after_hooks(
        &self,
        name: &str,
        args: &Value,
        mut result: Result<String, SkillError>,
    ) -> Result<String, SkillError> {
        for hook in &self.hooks {
            result = hook.after(name, args, result).await;
        }
        result
    }

    /// Run several tool calls. Read-only / concurrency-safe calls run in
    /// parallel; the rest run sequentially in the order they were given.
    pub async fn run_many(
        &self,
        calls: &[(String, String, Value)],
    ) -> Vec<(String, Result<String, SkillError>)> {
        let mut parallel: Vec<(String, &dyn DynSkill, &Value)> = Vec::new();
        let mut serial: Vec<(String, &dyn DynSkill, &Value)> = Vec::new();
        for (id, name, args) in calls {
            let Some(&idx) = self.index.get(name) else {
                serial.push((id.clone(), MISSING_SKILL_PLACEHOLDER, args));
                continue;
            };
            let dyn_skill = self.skills[idx].as_ref();
            if dyn_skill.metadata().is_concurrency_safe {
                parallel.push((id.clone(), dyn_skill, args));
            } else {
                serial.push((id.clone(), dyn_skill, args));
            }
        }

        let parallel_futs = parallel.into_iter().map(|(id, sk, args)| async move {
            let r = sk.run_json(args).await;
            (id, r)
        });
        let mut results: Vec<(String, Result<String, SkillError>)> = join_all(parallel_futs).await;

        for (id, sk, args) in serial {
            let r = if std::ptr::addr_of!(*sk) as *const ()
                == std::ptr::addr_of!(*MISSING_SKILL_PLACEHOLDER) as *const ()
            {
                Err(SkillError::Tool("unknown tool".into()))
            } else {
                sk.run_json(args).await
            };
            results.push((id, r));
        }
        results
    }
}

// A placeholder used by run_many to flag missing-tool entries without
// allocating a separate enum. The dispatch loop checks pointer identity.
static MISSING_SKILL_PLACEHOLDER: &MissingSkill = &MissingSkill;
struct MissingSkill;
#[async_trait]
impl DynSkill for MissingSkill {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: String::new(),
            description: String::new(),
            input_schema: Value::Null,
        }
    }
    fn metadata(&self) -> SkillMetadata {
        SkillMetadata {
            is_read_only: false,
            is_destructive: false,
            is_concurrency_safe: false,
        }
    }
    fn aliases(&self) -> Vec<&'static str> {
        Vec::new()
    }
    fn search_hint(&self) -> Option<&str> {
        None
    }
    fn should_defer(&self) -> bool {
        false
    }
    async fn run_json(&self, _args: &Value) -> Result<String, SkillError> {
        Err(SkillError::Tool("unknown tool".into()))
    }
}

/// Exponential backoff retry, suited for transient failures from outside
/// systems (network blips, rate limits). Retries only on
/// `SkillError::Transient`.
#[derive(Debug, Clone, Copy)]
pub struct RetryPolicy {
    pub max_attempts: u32,
    pub initial_delay: Duration,
    pub backoff: f64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay: Duration::from_millis(100),
            backoff: 2.0,
        }
    }
}

pub async fn with_retry<F, Fut, T>(policy: RetryPolicy, mut op: F) -> Result<T, SkillError>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T, SkillError>>,
{
    let mut delay = policy.initial_delay;
    let mut last_err: Option<SkillError> = None;
    for attempt in 0..policy.max_attempts {
        match op().await {
            Ok(v) => return Ok(v),
            Err(e) if e.is_transient() && attempt + 1 < policy.max_attempts => {
                tokio::time::sleep(delay).await;
                delay = delay.mul_f64(policy.backoff);
                last_err = Some(e);
            }
            Err(e) => return Err(e),
        }
    }
    Err(last_err.unwrap_or_else(|| SkillError::Tool("retry exhausted".into())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::time::Instant;

    #[derive(Deserialize, JsonSchema)]
    struct EchoArgs {
        message: String,
    }

    struct Echo;

    #[async_trait]
    impl Skill for Echo {
        type Input = EchoArgs;
        fn name(&self) -> &str {
            "echo"
        }
        fn description(&self) -> &str {
            "Echo a message back to the user."
        }
        async fn run(&self, input: EchoArgs) -> Result<String, SkillError> {
            Ok(input.message)
        }
    }

    #[derive(Deserialize, JsonSchema)]
    struct SleepArgs {
        ms: u64,
    }

    struct Sleep;

    #[async_trait]
    impl Skill for Sleep {
        type Input = SleepArgs;
        fn name(&self) -> &str {
            "sleep"
        }
        fn description(&self) -> &str {
            "Sleep for the given number of milliseconds."
        }
        async fn run(&self, input: SleepArgs) -> Result<String, SkillError> {
            tokio::time::sleep(Duration::from_millis(input.ms)).await;
            Ok(format!("slept {}ms", input.ms))
        }
    }

    #[tokio::test]
    async fn registry_dispatches_by_name() {
        let mut reg = Registry::new();
        reg.add(Echo);

        let args = serde_json::json!({ "message": "hi" });
        let out = reg.run("echo", &args).await.unwrap();
        assert_eq!(out, "hi");
    }

    #[tokio::test]
    async fn unknown_tool_is_an_error() {
        let reg = Registry::new();
        let err = reg.run("missing", &Value::Null).await.unwrap_err();
        assert!(matches!(err, SkillError::Tool(_)));
    }

    #[tokio::test]
    async fn definitions_include_schema() {
        let mut reg = Registry::new();
        reg.add(Echo);
        let defs = reg.definitions();
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].name, "echo");
        let schema = &defs[0].input_schema;
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["message"].is_object());
    }

    #[tokio::test]
    async fn concurrency_safe_calls_run_in_parallel() {
        let mut reg = Registry::new();
        reg.add(Sleep);

        let calls = vec![
            ("a".into(), "sleep".into(), serde_json::json!({"ms": 80})),
            ("b".into(), "sleep".into(), serde_json::json!({"ms": 80})),
            ("c".into(), "sleep".into(), serde_json::json!({"ms": 80})),
        ];

        let start = Instant::now();
        let results = reg.run_many(&calls).await;
        let elapsed = start.elapsed();

        assert_eq!(results.len(), 3);
        for (_, r) in &results {
            assert!(r.is_ok());
        }
        // Parallel: ~80ms; serial would be ~240ms. Allow generous slack.
        assert!(
            elapsed < Duration::from_millis(200),
            "expected parallel execution, took {elapsed:?}"
        );
    }

    #[tokio::test]
    async fn with_retry_recovers_from_transient_failures() {
        let count = AtomicU32::new(0);
        let result = with_retry(RetryPolicy::default(), || async {
            let n = count.fetch_add(1, Ordering::SeqCst);
            if n < 2 {
                Err(SkillError::Transient(format!("attempt {n}")))
            } else {
                Ok("ok")
            }
        })
        .await
        .unwrap();
        assert_eq!(result, "ok");
        assert_eq!(count.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn with_retry_gives_up_on_non_transient() {
        let count = AtomicU32::new(0);
        let policy = RetryPolicy {
            max_attempts: 3,
            initial_delay: Duration::from_millis(1),
            backoff: 1.0,
        };
        let result: Result<&str, _> = with_retry(policy, || async {
            count.fetch_add(1, Ordering::SeqCst);
            Err(SkillError::Tool("nope".into()))
        })
        .await;
        assert!(result.is_err());
        assert_eq!(count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn metadata_is_propagated() {
        struct Destructive;
        #[async_trait]
        impl Skill for Destructive {
            type Input = EchoArgs;
            fn name(&self) -> &str {
                "rm"
            }
            fn description(&self) -> &str {
                "Delete something."
            }
            fn is_read_only(&self) -> bool {
                false
            }
            fn is_destructive(&self) -> bool {
                true
            }
            async fn run(&self, _: EchoArgs) -> Result<String, SkillError> {
                Ok("ok".into())
            }
        }

        let mut reg = Registry::new();
        reg.add(Destructive);
        let md = reg.metadata("rm").unwrap();
        assert!(!md.is_read_only);
        assert!(md.is_destructive);
        assert!(!md.is_concurrency_safe);
    }

    #[tokio::test]
    async fn run_many_unknown_tool_yields_per_call_error() {
        let mut reg = Registry::new();
        reg.add(Echo);
        let calls = vec![
            (
                "a".into(),
                "echo".into(),
                serde_json::json!({"message": "hi"}),
            ),
            ("b".into(), "missing".into(), serde_json::json!({})),
        ];
        let results = reg.run_many(&calls).await;
        // Order: parallel (echo) finishes first; serial (missing) second.
        let by_id: HashMap<&str, &Result<String, SkillError>> =
            results.iter().map(|(id, r)| (id.as_str(), r)).collect();
        assert!(by_id["a"].is_ok());
        assert!(matches!(by_id["b"], Err(SkillError::Tool(_))));
        let _ = Mutex::new(()); // keep std::sync used
    }

    // --------- Hook + discovery tests ---------

    #[derive(Deserialize, JsonSchema)]
    struct RmArgs {
        path: String,
    }
    struct Rm;
    #[async_trait]
    impl Skill for Rm {
        type Input = RmArgs;
        fn name(&self) -> &str {
            "delete_file"
        }
        fn aliases(&self) -> &[&'static str] {
            &["rm"]
        }
        fn description(&self) -> &str {
            "Delete a file (irreversible)."
        }
        fn is_read_only(&self) -> bool {
            false
        }
        fn is_destructive(&self) -> bool {
            true
        }
        fn search_hint(&self) -> Option<&str> {
            Some("remove unlink trash")
        }
        async fn run(&self, _: RmArgs) -> Result<String, SkillError> {
            Ok("deleted".into())
        }
    }

    #[tokio::test]
    async fn aliases_resolve_to_the_same_skill() {
        let mut reg = Registry::new();
        reg.add(Rm);
        let primary = reg
            .run("delete_file", &serde_json::json!({ "path": "x" }))
            .await
            .unwrap();
        let alias = reg
            .run("rm", &serde_json::json!({ "path": "x" }))
            .await
            .unwrap();
        assert_eq!(primary, alias);
    }

    #[tokio::test]
    async fn search_matches_hint_and_alias() {
        let mut reg = Registry::new();
        reg.add(Rm);
        assert_eq!(reg.search("unlink").len(), 1);
        assert_eq!(reg.search("rm").len(), 1);
        assert_eq!(reg.search("nothing-matches").len(), 0);
    }

    struct Deferred;
    #[derive(Deserialize, JsonSchema)]
    struct DeferredArgs {}
    #[async_trait]
    impl Skill for Deferred {
        type Input = DeferredArgs;
        fn name(&self) -> &str {
            "rare_capability"
        }
        fn description(&self) -> &str {
            "Something rarely needed."
        }
        fn should_defer(&self) -> bool {
            true
        }
        async fn run(&self, _: DeferredArgs) -> Result<String, SkillError> {
            Ok("ran".into())
        }
    }

    #[tokio::test]
    async fn loaded_definitions_omit_deferred_skills() {
        let mut reg = Registry::new();
        reg.add(Echo).add(Deferred);
        let loaded: Vec<_> = reg
            .loaded_definitions()
            .into_iter()
            .map(|d| d.name)
            .collect();
        let all: Vec<_> = reg.definitions().into_iter().map(|d| d.name).collect();
        assert!(loaded.contains(&"echo".to_string()));
        assert!(!loaded.contains(&"rare_capability".to_string()));
        assert!(all.contains(&"rare_capability".to_string()));
    }

    #[tokio::test]
    async fn pre_hook_can_deny_destructive_call() {
        let mut reg = Registry::new();
        reg.add(Rm)
            .with_hook(PermissionGate::new(|_, md: SkillMetadata| {
                !md.is_destructive
            }));
        let err = reg
            .run("delete_file", &serde_json::json!({ "path": "x" }))
            .await
            .unwrap_err();
        assert!(matches!(err, SkillError::Safety(_)));
    }

    #[tokio::test]
    async fn pre_hook_can_replace_output() {
        struct CacheHook;
        #[async_trait]
        impl SkillHook for CacheHook {
            async fn before(
                &self,
                name: &str,
                _md: SkillMetadata,
                _args: &Value,
            ) -> Result<HookOutcome, SkillError> {
                if name == "echo" {
                    Ok(HookOutcome::Replace("cached".into()))
                } else {
                    Ok(HookOutcome::Proceed)
                }
            }
        }
        let mut reg = Registry::new();
        reg.add(Echo).with_hook(CacheHook);
        let out = reg
            .run("echo", &serde_json::json!({ "message": "live" }))
            .await
            .unwrap();
        assert_eq!(out, "cached");
    }

    #[tokio::test]
    async fn pre_hook_can_rewrite_args() {
        struct InjectPrefix;
        #[async_trait]
        impl SkillHook for InjectPrefix {
            async fn before(
                &self,
                _name: &str,
                _md: SkillMetadata,
                args: &Value,
            ) -> Result<HookOutcome, SkillError> {
                let mut new_args = args.clone();
                if let Some(s) = args.get("message").and_then(Value::as_str) {
                    let injected = format!("[redacted] {s}");
                    new_args["message"] = Value::String(injected);
                }
                Ok(HookOutcome::Modify(new_args))
            }
        }
        let mut reg = Registry::new();
        reg.add(Echo).with_hook(InjectPrefix);
        let out = reg
            .run("echo", &serde_json::json!({ "message": "secret" }))
            .await
            .unwrap();
        assert_eq!(out, "[redacted] secret");
    }
}
