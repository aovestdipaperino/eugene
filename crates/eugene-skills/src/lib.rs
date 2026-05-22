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

    /// Run the tool. The output is whatever string the model should see.
    /// Structured results travel as JSON inside this string.
    async fn run(&self, input: Self::Input) -> Result<String, SkillError>;
}

/// Object-safe wrapper so the registry can hold heterogeneous skills.
#[async_trait]
pub trait DynSkill: Send + Sync {
    fn definition(&self) -> ToolDefinition;
    fn metadata(&self) -> SkillMetadata;
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

    async fn run_json(&self, args: &Value) -> Result<String, SkillError> {
        let input: S::Input = serde_json::from_value(args.clone())?;
        self.run(input).await
    }
}

/// Owns the set of skills available to an agent.
#[derive(Default)]
pub struct Registry {
    skills: Vec<Box<dyn DynSkill>>,
    index: HashMap<String, usize>,
}

impl Registry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a skill. Later additions with the same name replace earlier ones.
    pub fn add<S: Skill>(&mut self, skill: S) -> &mut Self {
        let boxed: Box<dyn DynSkill> = Box::new(skill);
        let name = boxed.definition().name;
        if let Some(&idx) = self.index.get(&name) {
            self.skills[idx] = boxed;
        } else {
            self.index.insert(name, self.skills.len());
            self.skills.push(boxed);
        }
        self
    }

    /// Tool definitions, ready for the system prompt or the provider call.
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.skills.iter().map(|s| s.definition()).collect()
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
    pub async fn run(&self, name: &str, args: &Value) -> Result<String, SkillError> {
        let idx = self
            .index
            .get(name)
            .copied()
            .ok_or_else(|| SkillError::Tool(format!("unknown tool: {name}")))?;
        self.skills[idx].run_json(args).await
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
}
