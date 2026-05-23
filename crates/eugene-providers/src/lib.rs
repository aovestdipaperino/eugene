//! Provider trait and adapters.
//!
//! An agent's tool-calling loop has the same shape regardless of which
//! LLM provider sits behind it. The [`Provider`] trait normalises that
//! shape: a single `complete` method takes provider-neutral types
//! ([`Message`], [`Tool`], [`SystemBlock`]) and returns a
//! [`CompletionResponse`] in the same shape. Adapters translate to and
//! from each vendor's wire format.
//!
//! Two adapters ship with this crate: [`anthropic::AnthropicProvider`]
//! against the Messages API, and [`openai::OpenAiProvider`] against the
//! Chat Completions API. The OpenAI adapter doubles as the Ollama
//! adapter when its `base_url` is repointed at a local Ollama server.
//!
//! The trait does not cover every nuance of every API. It covers what
//! an agent loop actually uses: system prompt with optional cache
//! markers, tool definitions, message history, tool-call extraction,
//! and basic usage accounting. Provider-specific knobs (thinking
//! budgets, vision, streaming) belong to the concrete adapter.

use std::collections::HashMap;

use async_trait::async_trait;
use eugene_core::ToolDefinition;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

pub mod anthropic;
pub mod openai;

pub use anthropic::AnthropicProvider;
pub use openai::OpenAiProvider;

#[derive(Debug, Error)]
pub enum ProviderError {
    #[error("transport error: {0}")]
    Transport(String),

    #[error("API returned {status}: {body}")]
    Http { status: u16, body: String },

    #[error("decode error: {0}")]
    Decode(String),

    #[error("unsupported: {0}")]
    Unsupported(String),

    #[error("missing config: {0}")]
    MissingConfig(String),
}

impl ProviderError {
    /// 429 and 5xx are usually transient. Callers can pair this with
    /// the retry helper in `eugene-skills`.
    pub fn is_transient(&self) -> bool {
        matches!(self, ProviderError::Http { status, .. } if *status == 429 || *status >= 500)
            || matches!(self, ProviderError::Transport(_))
    }
}

/// A typed tier so callers can ask for "fast and cheap" or "smart and
/// expensive" without knowing the model id. The same idea Claude Code's
/// `getSmallFastModel()` captures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelTier {
    /// Cheap, fast, low-stakes: classification, routing, summarisation.
    Fast,
    /// The default working horse.
    Balanced,
    /// Expensive and slow but more accurate.
    Smart,
}

/// Top-of-the-cone description of a model that helps callers reason
/// about cost and fit.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub id: String,
    pub tier: ModelTier,
    /// Approximate USD price per 1K input tokens, when known.
    pub price_in_per_1k: Option<f64>,
    /// Approximate USD price per 1K output tokens, when known.
    pub price_out_per_1k: Option<f64>,
    /// ISO date string of the training cutoff, when known.
    pub knowledge_cutoff: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

/// Provider-neutral content block. Both inbound (request) and outbound
/// (response) messages use this enum.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MessageBlock {
    Text {
        text: String,
    },
    ToolCall {
        id: String,
        name: String,
        input: Value,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        is_error: bool,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Vec<MessageBlock>,
}

impl Message {
    pub fn user_text(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: vec![MessageBlock::Text { text: text.into() }],
        }
    }
    pub fn user_tool_results(blocks: Vec<MessageBlock>) -> Self {
        Self {
            role: Role::User,
            content: blocks,
        }
    }
    pub fn assistant_blocks(blocks: Vec<MessageBlock>) -> Self {
        Self {
            role: Role::Assistant,
            content: blocks,
        }
    }
}

/// One block of the system prompt with an optional cache marker. Maps
/// to Anthropic's `system: [{type: "text", text, cache_control}]` and
/// to the OpenAI system role's content (cache marker silently dropped).
#[derive(Debug, Clone, Default)]
pub struct SystemBlock {
    pub text: String,
    pub cache_break: bool,
}

impl SystemBlock {
    pub fn static_(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            cache_break: true,
        }
    }
    pub fn dynamic(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            cache_break: false,
        }
    }
}

/// A request to the provider.
#[derive(Debug, Clone, Default)]
pub struct CompletionRequest {
    pub system: Vec<SystemBlock>,
    pub tools: Vec<ToolDefinition>,
    pub messages: Vec<Message>,
    pub max_tokens: u32,
    pub temperature: Option<f32>,
}

impl CompletionRequest {
    pub fn new() -> Self {
        Self {
            max_tokens: 4096,
            ..Default::default()
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
    Other,
}

#[derive(Debug, Clone, Default)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub cache_read: u32,
    pub cache_create: u32,
}

#[derive(Debug, Clone)]
pub struct CompletionResponse {
    pub content: Vec<MessageBlock>,
    pub stop_reason: StopReason,
    pub usage: Usage,
}

impl CompletionResponse {
    /// Convenience: collect just the tool calls.
    pub fn tool_calls(&self) -> Vec<(&str, &str, &Value)> {
        self.content
            .iter()
            .filter_map(|b| match b {
                MessageBlock::ToolCall { id, name, input } => {
                    Some((id.as_str(), name.as_str(), input))
                }
                _ => None,
            })
            .collect()
    }

    /// Convenience: collect just the text blocks, joined.
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|b| match b {
                MessageBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Provider abstraction. Implementations translate between the
/// provider-neutral types in this crate and a vendor's wire format.
#[async_trait]
pub trait Provider: Send + Sync + 'static {
    fn name(&self) -> &str;
    fn model(&self) -> &str;

    /// Information about the model for cost-aware callers. Default
    /// returns `None` so adapters can stay minimal.
    fn model_info(&self) -> Option<ModelInfo> {
        None
    }

    async fn complete(
        &self,
        request: &CompletionRequest,
    ) -> Result<CompletionResponse, ProviderError>;
}

/// Look up an environment variable, returning a `MissingConfig` error
/// when absent. Used by the from-env constructors of adapters.
pub fn env_or(name: &str) -> Result<String, ProviderError> {
    std::env::var(name).map_err(|_| ProviderError::MissingConfig(name.to_string()))
}

/// Build a provider from environment variables. `EUGENE_PROVIDER`
/// picks the adapter (`anthropic` (default) | `openai` | `ollama`),
/// and each adapter reads its own API key / base URL from env.
pub fn provider_from_env() -> Result<Box<dyn Provider>, ProviderError> {
    let kind = std::env::var("EUGENE_PROVIDER")
        .unwrap_or_else(|_| "anthropic".into())
        .to_ascii_lowercase();
    match kind.as_str() {
        "anthropic" => Ok(Box::new(AnthropicProvider::from_env()?)),
        "openai" => Ok(Box::new(OpenAiProvider::from_env()?)),
        "ollama" => Ok(Box::new(OpenAiProvider::ollama_from_env()?)),
        other => Err(ProviderError::Unsupported(other.into())),
    }
}

// Allow Provider implementors to map ad-hoc reqwest errors uniformly.
pub(crate) fn map_reqwest(err: reqwest::Error) -> ProviderError {
    ProviderError::Transport(err.to_string())
}

// Allow Provider implementors to surface HTTP failures uniformly.
pub(crate) async fn check_status(
    resp: reqwest::Response,
) -> Result<reqwest::Response, ProviderError> {
    let status = resp.status();
    if status.is_success() {
        Ok(resp)
    } else {
        let body = resp.text().await.unwrap_or_default();
        Err(ProviderError::Http {
            status: status.as_u16(),
            body,
        })
    }
}

// Keep std::collections::HashMap used so future provider configs can
// share it without re-declaring; harmless at the crate level.
#[doc(hidden)]
pub(crate) fn _hashmap_used() -> HashMap<(), ()> {
    HashMap::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completion_response_extracts_text_and_calls() {
        let resp = CompletionResponse {
            content: vec![
                MessageBlock::Text {
                    text: "hello".into(),
                },
                MessageBlock::ToolCall {
                    id: "t1".into(),
                    name: "read_file".into(),
                    input: serde_json::json!({"path": "x"}),
                },
                MessageBlock::Text {
                    text: "world".into(),
                },
            ],
            stop_reason: StopReason::ToolUse,
            usage: Usage::default(),
        };
        assert_eq!(resp.text(), "hello\nworld");
        assert_eq!(resp.tool_calls().len(), 1);
        assert_eq!(resp.tool_calls()[0].1, "read_file");
    }

    #[test]
    fn transient_classification() {
        assert!(ProviderError::Transport("x".into()).is_transient());
        assert!(
            ProviderError::Http {
                status: 503,
                body: "".into()
            }
            .is_transient()
        );
        assert!(
            ProviderError::Http {
                status: 429,
                body: "".into()
            }
            .is_transient()
        );
        assert!(
            !ProviderError::Http {
                status: 400,
                body: "".into()
            }
            .is_transient()
        );
    }

    #[test]
    fn provider_from_env_defaults_to_anthropic() {
        // SAFETY: tests run single-threaded for this module by default;
        // but Rust nightly may parallelise. We guard with explicit
        // unset/restore to keep this hermetic.
        let prev = std::env::var("EUGENE_PROVIDER").ok();
        let prev_key = std::env::var("ANTHROPIC_API_KEY").ok();
        unsafe {
            std::env::remove_var("EUGENE_PROVIDER");
            std::env::set_var("ANTHROPIC_API_KEY", "test-key");
        }
        let p = provider_from_env().unwrap();
        assert_eq!(p.name(), "anthropic");
        unsafe {
            if let Some(v) = prev {
                std::env::set_var("EUGENE_PROVIDER", v);
            } else {
                std::env::remove_var("EUGENE_PROVIDER");
            }
            if let Some(v) = prev_key {
                std::env::set_var("ANTHROPIC_API_KEY", v);
            } else {
                std::env::remove_var("ANTHROPIC_API_KEY");
            }
        }
    }
}
