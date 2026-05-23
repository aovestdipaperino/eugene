//! Anthropic Messages API adapter.

use async_trait::async_trait;
use eugene_core::ToolDefinition;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    CompletionRequest, CompletionResponse, MessageBlock, ModelInfo, ModelTier, Provider,
    ProviderError, Role, StopReason, Usage, check_status, env_or, map_reqwest,
};

const ANTHROPIC_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";

pub struct AnthropicProvider {
    http: reqwest::Client,
    api_key: String,
    model: String,
    base_url: String,
}

impl AnthropicProvider {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            api_key: api_key.into(),
            model: model.into(),
            base_url: ANTHROPIC_URL.into(),
        }
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Reads `ANTHROPIC_API_KEY` and the optional `EUGENE_ANTHROPIC_MODEL`
    /// (default: `claude-sonnet-4-6`).
    pub fn from_env() -> Result<Self, ProviderError> {
        let api_key = env_or("ANTHROPIC_API_KEY")?;
        let model =
            std::env::var("EUGENE_ANTHROPIC_MODEL").unwrap_or_else(|_| "claude-sonnet-4-6".into());
        Ok(Self::new(api_key, model))
    }
}

#[async_trait]
impl Provider for AnthropicProvider {
    fn name(&self) -> &str {
        "anthropic"
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn model_info(&self) -> Option<ModelInfo> {
        let (tier, in_, out) = match self.model.as_str() {
            m if m.contains("haiku") => (ModelTier::Fast, 0.80, 4.00),
            m if m.contains("opus") => (ModelTier::Smart, 15.0, 75.0),
            _ => (ModelTier::Balanced, 3.0, 15.0),
        };
        Some(ModelInfo {
            id: self.model.clone(),
            tier,
            price_in_per_1k: Some(in_ / 1000.0),
            price_out_per_1k: Some(out / 1000.0),
            knowledge_cutoff: None,
        })
    }

    async fn complete(
        &self,
        request: &CompletionRequest,
    ) -> Result<CompletionResponse, ProviderError> {
        let body = AnthropicRequest::from(request, &self.model);
        let resp = self
            .http
            .post(&self.base_url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .json(&body)
            .send()
            .await
            .map_err(map_reqwest)?;
        let resp = check_status(resp).await?;
        let parsed: AnthropicResponse = resp
            .json()
            .await
            .map_err(|e| ProviderError::Decode(e.to_string()))?;
        Ok(parsed.into())
    }
}

// ---------------------------------------------------------------------------
// Wire types
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct AnthropicRequest<'a> {
    model: &'a str,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    system: Vec<AnthSysBlock>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<AnthTool>,
    messages: Vec<AnthMessage>,
}

#[derive(Serialize)]
struct AnthSysBlock {
    #[serde(rename = "type")]
    kind: &'static str,
    text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_control: Option<AnthCacheControl>,
}

#[derive(Serialize)]
struct AnthCacheControl {
    #[serde(rename = "type")]
    kind: &'static str,
}

#[derive(Serialize)]
struct AnthTool {
    name: String,
    description: String,
    input_schema: Value,
}

impl From<&ToolDefinition> for AnthTool {
    fn from(t: &ToolDefinition) -> Self {
        Self {
            name: t.name.clone(),
            description: t.description.clone(),
            input_schema: t.input_schema.clone(),
        }
    }
}

#[derive(Serialize)]
struct AnthMessage {
    role: &'static str,
    content: Vec<AnthBlock>,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthBlock {
    Text {
        text: String,
    },
    ToolUse {
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

#[derive(Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthBlock>,
    stop_reason: Option<String>,
    usage: Option<AnthUsage>,
}

#[derive(Deserialize)]
struct AnthUsage {
    #[serde(default)]
    input_tokens: u32,
    #[serde(default)]
    output_tokens: u32,
    #[serde(default)]
    cache_creation_input_tokens: u32,
    #[serde(default)]
    cache_read_input_tokens: u32,
}

impl<'a> AnthropicRequest<'a> {
    fn from(req: &CompletionRequest, model: &'a str) -> Self {
        let system: Vec<AnthSysBlock> = req
            .system
            .iter()
            .map(|b| AnthSysBlock {
                kind: "text",
                text: b.text.clone(),
                cache_control: if b.cache_break {
                    Some(AnthCacheControl { kind: "ephemeral" })
                } else {
                    None
                },
            })
            .collect();

        let tools = req.tools.iter().map(AnthTool::from).collect();
        let messages = req
            .messages
            .iter()
            .map(|m| AnthMessage {
                role: match m.role {
                    Role::User => "user",
                    Role::Assistant => "assistant",
                },
                content: m
                    .content
                    .iter()
                    .map(|b| match b {
                        MessageBlock::Text { text } => AnthBlock::Text { text: text.clone() },
                        MessageBlock::ToolCall { id, name, input } => AnthBlock::ToolUse {
                            id: id.clone(),
                            name: name.clone(),
                            input: input.clone(),
                        },
                        MessageBlock::ToolResult {
                            tool_use_id,
                            content,
                            is_error,
                        } => AnthBlock::ToolResult {
                            tool_use_id: tool_use_id.clone(),
                            content: content.clone(),
                            is_error: *is_error,
                        },
                    })
                    .collect(),
            })
            .collect();

        Self {
            model,
            max_tokens: req.max_tokens,
            temperature: req.temperature,
            system,
            tools,
            messages,
        }
    }
}

#[cfg(test)]
mod translation_tests {
    use super::*;
    use crate::{CompletionRequest, Message, MessageBlock, SystemBlock};

    #[test]
    fn cache_break_emits_cache_control() {
        let req = CompletionRequest {
            system: vec![
                SystemBlock::static_("persona"),
                SystemBlock::dynamic("today: 2026-05-23"),
            ],
            messages: vec![Message::user_text("hi")],
            ..CompletionRequest::new()
        };
        let body = AnthropicRequest::from(&req, "claude-x");
        let json = serde_json::to_value(&body).unwrap();
        let first = &json["system"][0];
        let second = &json["system"][1];
        assert_eq!(first["cache_control"]["type"], "ephemeral");
        assert!(second["cache_control"].is_null());
    }

    #[test]
    fn tool_result_lands_on_user_role() {
        let req = CompletionRequest {
            messages: vec![Message::user_tool_results(vec![MessageBlock::ToolResult {
                tool_use_id: "toolu_x".into(),
                content: "ok".into(),
                is_error: false,
            }])],
            ..CompletionRequest::new()
        };
        let body = AnthropicRequest::from(&req, "claude-x");
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(json["messages"][0]["role"], "user");
        assert_eq!(json["messages"][0]["content"][0]["type"], "tool_result");
        assert_eq!(json["messages"][0]["content"][0]["tool_use_id"], "toolu_x");
    }

    #[test]
    fn assistant_tool_use_blocks_survive_round_trip() {
        let req = CompletionRequest {
            messages: vec![Message::assistant_blocks(vec![
                MessageBlock::Text {
                    text: "Reading ...".into(),
                },
                MessageBlock::ToolCall {
                    id: "toolu_1".into(),
                    name: "read_file".into(),
                    input: serde_json::json!({"path": "Cargo.toml"}),
                },
            ])],
            ..CompletionRequest::new()
        };
        let body = AnthropicRequest::from(&req, "claude-x");
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(json["messages"][0]["role"], "assistant");
        assert_eq!(json["messages"][0]["content"][1]["type"], "tool_use");
        assert_eq!(json["messages"][0]["content"][1]["name"], "read_file");
        assert_eq!(
            json["messages"][0]["content"][1]["input"]["path"],
            "Cargo.toml"
        );
    }
}

impl From<AnthropicResponse> for CompletionResponse {
    fn from(r: AnthropicResponse) -> Self {
        let content = r
            .content
            .into_iter()
            .map(|b| match b {
                AnthBlock::Text { text } => MessageBlock::Text { text },
                AnthBlock::ToolUse { id, name, input } => {
                    MessageBlock::ToolCall { id, name, input }
                }
                AnthBlock::ToolResult {
                    tool_use_id,
                    content,
                    is_error,
                } => MessageBlock::ToolResult {
                    tool_use_id,
                    content,
                    is_error,
                },
            })
            .collect();

        let stop_reason = match r.stop_reason.as_deref() {
            Some("end_turn") => StopReason::EndTurn,
            Some("tool_use") => StopReason::ToolUse,
            Some("max_tokens") => StopReason::MaxTokens,
            _ => StopReason::Other,
        };

        let usage = r
            .usage
            .map(|u| Usage {
                input_tokens: u.input_tokens,
                output_tokens: u.output_tokens,
                cache_read: u.cache_read_input_tokens,
                cache_create: u.cache_creation_input_tokens,
            })
            .unwrap_or_default();

        CompletionResponse {
            content,
            stop_reason,
            usage,
        }
    }
}
