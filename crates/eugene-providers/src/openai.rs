//! OpenAI Chat Completions adapter.
//!
//! Doubles as an Ollama adapter: point `base_url` at a local Ollama
//! server (default `http://localhost:11434/v1/chat/completions`) and
//! the same shape works because Ollama exposes an OpenAI-compatible
//! endpoint.

use async_trait::async_trait;
use eugene_core::ToolDefinition;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    CompletionRequest, CompletionResponse, MessageBlock, ModelInfo, ModelTier, Provider,
    ProviderError, Role, StopReason, Usage, check_status, env_or, map_reqwest,
};

const OPENAI_URL: &str = "https://api.openai.com/v1/chat/completions";

pub struct OpenAiProvider {
    http: reqwest::Client,
    api_key: Option<String>,
    model: String,
    base_url: String,
    flavour: Flavour,
}

#[derive(Clone, Copy)]
enum Flavour {
    OpenAi,
    Ollama,
}

impl OpenAiProvider {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            api_key: Some(api_key.into()),
            model: model.into(),
            base_url: OPENAI_URL.into(),
            flavour: Flavour::OpenAi,
        }
    }

    /// Ollama doesn't need an API key. Point `base_url` at the local
    /// server's `/v1/chat/completions` route.
    pub fn ollama(model: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            api_key: None,
            model: model.into(),
            base_url: base_url.into(),
            flavour: Flavour::Ollama,
        }
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Reads `OPENAI_API_KEY` and `EUGENE_OPENAI_MODEL`
    /// (default: `gpt-4o-mini`).
    pub fn from_env() -> Result<Self, ProviderError> {
        let api_key = env_or("OPENAI_API_KEY")?;
        let model = std::env::var("EUGENE_OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o-mini".into());
        Ok(Self::new(api_key, model))
    }

    /// Reads `EUGENE_OLLAMA_BASE_URL` (default `http://localhost:11434`)
    /// and `EUGENE_OLLAMA_MODEL` (default `llama3.2`).
    pub fn ollama_from_env() -> Result<Self, ProviderError> {
        let base = std::env::var("EUGENE_OLLAMA_BASE_URL")
            .unwrap_or_else(|_| "http://localhost:11434".into());
        let url = format!("{}/v1/chat/completions", base.trim_end_matches('/'));
        let model = std::env::var("EUGENE_OLLAMA_MODEL").unwrap_or_else(|_| "llama3.2".into());
        Ok(Self::ollama(model, url))
    }
}

#[async_trait]
impl Provider for OpenAiProvider {
    fn name(&self) -> &str {
        match self.flavour {
            Flavour::OpenAi => "openai",
            Flavour::Ollama => "ollama",
        }
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn model_info(&self) -> Option<ModelInfo> {
        let (tier, in_, out) = match self.flavour {
            Flavour::Ollama => (ModelTier::Fast, 0.0, 0.0),
            Flavour::OpenAi => match self.model.as_str() {
                m if m.contains("mini") || m.contains("4.1-nano") => (ModelTier::Fast, 0.15, 0.60),
                m if m.contains("o1") || m.contains("4.1") => (ModelTier::Smart, 10.0, 30.0),
                _ => (ModelTier::Balanced, 2.5, 10.0),
            },
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
        let body = OaRequest::from(request, &self.model);
        let mut builder = self.http.post(&self.base_url).json(&body);
        if let Some(key) = &self.api_key {
            builder = builder.bearer_auth(key);
        }
        let resp = builder.send().await.map_err(map_reqwest)?;
        let resp = check_status(resp).await?;
        let parsed: OaResponse = resp
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
struct OaRequest<'a> {
    model: &'a str,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    messages: Vec<OaMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<OaTool>,
}

#[derive(Serialize)]
struct OaMessage {
    role: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tool_calls: Vec<OaToolCall>,
}

#[derive(Serialize)]
struct OaToolCall {
    id: String,
    #[serde(rename = "type")]
    kind: &'static str,
    function: OaToolCallFunction,
}

#[derive(Serialize)]
struct OaToolCallFunction {
    name: String,
    /// OpenAI sends arguments as a JSON-encoded string.
    arguments: String,
}

#[derive(Serialize)]
struct OaTool {
    #[serde(rename = "type")]
    kind: &'static str,
    function: OaToolFunction,
}

#[derive(Serialize)]
struct OaToolFunction {
    name: String,
    description: String,
    parameters: Value,
}

impl From<&ToolDefinition> for OaTool {
    fn from(t: &ToolDefinition) -> Self {
        Self {
            kind: "function",
            function: OaToolFunction {
                name: t.name.clone(),
                description: t.description.clone(),
                parameters: t.input_schema.clone(),
            },
        }
    }
}

#[derive(Deserialize)]
struct OaResponse {
    choices: Vec<OaChoice>,
    #[serde(default)]
    usage: Option<OaUsage>,
}

#[derive(Deserialize)]
struct OaChoice {
    message: OaResponseMessage,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct OaResponseMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<OaResponseToolCall>,
}

#[derive(Deserialize)]
struct OaResponseToolCall {
    id: String,
    #[serde(default)]
    function: OaResponseFunction,
}

#[derive(Deserialize, Default)]
struct OaResponseFunction {
    name: String,
    arguments: String,
}

#[derive(Deserialize)]
struct OaUsage {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
}

impl<'a> OaRequest<'a> {
    fn from(req: &CompletionRequest, model: &'a str) -> Self {
        let mut messages: Vec<OaMessage> = Vec::new();

        // OpenAI's system message is a single string. Flatten the cache
        // blocks; the cache_control marker is silently dropped (the API
        // doesn't support per-block caching at this endpoint).
        if !req.system.is_empty() {
            let combined = req
                .system
                .iter()
                .map(|b| b.text.as_str())
                .collect::<Vec<_>>()
                .join("\n\n");
            messages.push(OaMessage {
                role: "system",
                content: Some(combined),
                tool_call_id: None,
                tool_calls: Vec::new(),
            });
        }

        for m in &req.messages {
            // Split each provider-neutral message into the assistant/tool
            // shape OpenAI expects. A user message with ToolResult blocks
            // becomes one OpenAI `role: "tool"` message per result.
            match m.role {
                Role::User => {
                    let mut text_parts: Vec<&str> = Vec::new();
                    for b in &m.content {
                        match b {
                            MessageBlock::Text { text } => text_parts.push(text),
                            MessageBlock::ToolResult {
                                tool_use_id,
                                content,
                                ..
                            } => {
                                messages.push(OaMessage {
                                    role: "tool",
                                    content: Some(content.clone()),
                                    tool_call_id: Some(tool_use_id.clone()),
                                    tool_calls: Vec::new(),
                                });
                            }
                            MessageBlock::ToolCall { .. } => {
                                // Should not appear on user-role messages.
                            }
                        }
                    }
                    if !text_parts.is_empty() {
                        messages.push(OaMessage {
                            role: "user",
                            content: Some(text_parts.join("\n")),
                            tool_call_id: None,
                            tool_calls: Vec::new(),
                        });
                    }
                }
                Role::Assistant => {
                    let mut text_parts: Vec<&str> = Vec::new();
                    let mut tool_calls: Vec<OaToolCall> = Vec::new();
                    for b in &m.content {
                        match b {
                            MessageBlock::Text { text } => text_parts.push(text),
                            MessageBlock::ToolCall { id, name, input } => {
                                tool_calls.push(OaToolCall {
                                    id: id.clone(),
                                    kind: "function",
                                    function: OaToolCallFunction {
                                        name: name.clone(),
                                        arguments: input.to_string(),
                                    },
                                });
                            }
                            MessageBlock::ToolResult { .. } => {
                                // Should not appear on assistant-role messages.
                            }
                        }
                    }
                    let content = if text_parts.is_empty() {
                        None
                    } else {
                        Some(text_parts.join("\n"))
                    };
                    messages.push(OaMessage {
                        role: "assistant",
                        content,
                        tool_call_id: None,
                        tool_calls,
                    });
                }
            }
        }

        Self {
            model,
            max_tokens: req.max_tokens,
            temperature: req.temperature,
            messages,
            tools: req.tools.iter().map(OaTool::from).collect(),
        }
    }
}

#[cfg(test)]
mod translation_tests {
    use super::*;
    use crate::{CompletionRequest, Message, MessageBlock, SystemBlock};

    #[test]
    fn cache_blocks_collapse_into_single_system_message() {
        let req = CompletionRequest {
            system: vec![
                SystemBlock::static_("persona"),
                SystemBlock::dynamic("today: 2026-05-23"),
            ],
            messages: vec![Message::user_text("hi")],
            ..CompletionRequest::new()
        };
        let body = OaRequest::from(&req, "gpt-4o-mini");
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(json["messages"][0]["role"], "system");
        assert!(
            json["messages"][0]["content"]
                .as_str()
                .unwrap()
                .contains("persona")
        );
        assert!(
            json["messages"][0]["content"]
                .as_str()
                .unwrap()
                .contains("today: 2026-05-23")
        );
    }

    #[test]
    fn tool_result_becomes_tool_role_message() {
        let req = CompletionRequest {
            messages: vec![Message::user_tool_results(vec![MessageBlock::ToolResult {
                tool_use_id: "call_x".into(),
                content: "ok".into(),
                is_error: false,
            }])],
            ..CompletionRequest::new()
        };
        let body = OaRequest::from(&req, "gpt-4o-mini");
        let json = serde_json::to_value(&body).unwrap();
        let msg = &json["messages"][0];
        assert_eq!(msg["role"], "tool");
        assert_eq!(msg["tool_call_id"], "call_x");
        assert_eq!(msg["content"], "ok");
    }

    #[test]
    fn assistant_tool_call_uses_arguments_as_string() {
        let req = CompletionRequest {
            messages: vec![Message::assistant_blocks(vec![MessageBlock::ToolCall {
                id: "call_1".into(),
                name: "read_file".into(),
                input: serde_json::json!({"path": "Cargo.toml"}),
            }])],
            ..CompletionRequest::new()
        };
        let body = OaRequest::from(&req, "gpt-4o-mini");
        let json = serde_json::to_value(&body).unwrap();
        let msg = &json["messages"][0];
        assert_eq!(msg["role"], "assistant");
        let tc = &msg["tool_calls"][0];
        assert_eq!(tc["type"], "function");
        // OpenAI requires arguments as a JSON-encoded string.
        let args = tc["function"]["arguments"].as_str().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(args).unwrap();
        assert_eq!(parsed["path"], "Cargo.toml");
    }

    #[test]
    fn tool_definitions_wrap_in_function_envelope() {
        use eugene_core::ToolDefinition;
        let req = CompletionRequest {
            tools: vec![ToolDefinition {
                name: "read_file".into(),
                description: "Read.".into(),
                input_schema: serde_json::json!({"type": "object"}),
            }],
            messages: vec![Message::user_text("hi")],
            ..CompletionRequest::new()
        };
        let body = OaRequest::from(&req, "gpt-4o-mini");
        let json = serde_json::to_value(&body).unwrap();
        let tool = &json["tools"][0];
        assert_eq!(tool["type"], "function");
        assert_eq!(tool["function"]["name"], "read_file");
    }
}

impl From<OaResponse> for CompletionResponse {
    fn from(r: OaResponse) -> Self {
        let choice = r.choices.into_iter().next();
        let (content, stop_reason) = match choice {
            Some(c) => {
                let mut blocks: Vec<MessageBlock> = Vec::new();
                if let Some(text) = c.message.content {
                    if !text.is_empty() {
                        blocks.push(MessageBlock::Text { text });
                    }
                }
                for tc in c.message.tool_calls {
                    let input: Value =
                        serde_json::from_str(&tc.function.arguments).unwrap_or(Value::Null);
                    blocks.push(MessageBlock::ToolCall {
                        id: tc.id,
                        name: tc.function.name,
                        input,
                    });
                }
                let stop = match c.finish_reason.as_deref() {
                    Some("tool_calls") => StopReason::ToolUse,
                    Some("stop") => StopReason::EndTurn,
                    Some("length") => StopReason::MaxTokens,
                    _ => StopReason::Other,
                };
                (blocks, stop)
            }
            None => (Vec::new(), StopReason::Other),
        };

        let usage = r
            .usage
            .map(|u| Usage {
                input_tokens: u.prompt_tokens,
                output_tokens: u.completion_tokens,
                cache_read: 0,
                cache_create: 0,
            })
            .unwrap_or_default();

        CompletionResponse {
            content,
            stop_reason,
            usage,
        }
    }
}
