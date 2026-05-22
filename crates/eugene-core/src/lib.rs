//! Provider-agnostic types shared across the eugene agent crates.
//!
//! The shape of a conversation is the same regardless of which LLM provider
//! sits behind it. This crate names the parts so the rest of the workspace
//! can talk about messages, roles, and tool calls without dragging in a
//! specific SDK.

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    /// JSON-encoded arguments produced by the model. Validate before use.
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    /// Only set when `role == Role::Tool`; pairs the result with the call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: Some(content.into()),
            tool_calls: Vec::new(),
            tool_call_id: None,
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: Some(content.into()),
            tool_calls: Vec::new(),
            tool_call_id: None,
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: Some(content.into()),
            tool_calls: Vec::new(),
            tool_call_id: None,
        }
    }

    pub fn assistant_with_tool_calls(tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: Role::Assistant,
            content: None,
            tool_calls,
            tool_call_id: None,
        }
    }

    pub fn tool_result(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: Some(content.into()),
            tool_calls: Vec::new(),
            tool_call_id: Some(tool_call_id.into()),
        }
    }
}

/// Provider-neutral description of a callable tool. Conversion to the
/// provider-specific shape (Anthropic `Tool`, OpenAI `ChatCompletionTool`,
/// Gemini `FunctionDeclaration`) lives in the provider adapter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    /// JSON Schema describing the tool's input. Typically produced by
    /// `schemars::schema_for!` on the input struct.
    pub input_schema: serde_json::Value,
}

#[derive(Debug, Error)]
pub enum AgentError {
    #[error("agent did not converge within {0} turns")]
    NoConvergence(usize),

    #[error("unknown tool: {0}")]
    UnknownTool(String),

    #[error("tool argument decode failed: {0}")]
    BadArguments(#[from] serde_json::Error),

    #[error("provider error: {0}")]
    Provider(String),

    #[error("tool error: {0}")]
    Tool(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_constructors_round_trip() {
        let msg = Message::user("hello");
        let json = serde_json::to_string(&msg).unwrap();
        let back: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(back.role, Role::User);
        assert_eq!(back.content.as_deref(), Some("hello"));
    }

    #[test]
    fn tool_result_carries_id() {
        let msg = Message::tool_result("call_42", "ok");
        assert_eq!(msg.role, Role::Tool);
        assert_eq!(msg.tool_call_id.as_deref(), Some("call_42"));
    }
}
