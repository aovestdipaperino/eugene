//! MCP server and client adapters built on `rmcp`.
//!
//! This crate bridges Eugene's `eugene-skills::Registry` with the Model
//! Context Protocol in both directions. The server adapter exposes any
//! registry as an MCP server (stdio transport, suitable for being
//! spawned as a child process by an MCP host like Claude Code or the
//! MCP Inspector). The client adapter connects to an MCP server, lists
//! its tools, and surfaces each one as a `DynSkill` that the registry
//! can dispatch to as if it were local.
//!
//! Two functions cover the common case. [`serve_stdio`] runs a
//! registry as a stdio MCP server until the parent closes the pipe.
//! [`connect_child_process`] spawns a server as a subprocess and
//! returns a handle whose `into_skills()` lets you fold the remote
//! tools into a local registry.

use std::sync::Arc;

use async_trait::async_trait;
use eugene_core::ToolDefinition;
use eugene_skills::{DynSkill, HookOutcome, Registry, SkillError, SkillHook, SkillMetadata};
use rmcp::handler::server::ServerHandler;
use rmcp::model::{
    CallToolRequestParams, CallToolResult, Content, ErrorData, Implementation, JsonObject,
    ListToolsResult, PaginatedRequestParams, ServerCapabilities, ServerInfo, Tool, ToolsCapability,
};
use rmcp::service::{RequestContext, RoleServer};
use rmcp::transport::TokioChildProcess;
use rmcp::{ServiceExt, serve_server};
use serde_json::Value;
use thiserror::Error;
use tokio::process::Command;

#[derive(Debug, Error)]
pub enum McpError {
    #[error("rmcp service error: {0}")]
    Service(String),

    #[error("initialise error: {0}")]
    Init(String),

    #[error("transport error: {0}")]
    Transport(String),

    #[error("call error: {0}")]
    Call(String),
}

// ---------------------------------------------------------------------------
// Server side
// ---------------------------------------------------------------------------

/// Adapter that exposes a `eugene-skills::Registry` as an MCP server.
/// One instance per server. The registry is shared by `Arc` so multiple
/// servers can mount the same skill set.
pub struct RegistryServer {
    registry: Arc<Registry>,
    server_name: String,
    server_version: String,
}

impl RegistryServer {
    pub fn new(registry: Arc<Registry>) -> Self {
        Self {
            registry,
            server_name: "eugene".into(),
            server_version: env!("CARGO_PKG_VERSION").into(),
        }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.server_name = name.into();
        self
    }

    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.server_version = version.into();
        self
    }
}

impl ServerHandler for RegistryServer {
    fn get_info(&self) -> ServerInfo {
        let mut capabilities = ServerCapabilities::default();
        capabilities.tools = Some(ToolsCapability::default());
        let mut info = ServerInfo::default();
        info.capabilities = capabilities;
        info.server_info = Implementation::new(&self.server_name, &self.server_version);
        info.instructions = Some("Tools provided by an `eugene-skills::Registry`.".to_string());
        info
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _ctx: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, ErrorData> {
        let tools = self
            .registry
            .definitions()
            .into_iter()
            .map(tool_def_to_rmcp_tool)
            .collect();
        Ok(ListToolsResult::with_all_items(tools))
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _ctx: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, ErrorData> {
        let args = request
            .arguments
            .map(Value::Object)
            .unwrap_or(Value::Object(Default::default()));
        match self.registry.run(&request.name, &args).await {
            Ok(out) => Ok(CallToolResult::success(vec![Content::text(out)])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "ERROR: {e}"
            ))])),
        }
    }
}

/// Convert a provider-neutral `ToolDefinition` to the rmcp `Tool` shape.
fn tool_def_to_rmcp_tool(def: ToolDefinition) -> Tool {
    let schema: JsonObject = match def.input_schema {
        Value::Object(m) => m,
        _ => Default::default(),
    };
    Tool::new(def.name, def.description, Arc::new(schema))
}

/// Run the given registry as a stdio MCP server until the parent
/// closes the pipe. This is the canonical way to expose a Rust agent's
/// skills to a Node/Python MCP host.
pub async fn serve_stdio(registry: Arc<Registry>) -> Result<(), McpError> {
    let server = RegistryServer::new(registry);
    let transport = rmcp::transport::io::stdio();
    let running = serve_server(server, transport)
        .await
        .map_err(|e| McpError::Init(e.to_string()))?;
    running
        .waiting()
        .await
        .map_err(|e| McpError::Service(e.to_string()))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Client side
// ---------------------------------------------------------------------------

/// A handle to a running MCP client connection. Drop the handle to
/// terminate the connection (and, in the child-process case, the
/// subprocess too).
pub struct McpConnection {
    service: rmcp::service::RunningService<rmcp::service::RoleClient, ()>,
}

impl McpConnection {
    /// List the tools the server exposes.
    pub async fn list_tools(&self) -> Result<Vec<Tool>, McpError> {
        self.service
            .peer()
            .list_all_tools()
            .await
            .map_err(|e| McpError::Call(e.to_string()))
    }

    /// Call a remote tool by name.
    pub async fn call(&self, name: &str, args: Value) -> Result<String, McpError> {
        let arguments = match args {
            Value::Object(o) => Some(o),
            _ => None,
        };
        let result = self
            .service
            .peer()
            .call_tool({
                let mut p = CallToolRequestParams::new(name.to_string());
                if let Some(a) = arguments {
                    p = p.with_arguments(a);
                }
                p
            })
            .await
            .map_err(|e| McpError::Call(e.to_string()))?;
        let text = result
            .content
            .into_iter()
            .filter_map(|c| c.as_text().map(|t| t.text.clone()))
            .collect::<Vec<_>>()
            .join("\n");
        Ok(text)
    }

    /// Turn the remote tools into `DynSkill`s. Each remote tool becomes
    /// a `RemoteSkill` that the local `Registry` can dispatch to.
    pub async fn into_skills(self) -> Result<Vec<RemoteSkill>, McpError> {
        let tools = self.list_tools().await?;
        let service = Arc::new(self.service);
        Ok(tools
            .into_iter()
            .map(|t| RemoteSkill {
                service: service.clone(),
                tool: t,
            })
            .collect())
    }
}

/// Spawn an MCP server as a child process and connect to it over stdio.
pub async fn connect_child_process(
    command: impl AsRef<str>,
    args: &[&str],
) -> Result<McpConnection, McpError> {
    let mut cmd = Command::new(command.as_ref());
    for a in args {
        cmd.arg(a);
    }
    let transport = TokioChildProcess::new(cmd).map_err(|e| McpError::Transport(e.to_string()))?;
    let service = ().serve(transport).await.map_err(|e| McpError::Init(e.to_string()))?;
    Ok(McpConnection { service })
}

// ---------------------------------------------------------------------------
// Remote skill: a DynSkill backed by an MCP peer
// ---------------------------------------------------------------------------

pub struct RemoteSkill {
    service: Arc<rmcp::service::RunningService<rmcp::service::RoleClient, ()>>,
    tool: Tool,
}

#[async_trait]
impl DynSkill for RemoteSkill {
    fn definition(&self) -> ToolDefinition {
        let schema = Value::Object((*self.tool.input_schema).clone());
        ToolDefinition {
            name: self.tool.name.to_string(),
            description: self.tool.description.as_deref().unwrap_or("").to_string(),
            input_schema: schema,
        }
    }

    fn metadata(&self) -> SkillMetadata {
        // Remote tools have unknown destructiveness. Default to "not
        // safe to parallelise" to match the caution of Claude Code's
        // MCP integration, which gates MCP calls behind explicit
        // permission rules.
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

    async fn run_json(&self, args: &Value) -> Result<String, SkillError> {
        let arguments = match args {
            Value::Object(o) => Some(o.clone()),
            _ => None,
        };
        let mut params = CallToolRequestParams::new(self.tool.name.clone());
        if let Some(a) = arguments {
            params = params.with_arguments(a);
        }
        let result = self
            .service
            .peer()
            .call_tool(params)
            .await
            .map_err(|e| SkillError::Tool(format!("mcp: {e}")))?;
        let text = result
            .content
            .into_iter()
            .filter_map(|c| c.as_text().map(|t| t.text.clone()))
            .collect::<Vec<_>>()
            .join("\n");
        if result.is_error == Some(true) {
            Err(SkillError::Tool(format!(
                "remote tool returned error: {text}"
            )))
        } else {
            Ok(text)
        }
    }
}

// ---------------------------------------------------------------------------
// Optional hook: enforce a per-tool allowlist for remote (MCP) skills.
// Mirrors Claude Code's `channelAllowlist` / `channelPermissions`.
// ---------------------------------------------------------------------------

pub struct McpAllowlistHook {
    allowed: std::collections::HashSet<String>,
}

impl McpAllowlistHook {
    pub fn new<I, S>(allowed: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            allowed: allowed.into_iter().map(Into::into).collect(),
        }
    }
}

#[async_trait]
impl SkillHook for McpAllowlistHook {
    async fn before(
        &self,
        skill_name: &str,
        _metadata: SkillMetadata,
        _args: &Value,
    ) -> Result<HookOutcome, SkillError> {
        if self.allowed.contains(skill_name) {
            Ok(HookOutcome::Proceed)
        } else {
            Ok(HookOutcome::Deny(format!(
                "MCP tool `{skill_name}` is not on the allowlist",
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_def_translates_to_rmcp_tool() {
        let def = ToolDefinition {
            name: "read_file".into(),
            description: "Read a file.".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": { "path": { "type": "string" } },
                "required": ["path"]
            }),
        };
        let tool = tool_def_to_rmcp_tool(def);
        assert_eq!(tool.name, "read_file");
        assert_eq!(tool.description.as_deref(), Some("Read a file."));
        assert_eq!(tool.input_schema.get("type").unwrap(), "object");
    }

    #[test]
    fn allowlist_hook_denies_unlisted() {
        let hook = McpAllowlistHook::new(["read_file"]);
        let md = SkillMetadata {
            is_read_only: true,
            is_destructive: false,
            is_concurrency_safe: true,
        };
        let rt = tokio::runtime::Runtime::new().unwrap();
        let allowed =
            rt.block_on(async { hook.before("read_file", md, &Value::Null).await.unwrap() });
        let denied = rt.block_on(async {
            hook.before("delete_everything", md, &Value::Null)
                .await
                .unwrap()
        });
        assert!(matches!(allowed, HookOutcome::Proceed));
        assert!(matches!(denied, HookOutcome::Deny(_)));
    }
}
