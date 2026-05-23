//! E2B-backed sandbox.
//!
//! Calls out to the E2B HTTP API, which runs each execution in a
//! Firecracker microVM with its own kernel and filesystem. The right
//! choice when `LocalSandbox` is not enough isolation: untrusted code,
//! multi-tenant workloads, anything that needs network or filesystem
//! that the host should not see.
//!
//! This module is feature-gated behind `e2b` because it pulls in
//! `reqwest` and is only useful when the caller has an E2B API key.
//! The implementation here is intentionally a thin shape: a real
//! production adapter would handle session reuse, file uploads,
//! streaming, and retries. Treat this file as the seam where the
//! `Sandbox` trait meets a remote runtime.

use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::{ExecResult, Limits, Sandbox, SandboxError};

const E2B_API_BASE: &str = "https://api.e2b.dev";

/// Configuration for an E2B sandbox session.
pub struct E2bSandbox {
    http: reqwest::Client,
    api_key: String,
    template: String,
    base_url: String,
}

impl E2bSandbox {
    pub fn new(api_key: impl Into<String>, template: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            api_key: api_key.into(),
            template: template.into(),
            base_url: E2B_API_BASE.to_string(),
        }
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }
}

#[derive(Serialize)]
struct ExecRequest<'a> {
    command: &'a str,
    timeout_ms: u64,
}

#[derive(Deserialize)]
struct ExecResponse {
    #[serde(default)]
    stdout: String,
    #[serde(default)]
    stderr: String,
    #[serde(default)]
    exit_code: Option<i32>,
    #[serde(default)]
    timed_out: bool,
}

#[async_trait]
impl Sandbox for E2bSandbox {
    async fn execute(&self, command: &str, limits: &Limits) -> Result<ExecResult, SandboxError> {
        let url = format!("{}/sandbox/{}/exec", self.base_url, self.template);
        let body = ExecRequest {
            command,
            timeout_ms: limits.timeout.as_millis() as u64,
        };
        let resp = self
            .http
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .timeout(limits.timeout + Duration::from_secs(5))
            .send()
            .await
            .map_err(|e| SandboxError::Backend(e.to_string()))?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(SandboxError::Backend(format!(
                "E2B {}: {text}",
                status.as_u16()
            )));
        }
        let parsed: ExecResponse = resp
            .json()
            .await
            .map_err(|e| SandboxError::Backend(e.to_string()))?;
        let cap = limits.max_output_bytes;
        let output_truncated = parsed.stdout.len() > cap || parsed.stderr.len() > cap;
        let stdout = if parsed.stdout.len() > cap {
            parsed.stdout[..cap].to_string()
        } else {
            parsed.stdout
        };
        let stderr = if parsed.stderr.len() > cap {
            parsed.stderr[..cap].to_string()
        } else {
            parsed.stderr
        };
        Ok(ExecResult {
            stdout,
            stderr,
            exit_code: parsed.exit_code,
            timed_out: parsed.timed_out,
            output_truncated,
        })
    }
}
