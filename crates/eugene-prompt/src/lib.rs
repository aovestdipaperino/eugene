//! Four-layer prompt builder with a static/dynamic cache boundary.
//!
//! A system prompt that survives production has structure. This crate names
//! the four layers (identity, instructions, output constraints, dynamic
//! context) and renders them in a stable order with a boundary between the
//! static prefix and the per-turn dynamic suffix.
//!
//! The boundary matters because the Anthropic Messages API supports prompt
//! caching: a `system` field sent as an array of text blocks, with a
//! `cache_control: {"type": "ephemeral"}` marker on the last block of the
//! prefix, will reuse the cached prefix across requests within a 5-minute
//! TTL. Cache hits are billed at roughly 10% of input cost, which adds up
//! fast on agents that send the same instructions every turn.
//!
//! ```
//! use eugene_prompt::SystemPromptBuilder;
//!
//! let prompt = SystemPromptBuilder::new()
//!     .identity("You are Eugene, a careful research assistant.")
//!     .instruction("Use `read_file` when the user asks about file contents.")
//!     .output_constraints("Keep replies under 200 words.")
//!     .context(format!("Today is {}.", "2026-05-22"))
//!     .build();
//!
//! assert!(prompt.cacheable_prefix().contains("## Identity"));
//! assert!(prompt.dynamic_suffix().contains("Today is"));
//! assert_eq!(prompt.cache_blocks().len(), 2);
//! ```

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// A single few-shot example. Always rendered in the static prefix.
#[derive(Debug, Clone)]
pub struct Example {
    pub user: String,
    pub assistant: String,
}

impl Example {
    pub fn new(user: impl Into<String>, assistant: impl Into<String>) -> Self {
        Self {
            user: user.into(),
            assistant: assistant.into(),
        }
    }
}

/// A renderable block ready to be handed to a provider that supports
/// prompt caching. Anthropic's API maps this directly: each block becomes
/// `{type: "text", text: ..., cache_control: ...}` with `cache_control`
/// set only on `cache_breakpoint` entries.
#[derive(Debug, Clone)]
pub struct CacheBlock {
    pub text: String,
    /// If true, the provider should cut a cache boundary at the end of
    /// this block. With Anthropic, the API caches everything up to and
    /// including this block.
    pub cache_breakpoint: bool,
}

#[derive(Debug, Clone, Default)]
pub struct SystemPromptBuilder {
    identity: Option<String>,
    instructions: Vec<String>,
    output: Option<String>,
    examples: Vec<Example>,
    context: Vec<String>,
    extra_static: Vec<(String, String)>,
    extra_dynamic: Vec<(String, String)>,
}

impl SystemPromptBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Layer 1, static: who the agent is. One paragraph, present tense.
    pub fn identity(mut self, identity: impl Into<String>) -> Self {
        self.identity = Some(identity.into());
        self
    }

    /// Layer 2, static: one behavioural rule. Call repeatedly to grow the
    /// list. Each rule renders as a bullet item.
    pub fn instruction(mut self, instruction: impl Into<String>) -> Self {
        self.instructions.push(instruction.into());
        self
    }

    pub fn instructions<I, S>(mut self, instructions: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.instructions
            .extend(instructions.into_iter().map(Into::into));
        self
    }

    /// Layer 3, static: what the answer should look like.
    pub fn output_constraints(mut self, constraints: impl Into<String>) -> Self {
        self.output = Some(constraints.into());
        self
    }

    /// Layer 4, static: a few-shot example. Examples are part of the
    /// persona and live in the cacheable prefix.
    pub fn example(mut self, user: impl Into<String>, assistant: impl Into<String>) -> Self {
        self.examples.push(Example::new(user, assistant));
        self
    }

    /// Dynamic: facts that change per request (date, locale, project root,
    /// available skills). Lives after the cache boundary.
    pub fn context(mut self, context: impl Into<String>) -> Self {
        self.context.push(context.into());
        self
    }

    /// Escape hatch: add a named section to the static prefix.
    pub fn static_section(mut self, name: impl Into<String>, body: impl Into<String>) -> Self {
        self.extra_static.push((name.into(), body.into()));
        self
    }

    /// Escape hatch: add a named section to the dynamic suffix.
    pub fn dynamic_section(mut self, name: impl Into<String>, body: impl Into<String>) -> Self {
        self.extra_dynamic.push((name.into(), body.into()));
        self
    }

    pub fn build(self) -> SystemPrompt {
        let mut static_layers: Vec<(String, String)> = Vec::new();

        if let Some(identity) = self.identity {
            static_layers.push(("Identity".into(), identity));
        }
        if !self.instructions.is_empty() {
            let body = self
                .instructions
                .iter()
                .map(|i| format!("- {i}"))
                .collect::<Vec<_>>()
                .join("\n");
            static_layers.push(("Instructions".into(), body));
        }
        if let Some(output) = self.output {
            static_layers.push(("Output".into(), output));
        }
        if !self.examples.is_empty() {
            let body = self
                .examples
                .iter()
                .enumerate()
                .map(|(i, ex)| {
                    format!(
                        "Example {}:\nUser: {}\nAssistant: {}",
                        i + 1,
                        ex.user,
                        ex.assistant
                    )
                })
                .collect::<Vec<_>>()
                .join("\n\n");
            static_layers.push(("Examples".into(), body));
        }
        for (name, body) in self.extra_static {
            static_layers.push((name, body));
        }

        let mut dynamic_layers: Vec<(String, String)> = Vec::new();
        if !self.context.is_empty() {
            dynamic_layers.push(("Context".into(), self.context.join("\n\n")));
        }
        for (name, body) in self.extra_dynamic {
            dynamic_layers.push((name, body));
        }

        let cacheable_prefix = render_layers(&static_layers);
        let dynamic_suffix = render_layers(&dynamic_layers);

        SystemPrompt {
            cacheable_prefix,
            dynamic_suffix,
        }
    }
}

fn render_layers(layers: &[(String, String)]) -> String {
    layers
        .iter()
        .map(|(header, body)| format!("## {header}\n\n{body}"))
        .collect::<Vec<_>>()
        .join("\n\n")
}

#[derive(Debug, Clone)]
pub struct SystemPrompt {
    cacheable_prefix: String,
    dynamic_suffix: String,
}

impl SystemPrompt {
    /// The static layers, ready to send as the cacheable portion of an
    /// Anthropic `system` array.
    pub fn cacheable_prefix(&self) -> &str {
        &self.cacheable_prefix
    }

    /// The dynamic layers, ready to send as the per-turn portion.
    pub fn dynamic_suffix(&self) -> &str {
        &self.dynamic_suffix
    }

    /// The full prompt as one string. Convenient when the provider does
    /// not support per-block cache markers.
    pub fn as_string(&self) -> String {
        match (
            self.cacheable_prefix.is_empty(),
            self.dynamic_suffix.is_empty(),
        ) {
            (true, true) => String::new(),
            (false, true) => self.cacheable_prefix.clone(),
            (true, false) => self.dynamic_suffix.clone(),
            (false, false) => {
                format!("{}\n\n{}", self.cacheable_prefix, self.dynamic_suffix)
            }
        }
    }

    /// Rendered as one block per cache region. The first block carries the
    /// cache boundary (`cache_breakpoint = true`); the second does not.
    /// Pass these straight to a provider that supports per-block caching.
    pub fn cache_blocks(&self) -> Vec<CacheBlock> {
        let mut blocks = Vec::new();
        if !self.cacheable_prefix.is_empty() {
            blocks.push(CacheBlock {
                text: self.cacheable_prefix.clone(),
                cache_breakpoint: true,
            });
        }
        if !self.dynamic_suffix.is_empty() {
            blocks.push(CacheBlock {
                text: self.dynamic_suffix.clone(),
                cache_breakpoint: false,
            });
        }
        blocks
    }

    /// Stable hash of the rendered prompt. Use it as a fingerprint in
    /// regression tests: when this changes, the prompt changed.
    pub fn fingerprint(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.cacheable_prefix.hash(&mut hasher);
        self.dynamic_suffix.hash(&mut hasher);
        hasher.finish()
    }

    /// Stable hash of the cacheable prefix alone. This is the fingerprint
    /// the prompt cache keys on: when the prefix hash is unchanged, the
    /// next request hits the cache.
    pub fn prefix_fingerprint(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.cacheable_prefix.hash(&mut hasher);
        hasher.finish()
    }
}

impl std::fmt::Display for SystemPrompt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.as_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_builder_yields_empty_prompt() {
        let p = SystemPromptBuilder::new().build();
        assert!(p.cacheable_prefix().is_empty());
        assert!(p.dynamic_suffix().is_empty());
        assert_eq!(p.as_string(), "");
        assert!(p.cache_blocks().is_empty());
    }

    #[test]
    fn static_layers_render_in_canonical_order() {
        let p = SystemPromptBuilder::new()
            .example("u", "a")
            .output_constraints("plain prose")
            .instruction("be terse")
            .identity("you are Eugene")
            .build();

        let prefix = p.cacheable_prefix();
        let id = prefix.find("## Identity").unwrap();
        let instr = prefix.find("## Instructions").unwrap();
        let out = prefix.find("## Output").unwrap();
        let ex = prefix.find("## Examples").unwrap();

        assert!(id < instr);
        assert!(instr < out);
        assert!(out < ex);
    }

    #[test]
    fn context_lives_in_dynamic_suffix() {
        let p = SystemPromptBuilder::new()
            .identity("you are Eugene")
            .context("today is 2026-05-22")
            .build();
        assert!(p.cacheable_prefix().contains("Identity"));
        assert!(!p.cacheable_prefix().contains("Context"));
        assert!(p.dynamic_suffix().contains("Context"));
        assert!(p.dynamic_suffix().contains("today is 2026-05-22"));
    }

    #[test]
    fn cache_blocks_have_breakpoint_on_prefix_only() {
        let p = SystemPromptBuilder::new()
            .identity("X")
            .context("Y")
            .build();
        let blocks = p.cache_blocks();
        assert_eq!(blocks.len(), 2);
        assert!(blocks[0].cache_breakpoint);
        assert!(!blocks[1].cache_breakpoint);
    }

    #[test]
    fn cache_blocks_have_one_block_when_only_static() {
        let p = SystemPromptBuilder::new().identity("X").build();
        let blocks = p.cache_blocks();
        assert_eq!(blocks.len(), 1);
        assert!(blocks[0].cache_breakpoint);
    }

    #[test]
    fn instructions_render_as_bulleted_list() {
        let p = SystemPromptBuilder::new()
            .instruction("first")
            .instruction("second")
            .build();
        assert!(p.cacheable_prefix().contains("- first\n- second"));
    }

    #[test]
    fn prefix_fingerprint_is_unchanged_by_dynamic_drift() {
        let a = SystemPromptBuilder::new()
            .identity("X")
            .context("Today is 2026-05-22")
            .build();
        let b = SystemPromptBuilder::new()
            .identity("X")
            .context("Today is 2026-05-23")
            .build();
        assert_eq!(a.prefix_fingerprint(), b.prefix_fingerprint());
        assert_ne!(a.fingerprint(), b.fingerprint());
    }

    #[test]
    fn extra_dynamic_sections_join_the_suffix() {
        let p = SystemPromptBuilder::new()
            .identity("X")
            .dynamic_section("Env", "<env>cwd=/tmp</env>")
            .build();
        assert!(p.dynamic_suffix().contains("## Env"));
        assert!(p.dynamic_suffix().contains("<env>cwd=/tmp</env>"));
    }
}
