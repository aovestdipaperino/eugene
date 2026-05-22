//! Four-layer prompt builder.
//!
//! A system prompt that survives production has structure. This crate names
//! the four layers (identity, instructions, output constraints, dynamic
//! context) and renders them in a stable order. The output is a plain
//! string that any LLM provider can consume as a system message.
//!
//! ```
//! use eugene_prompt::SystemPromptBuilder;
//!
//! let prompt = SystemPromptBuilder::new()
//!     .identity("You are Eugene, a careful research assistant.")
//!     .instruction("Use `read_file` when the user asks about file contents.")
//!     .instruction("Answer in plain prose. No markdown.")
//!     .output_constraints("Keep replies under 200 words.")
//!     .context(format!("Today is {}.", "2026-05-22"))
//!     .build();
//!
//! assert!(prompt.as_str().contains("## Identity"));
//! ```

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

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

#[derive(Debug, Clone, Default)]
pub struct SystemPromptBuilder {
    identity: Option<String>,
    instructions: Vec<String>,
    output: Option<String>,
    context: Vec<String>,
    examples: Vec<Example>,
}

impl SystemPromptBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Layer 1: who the agent is. One paragraph, present tense.
    pub fn identity(mut self, identity: impl Into<String>) -> Self {
        self.identity = Some(identity.into());
        self
    }

    /// Layer 2: a single behavioural rule. Call repeatedly to build the list.
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

    /// Layer 3: what the answer should look like.
    pub fn output_constraints(mut self, constraints: impl Into<String>) -> Self {
        self.output = Some(constraints.into());
        self
    }

    /// Layer 4: dynamic facts injected at construction time (date, user
    /// locale, available skills). Call repeatedly for multiple snippets.
    pub fn context(mut self, context: impl Into<String>) -> Self {
        self.context.push(context.into());
        self
    }

    pub fn example(mut self, user: impl Into<String>, assistant: impl Into<String>) -> Self {
        self.examples.push(Example::new(user, assistant));
        self
    }

    pub fn build(self) -> SystemPrompt {
        let mut layers: Vec<(&'static str, String)> = Vec::new();

        if let Some(identity) = self.identity {
            layers.push(("Identity", identity));
        }

        if !self.instructions.is_empty() {
            let body = self
                .instructions
                .iter()
                .map(|i| format!("- {i}"))
                .collect::<Vec<_>>()
                .join("\n");
            layers.push(("Instructions", body));
        }

        if let Some(output) = self.output {
            layers.push(("Output", output));
        }

        if !self.context.is_empty() {
            let body = self.context.join("\n\n");
            layers.push(("Context", body));
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
            layers.push(("Examples", body));
        }

        let text = layers
            .iter()
            .map(|(header, body)| format!("## {header}\n\n{body}"))
            .collect::<Vec<_>>()
            .join("\n\n");

        SystemPrompt { text, layers }
    }
}

#[derive(Debug, Clone)]
pub struct SystemPrompt {
    text: String,
    layers: Vec<(&'static str, String)>,
}

impl SystemPrompt {
    pub fn as_str(&self) -> &str {
        &self.text
    }

    pub fn into_string(self) -> String {
        self.text
    }

    pub fn layers(&self) -> &[(&'static str, String)] {
        &self.layers
    }

    /// Stable hash of the rendered prompt. Use it as a fingerprint in
    /// regression tests: when this changes, the prompt changed.
    pub fn fingerprint(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.text.hash(&mut hasher);
        hasher.finish()
    }
}

impl std::fmt::Display for SystemPrompt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_builder_yields_empty_prompt() {
        let p = SystemPromptBuilder::new().build();
        assert_eq!(p.as_str(), "");
        assert!(p.layers().is_empty());
    }

    #[test]
    fn layers_render_in_canonical_order() {
        let p = SystemPromptBuilder::new()
            .context("today is 2026-05-22")
            .output_constraints("plain prose")
            .instruction("be terse")
            .identity("you are Eugene")
            .build();

        let text = p.as_str();
        let id = text.find("## Identity").unwrap();
        let instr = text.find("## Instructions").unwrap();
        let out = text.find("## Output").unwrap();
        let ctx = text.find("## Context").unwrap();

        assert!(id < instr);
        assert!(instr < out);
        assert!(out < ctx);
    }

    #[test]
    fn instructions_render_as_bulleted_list() {
        let p = SystemPromptBuilder::new()
            .instruction("first")
            .instruction("second")
            .build();
        assert!(p.as_str().contains("- first\n- second"));
    }

    #[test]
    fn examples_carry_user_and_assistant_turns() {
        let p = SystemPromptBuilder::new()
            .example("hi", "hello")
            .example("bye", "goodbye")
            .build();
        let text = p.as_str();
        assert!(text.contains("Example 1:"));
        assert!(text.contains("User: hi"));
        assert!(text.contains("Assistant: goodbye"));
    }

    #[test]
    fn fingerprint_is_stable_across_builds() {
        let a = SystemPromptBuilder::new()
            .identity("X")
            .instruction("Y")
            .build();
        let b = SystemPromptBuilder::new()
            .identity("X")
            .instruction("Y")
            .build();
        assert_eq!(a.fingerprint(), b.fingerprint());
    }

    #[test]
    fn fingerprint_detects_drift() {
        let a = SystemPromptBuilder::new().identity("X").build();
        let b = SystemPromptBuilder::new().identity("Y").build();
        assert_ne!(a.fingerprint(), b.fingerprint());
    }

    #[test]
    fn instruction_order_is_preserved_in_fingerprint() {
        let a = SystemPromptBuilder::new()
            .instruction("a")
            .instruction("b")
            .build();
        let b = SystemPromptBuilder::new()
            .instruction("b")
            .instruction("a")
            .build();
        assert_ne!(a.fingerprint(), b.fingerprint());
    }
}
