//! Layered memory for eugene agents.
//!
//! Two stores ship in this crate, one trait covers both, and a small
//! amount of glue lets the agent loop pick what to remember and what
//! to recall.
//!
//! [`FileMemory`] is the file-based pattern Claude Code uses: each
//! memory is a markdown file with YAML-ish frontmatter (`name`,
//! `description`, `type`) and a free-form body. A `MEMORY.md` index at
//! the root of the memory directory is loaded into the system prompt
//! verbatim; individual memory files are surfaced on demand via the
//! [`select_relevant_memories`] helper, which asks a cheap LLM to pick
//! files based on their description alone.
//!
//! [`VectorStore`] is the trait every embedding-based memory satisfies.
//! The crate ships [`InMemoryVectorStore`] as a tiny cosine-similarity
//! implementation; production deployments typically swap in a backend
//! like Qdrant or pgvector behind the same trait.
//!
//! ```
//! use eugene_memory::{FileMemory, MemoryEntry, MemoryType};
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let dir = tempfile::tempdir()?;
//! let mem = FileMemory::open(dir.path())?;
//! mem.add(MemoryEntry::new(
//!     "rust-typing",
//!     "User prefers terse, no-hedge explanations.",
//!     MemoryType::User,
//!     "The user has been writing Rust for years and finds verbose explanations annoying.",
//! )).await?;
//! let headers = mem.list().await?;
//! assert_eq!(headers.len(), 1);
//! # Ok(()) }
//! ```

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::SystemTime;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;
use tokio::fs;
use tokio::io::AsyncWriteExt;

#[derive(Debug, Error)]
pub enum MemoryError {
    #[error("io: {0}")]
    Io(String),

    #[error("parse: {0}")]
    Parse(String),

    #[error("not found: {0}")]
    NotFound(String),

    #[error("backend: {0}")]
    Backend(String),
}

impl From<std::io::Error> for MemoryError {
    fn from(e: std::io::Error) -> Self {
        MemoryError::Io(e.to_string())
    }
}

// ---------------------------------------------------------------------------
// File-based memory (Claude Code pattern)
// ---------------------------------------------------------------------------

/// The four memory types Claude Code names. Other taxonomies fit by
/// extending this enum or by using [`MemoryType::Other`] for free-form
/// labels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MemoryType {
    /// About the user themselves.
    User,
    /// Corrections and confirmations the user has given over time.
    Feedback,
    /// Ongoing initiatives, decisions, and incidents within a project.
    Project,
    /// Pointers to external systems (Jira project, Slack channel, ...).
    Reference,
    /// Anything else.
    Other,
}

impl MemoryType {
    pub fn as_str(self) -> &'static str {
        match self {
            MemoryType::User => "user",
            MemoryType::Feedback => "feedback",
            MemoryType::Project => "project",
            MemoryType::Reference => "reference",
            MemoryType::Other => "other",
        }
    }

    pub fn parse(s: &str) -> MemoryType {
        match s.trim().to_ascii_lowercase().as_str() {
            "user" => MemoryType::User,
            "feedback" => MemoryType::Feedback,
            "project" => MemoryType::Project,
            "reference" => MemoryType::Reference,
            _ => MemoryType::Other,
        }
    }
}

/// A full memory entry, body included.
#[derive(Debug, Clone)]
pub struct MemoryEntry {
    pub name: String,
    pub description: String,
    pub memory_type: MemoryType,
    pub body: String,
}

impl MemoryEntry {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        memory_type: MemoryType,
        body: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            memory_type,
            body: body.into(),
        }
    }
}

/// The selector view: just the metadata, no body. This is what the LLM
/// router sees when picking which memories to load in full.
#[derive(Debug, Clone)]
pub struct MemoryHeader {
    pub name: String,
    pub description: String,
    pub memory_type: MemoryType,
    pub path: PathBuf,
    pub mtime: SystemTime,
}

/// File-backed memory store. One markdown file per memory.
/// File layout:
///
/// ```text
/// ---
/// name: rust-typing
/// description: User prefers terse explanations.
/// type: user
/// ---
///
/// The user has been writing Rust for years ...
/// ```
pub struct FileMemory {
    dir: PathBuf,
}

impl FileMemory {
    pub fn open(dir: impl AsRef<Path>) -> Result<Self, MemoryError> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir)?;
        Ok(Self { dir })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    fn path_for(&self, name: &str) -> PathBuf {
        let safe = name
            .chars()
            .map(|c| {
                if c.is_alphanumeric() || c == '-' || c == '_' {
                    c
                } else {
                    '_'
                }
            })
            .collect::<String>();
        self.dir.join(format!("{safe}.md"))
    }

    pub async fn add(&self, entry: MemoryEntry) -> Result<(), MemoryError> {
        let path = self.path_for(&entry.name);
        let content = render(&entry);
        let mut f = fs::File::create(&path).await?;
        f.write_all(content.as_bytes()).await?;
        f.flush().await?;
        Ok(())
    }

    pub async fn load(&self, name: &str) -> Result<MemoryEntry, MemoryError> {
        let path = self.path_for(name);
        let raw = fs::read_to_string(&path)
            .await
            .map_err(|_| MemoryError::NotFound(name.to_string()))?;
        parse(&raw).ok_or_else(|| MemoryError::Parse(name.to_string()))
    }

    pub async fn delete(&self, name: &str) -> Result<(), MemoryError> {
        let path = self.path_for(name);
        match fs::remove_file(&path).await {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                Err(MemoryError::NotFound(name.to_string()))
            }
            Err(e) => Err(MemoryError::Io(e.to_string())),
        }
    }

    /// Scan the memory directory and return the lightweight header for
    /// every memory file. Skips MEMORY.md (the entrypoint loaded into
    /// the system prompt separately).
    pub async fn list(&self) -> Result<Vec<MemoryHeader>, MemoryError> {
        let mut headers = Vec::new();
        let mut read = fs::read_dir(&self.dir).await?;
        while let Some(entry) = read.next_entry().await? {
            let path = entry.path();
            if !path.extension().map(|e| e == "md").unwrap_or(false) {
                continue;
            }
            if path.file_name().map(|n| n == "MEMORY.md").unwrap_or(false) {
                continue;
            }
            let raw = match fs::read_to_string(&path).await {
                Ok(s) => s,
                Err(_) => continue,
            };
            let Some(parsed) = parse(&raw) else { continue };
            let mtime = entry
                .metadata()
                .await
                .ok()
                .and_then(|m| m.modified().ok())
                .unwrap_or(SystemTime::UNIX_EPOCH);
            headers.push(MemoryHeader {
                name: parsed.name,
                description: parsed.description,
                memory_type: parsed.memory_type,
                path,
                mtime,
            });
        }
        headers.sort_by(|a, b| b.mtime.cmp(&a.mtime));
        Ok(headers)
    }

    /// Read the MEMORY.md entrypoint, truncated to `max_bytes` if
    /// necessary. Mirrors Claude Code's `truncateEntrypointContent`:
    /// the truncation appends a warning so the model knows it was cut.
    pub async fn read_entrypoint(&self, max_bytes: usize) -> Result<Option<String>, MemoryError> {
        let path = self.dir.join("MEMORY.md");
        match fs::read_to_string(&path).await {
            Ok(text) => {
                let trimmed = text.trim();
                if trimmed.len() <= max_bytes {
                    Ok(Some(trimmed.to_string()))
                } else {
                    let cut = trimmed[..max_bytes]
                        .rsplit_once('\n')
                        .map(|(a, _)| a)
                        .unwrap_or(&trimmed[..max_bytes]);
                    Ok(Some(format!(
                        "{cut}\n\n[memory entrypoint truncated: file exceeds {max_bytes} bytes]"
                    )))
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(MemoryError::Io(e.to_string())),
        }
    }
}

fn render(entry: &MemoryEntry) -> String {
    format!(
        "---\nname: {}\ndescription: {}\ntype: {}\n---\n\n{}\n",
        entry.name,
        // Single-line in frontmatter; collapse newlines defensively.
        entry.description.replace('\n', " "),
        entry.memory_type.as_str(),
        entry.body.trim_end(),
    )
}

/// Parse a markdown file with YAML-ish frontmatter. The parser is
/// deliberately small: the front delimiter is `---` on its own line, the
/// trailing delimiter likewise, and inside the frontmatter we accept
/// `key: value` lines. Anything else gets ignored, so legacy files do
/// not crash the loader.
pub fn parse(raw: &str) -> Option<MemoryEntry> {
    let bytes = raw.trim_start();
    let after_open = bytes.strip_prefix("---\n")?;
    let (front, body) = after_open.split_once("\n---")?;
    let body = body.trim_start_matches('\n').to_string();

    let mut name = None;
    let mut description = None;
    let mut memory_type = MemoryType::Other;
    for line in front.lines() {
        if let Some((k, v)) = line.split_once(':') {
            let v = v.trim();
            match k.trim() {
                "name" => name = Some(v.to_string()),
                "description" => description = Some(v.to_string()),
                "type" => memory_type = MemoryType::parse(v),
                _ => {}
            }
        }
    }
    Some(MemoryEntry {
        name: name?,
        description: description.unwrap_or_default(),
        memory_type,
        body,
    })
}

// ---------------------------------------------------------------------------
// LLM-routed selection (Claude Code pattern)
// ---------------------------------------------------------------------------

/// Format a memory manifest for the selector LLM. Each line is one
/// memory: its name, type, and one-line description. The selector
/// returns names; the caller loads those memories in full.
pub fn format_manifest(headers: &[MemoryHeader]) -> String {
    headers
        .iter()
        .map(|h| {
            format!(
                "- {} [{}]: {}",
                h.name,
                h.memory_type.as_str(),
                h.description
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Apply a selector function (typically an LLM call) to pick the most
/// relevant memories for `query`. Returns the loaded entries for the
/// names the selector returned, up to `limit`. Order follows the
/// selector's order.
pub async fn select_relevant_memories<F, Fut>(
    query: &str,
    mem: &FileMemory,
    limit: usize,
    selector: F,
) -> Result<Vec<MemoryEntry>, MemoryError>
where
    F: FnOnce(&str, String) -> Fut,
    Fut: std::future::Future<Output = Result<Vec<String>, MemoryError>>,
{
    let headers = mem.list().await?;
    if headers.is_empty() {
        return Ok(Vec::new());
    }
    let manifest = format_manifest(&headers);
    let picked = selector(query, manifest).await?;
    let mut entries = Vec::with_capacity(limit.min(picked.len()));
    for name in picked.into_iter().take(limit) {
        if let Ok(entry) = mem.load(&name).await {
            entries.push(entry);
        }
    }
    Ok(entries)
}

// ---------------------------------------------------------------------------
// Vector memory
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct VectorHit {
    pub id: String,
    pub score: f32,
    pub payload: Value,
}

/// Anything that can store and search dense vectors satisfies this
/// trait. The crate ships [`InMemoryVectorStore`]; production setups
/// wrap a backend like Qdrant or pgvector here.
#[async_trait]
pub trait VectorStore: Send + Sync + 'static {
    async fn upsert(
        &self,
        id: &str,
        embedding: Vec<f32>,
        payload: Value,
    ) -> Result<(), MemoryError>;

    async fn search(&self, embedding: &[f32], limit: usize) -> Result<Vec<VectorHit>, MemoryError>;

    async fn delete(&self, id: &str) -> Result<(), MemoryError>;
}

/// Trait for anything that turns text into a vector. Implementations
/// call the model of your choice: Anthropic does not expose embeddings
/// directly, so most real deployments use OpenAI, Voyage AI, or a
/// local model via Ollama.
#[async_trait]
pub trait Embedder: Send + Sync + 'static {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, MemoryError>;
}

#[derive(Default)]
struct VectorRow {
    embedding: Vec<f32>,
    payload: Value,
}

/// Cosine-similarity in-memory vector store. Use for tests, prototypes,
/// and small-scale deployments (a few thousand entries). Beyond that,
/// reach for Qdrant.
pub struct InMemoryVectorStore {
    rows: Mutex<std::collections::HashMap<String, VectorRow>>,
}

impl Default for InMemoryVectorStore {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemoryVectorStore {
    pub fn new() -> Self {
        Self {
            rows: Mutex::new(std::collections::HashMap::new()),
        }
    }

    pub fn len(&self) -> usize {
        self.rows.lock().map(|g| g.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = (na.sqrt() * nb.sqrt()).max(f32::EPSILON);
    dot / denom
}

#[async_trait]
impl VectorStore for InMemoryVectorStore {
    async fn upsert(
        &self,
        id: &str,
        embedding: Vec<f32>,
        payload: Value,
    ) -> Result<(), MemoryError> {
        let mut g = self
            .rows
            .lock()
            .map_err(|e| MemoryError::Backend(e.to_string()))?;
        g.insert(id.to_string(), VectorRow { embedding, payload });
        Ok(())
    }

    async fn search(&self, embedding: &[f32], limit: usize) -> Result<Vec<VectorHit>, MemoryError> {
        let g = self
            .rows
            .lock()
            .map_err(|e| MemoryError::Backend(e.to_string()))?;
        let mut scored: Vec<(String, f32, Value)> = g
            .iter()
            .map(|(id, row)| {
                let score = cosine_similarity(embedding, &row.embedding);
                (id.clone(), score, row.payload.clone())
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);
        Ok(scored
            .into_iter()
            .map(|(id, score, payload)| VectorHit { id, score, payload })
            .collect())
    }

    async fn delete(&self, id: &str) -> Result<(), MemoryError> {
        let mut g = self
            .rows
            .lock()
            .map_err(|e| MemoryError::Backend(e.to_string()))?;
        g.remove(id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn file_memory_round_trip() {
        let tmp = tempfile::tempdir().unwrap();
        let mem = FileMemory::open(tmp.path()).unwrap();
        let entry = MemoryEntry::new(
            "rust-typing",
            "User prefers terse explanations.",
            MemoryType::User,
            "The user has been writing Rust for years.",
        );
        mem.add(entry.clone()).await.unwrap();

        let loaded = mem.load("rust-typing").await.unwrap();
        assert_eq!(loaded.name, "rust-typing");
        assert_eq!(loaded.memory_type, MemoryType::User);
        assert!(loaded.body.contains("writing Rust"));
    }

    #[tokio::test]
    async fn list_returns_headers_sorted_by_recency() {
        let tmp = tempfile::tempdir().unwrap();
        let mem = FileMemory::open(tmp.path()).unwrap();
        for (i, name) in ["older", "middle", "newest"].iter().enumerate() {
            mem.add(MemoryEntry::new(
                *name,
                format!("memory {i}"),
                MemoryType::Project,
                format!("body {i}"),
            ))
            .await
            .unwrap();
            // Force distinct mtimes.
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        let headers = mem.list().await.unwrap();
        assert_eq!(headers.len(), 3);
        assert_eq!(headers[0].name, "newest");
        assert_eq!(headers[2].name, "older");
    }

    #[tokio::test]
    async fn entrypoint_returns_none_when_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let mem = FileMemory::open(tmp.path()).unwrap();
        let entry = mem.read_entrypoint(25_000).await.unwrap();
        assert!(entry.is_none());
    }

    #[tokio::test]
    async fn entrypoint_truncates_with_warning() {
        let tmp = tempfile::tempdir().unwrap();
        let mem = FileMemory::open(tmp.path()).unwrap();
        let mut huge = String::new();
        for i in 0..2_000 {
            huge.push_str(&format!("line {i}\n"));
        }
        fs::write(tmp.path().join("MEMORY.md"), &huge)
            .await
            .unwrap();
        let read = mem.read_entrypoint(2_000).await.unwrap().unwrap();
        assert!(read.len() < huge.len());
        assert!(read.contains("truncated"));
    }

    #[tokio::test]
    async fn list_skips_memory_dot_md() {
        let tmp = tempfile::tempdir().unwrap();
        let mem = FileMemory::open(tmp.path()).unwrap();
        fs::write(tmp.path().join("MEMORY.md"), "# top-level index")
            .await
            .unwrap();
        mem.add(MemoryEntry::new(
            "alpha",
            "an entry",
            MemoryType::Project,
            "body",
        ))
        .await
        .unwrap();
        let headers = mem.list().await.unwrap();
        assert_eq!(headers.len(), 1);
        assert_eq!(headers[0].name, "alpha");
    }

    #[tokio::test]
    async fn manifest_includes_type_and_description() {
        let h = vec![
            MemoryHeader {
                name: "rust-typing".into(),
                description: "User prefers terse.".into(),
                memory_type: MemoryType::User,
                path: PathBuf::new(),
                mtime: SystemTime::UNIX_EPOCH,
            },
            MemoryHeader {
                name: "auth-rewrite".into(),
                description: "Driven by legal.".into(),
                memory_type: MemoryType::Project,
                path: PathBuf::new(),
                mtime: SystemTime::UNIX_EPOCH,
            },
        ];
        let m = format_manifest(&h);
        assert!(m.contains("[user]"));
        assert!(m.contains("[project]"));
        assert!(m.contains("terse"));
        assert!(m.contains("legal"));
    }

    #[tokio::test]
    async fn select_loads_picked_entries() {
        let tmp = tempfile::tempdir().unwrap();
        let mem = FileMemory::open(tmp.path()).unwrap();
        mem.add(MemoryEntry::new(
            "alpha",
            "first",
            MemoryType::Project,
            "alpha body",
        ))
        .await
        .unwrap();
        mem.add(MemoryEntry::new(
            "beta",
            "second",
            MemoryType::Project,
            "beta body",
        ))
        .await
        .unwrap();
        let picked =
            select_relevant_memories("anything", &mem, 5, |_query, _manifest| async move {
                Ok(vec!["beta".to_string()])
            })
            .await
            .unwrap();
        assert_eq!(picked.len(), 1);
        assert_eq!(picked[0].name, "beta");
        assert_eq!(picked[0].body.trim(), "beta body");
    }

    #[tokio::test]
    async fn cosine_search_ranks_closest_first() {
        let store = InMemoryVectorStore::new();
        store
            .upsert("a", vec![1.0, 0.0, 0.0], serde_json::json!({"name": "a"}))
            .await
            .unwrap();
        store
            .upsert("b", vec![0.0, 1.0, 0.0], serde_json::json!({"name": "b"}))
            .await
            .unwrap();
        store
            .upsert("c", vec![0.9, 0.1, 0.0], serde_json::json!({"name": "c"}))
            .await
            .unwrap();
        let hits = store.search(&[1.0, 0.0, 0.0], 3).await.unwrap();
        assert_eq!(hits[0].id, "a");
        assert_eq!(hits[1].id, "c");
        assert_eq!(hits[2].id, "b");
        assert!(hits[0].score > hits[1].score);
    }

    #[tokio::test]
    async fn vector_store_round_trip_with_delete() {
        let store = InMemoryVectorStore::new();
        store
            .upsert("a", vec![1.0, 0.0], serde_json::json!({}))
            .await
            .unwrap();
        assert_eq!(store.len(), 1);
        store.delete("a").await.unwrap();
        assert_eq!(store.len(), 0);
    }
}
