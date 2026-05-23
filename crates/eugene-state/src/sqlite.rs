//! SQLite-backed checkpointer.
//!
//! Schema: a single `checkpoints` table keyed by `run_id`. The state is
//! stored as JSON in one column so the same store works for any `S` that
//! implements `Serialize` / `DeserializeOwned`. The implementation uses
//! `rusqlite` (sync) inside `tokio::task::spawn_blocking` to stay out of
//! the async runtime.

use std::path::Path;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use rusqlite::{Connection, params};
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::{Checkpoint, Checkpointer, GraphError};

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS checkpoints (
    run_id     TEXT PRIMARY KEY,
    next_node  TEXT NOT NULL,
    state_json TEXT NOT NULL,
    updated_at INTEGER NOT NULL
);
"#;

pub struct SqliteCheckpointer {
    conn: Arc<Mutex<Connection>>,
}

impl SqliteCheckpointer {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, GraphError> {
        let conn = Connection::open(path).map_err(map_err)?;
        conn.execute_batch(SCHEMA).map_err(map_err)?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    pub fn open_in_memory() -> Result<Self, GraphError> {
        let conn = Connection::open_in_memory().map_err(map_err)?;
        conn.execute_batch(SCHEMA).map_err(map_err)?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }
}

fn map_err<E: std::fmt::Display>(e: E) -> GraphError {
    GraphError::Checkpoint(e.to_string())
}

#[async_trait]
impl<S> Checkpointer<S> for SqliteCheckpointer
where
    S: Serialize + DeserializeOwned + Send + Sync + 'static,
{
    async fn save(&self, run_id: &str, next_node: &str, state: &S) -> Result<(), GraphError> {
        let json = serde_json::to_string(state).map_err(map_err)?;
        let now = epoch_seconds();
        let run_id = run_id.to_string();
        let next_node = next_node.to_string();
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let guard = conn.lock().map_err(|e| e.to_string())?;
            guard
                .execute(
                    "INSERT INTO checkpoints (run_id, next_node, state_json, updated_at) \
                     VALUES (?1, ?2, ?3, ?4) \
                     ON CONFLICT(run_id) DO UPDATE SET \
                       next_node = excluded.next_node, \
                       state_json = excluded.state_json, \
                       updated_at = excluded.updated_at",
                    params![run_id, next_node, json, now],
                )
                .map_err(|e| e.to_string())?;
            Ok::<_, String>(())
        })
        .await
        .map_err(map_err)?
        .map_err(GraphError::Checkpoint)
    }

    async fn load(&self, run_id: &str) -> Result<Option<Checkpoint<S>>, GraphError> {
        let run_id = run_id.to_string();
        let conn = self.conn.clone();
        let row = tokio::task::spawn_blocking(move || {
            let guard = conn.lock().map_err(|e| e.to_string())?;
            let mut stmt = guard
                .prepare("SELECT next_node, state_json FROM checkpoints WHERE run_id = ?1")
                .map_err(|e| e.to_string())?;
            let mut rows = stmt.query(params![run_id]).map_err(|e| e.to_string())?;
            if let Some(r) = rows.next().map_err(|e| e.to_string())? {
                let next: String = r.get(0).map_err(|e| e.to_string())?;
                let json: String = r.get(1).map_err(|e| e.to_string())?;
                Ok::<_, String>(Some((next, json)))
            } else {
                Ok(None)
            }
        })
        .await
        .map_err(map_err)?
        .map_err(GraphError::Checkpoint)?;

        match row {
            Some((next_node, json)) => {
                let state: S = serde_json::from_str(&json).map_err(map_err)?;
                Ok(Some(Checkpoint { next_node, state }))
            }
            None => Ok(None),
        }
    }

    async fn delete(&self, run_id: &str) -> Result<(), GraphError> {
        let run_id = run_id.to_string();
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let guard = conn.lock().map_err(|e| e.to_string())?;
            guard
                .execute("DELETE FROM checkpoints WHERE run_id = ?1", params![run_id])
                .map_err(|e| e.to_string())?;
            Ok::<_, String>(())
        })
        .await
        .map_err(map_err)?
        .map_err(GraphError::Checkpoint)
    }
}

fn epoch_seconds() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
    struct S {
        n: u32,
        msg: String,
    }

    #[tokio::test]
    async fn round_trip_in_memory() {
        let ckpt = SqliteCheckpointer::open_in_memory().unwrap();
        let s = S {
            n: 42,
            msg: "hello".into(),
        };

        <SqliteCheckpointer as Checkpointer<S>>::save(&ckpt, "r1", "next", &s)
            .await
            .unwrap();
        let loaded = <SqliteCheckpointer as Checkpointer<S>>::load(&ckpt, "r1")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(loaded.next_node, "next");
        assert_eq!(loaded.state, s);

        <SqliteCheckpointer as Checkpointer<S>>::delete(&ckpt, "r1")
            .await
            .unwrap();
        let after = <SqliteCheckpointer as Checkpointer<S>>::load(&ckpt, "r1")
            .await
            .unwrap();
        assert!(after.is_none());
    }

    #[tokio::test]
    async fn survives_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ckpt.db");

        {
            let ckpt = SqliteCheckpointer::open(&path).unwrap();
            let s = S {
                n: 7,
                msg: "saved".into(),
            };
            <SqliteCheckpointer as Checkpointer<S>>::save(&ckpt, "r1", "next", &s)
                .await
                .unwrap();
        }

        let ckpt = SqliteCheckpointer::open(&path).unwrap();
        let loaded = <SqliteCheckpointer as Checkpointer<S>>::load(&ckpt, "r1")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(loaded.state.msg, "saved");
    }
}
