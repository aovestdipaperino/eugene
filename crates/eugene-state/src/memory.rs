use std::collections::HashMap;
use std::sync::Mutex;

use async_trait::async_trait;

use crate::{Checkpoint, Checkpointer, GraphError};

/// In-memory checkpoint store. Useful for tests and for short-lived
/// graphs whose state does not need to survive process restarts.
pub struct InMemoryCheckpointer<S>
where
    S: Send + Sync + 'static,
{
    inner: Mutex<HashMap<String, Checkpoint<S>>>,
}

impl<S> Default for InMemoryCheckpointer<S>
where
    S: Send + Sync + 'static,
{
    fn default() -> Self {
        Self {
            inner: Mutex::new(HashMap::new()),
        }
    }
}

impl<S> InMemoryCheckpointer<S>
where
    S: Send + Sync + 'static,
{
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl<S> Checkpointer<S> for InMemoryCheckpointer<S>
where
    S: Clone + Send + Sync + 'static,
{
    async fn save(&self, run_id: &str, next_node: &str, state: &S) -> Result<(), GraphError> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|e| GraphError::Checkpoint(e.to_string()))?;
        guard.insert(
            run_id.to_string(),
            Checkpoint {
                next_node: next_node.to_string(),
                state: state.clone(),
            },
        );
        Ok(())
    }

    async fn load(&self, run_id: &str) -> Result<Option<Checkpoint<S>>, GraphError> {
        let guard = self
            .inner
            .lock()
            .map_err(|e| GraphError::Checkpoint(e.to_string()))?;
        Ok(guard.get(run_id).map(|cp| Checkpoint {
            next_node: cp.next_node.clone(),
            state: cp.state.clone(),
        }))
    }

    async fn delete(&self, run_id: &str) -> Result<(), GraphError> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|e| GraphError::Checkpoint(e.to_string()))?;
        guard.remove(run_id);
        Ok(())
    }
}
