//! In-memory message store with TTL expiry + disk persistence.
//!
//! Each entry is persisted as a JSONL file under `messages/{id}.jsonl`.

use crate::types::chat::MessageRequest;
use rustc_hash::FxHashMap as HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::RwLock;

const DEFAULT_TTL: Duration = Duration::from_secs(30 * 60);

// ── StoredMessages ───────────────────────────────────────────────────────────

#[derive(Clone)]
struct StoredMessages {
    messages: Vec<MessageRequest>,
    created_at: Instant,
}

// ── Store ────────────────────────────────────────────────────────────────────

/// Thread-safe, in-memory message store with TTL cleanup and disk persistence.
#[derive(Clone)]
pub struct Store {
    inner: Arc<RwLock<HashMap<String, StoredMessages>>>,
    ttl: Duration,
    dir: Option<PathBuf>,
    cancel_tokens: Arc<RwLock<HashMap<String, tokio::sync::watch::Sender<bool>>>>,
}

impl Store {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(HashMap::default())),
            ttl: DEFAULT_TTL,
            dir: None,
            cancel_tokens: Arc::new(RwLock::new(HashMap::default())),
        }
    }

    pub fn with_dir(dir: PathBuf) -> Self {
        std::fs::create_dir_all(dir.join("messages")).ok();
        Self {
            inner: Arc::new(RwLock::new(HashMap::default())),
            ttl: DEFAULT_TTL,
            dir: Some(dir),
            cancel_tokens: Arc::new(RwLock::new(HashMap::default())),
        }
    }

    // ── CRUD ──────────────────────────────────────────────────────────────

    /// Store messages for a response ID.
    pub async fn put(&self, id: String, messages: Vec<MessageRequest>) {
        self.inner.write().await.insert(
            id.clone(),
            StoredMessages {
                messages,
                created_at: Instant::now(),
            },
        );

        if let Some(ref dir) = self.dir {
            let dir = dir.clone();
            let id_c = id.clone();
            let msgs = self
                .inner
                .read()
                .await
                .get(&id_c)
                .map(|e| e.messages.clone());
            tokio::spawn(async move {
                if let Some(msgs) = msgs {
                    let path = messages_path(&dir, &id_c);
                    if let Err(e) = write_messages(&path, &msgs).await {
                        tracing::error!(id = %id_c, error = %e, "Failed to persist messages");
                    }
                }
            });
        }
    }

    /// Retrieve stored messages by ID. Returns None if not found or expired.
    pub async fn get(&self, id: &str) -> Option<Vec<MessageRequest>> {
        {
            let g = self.inner.read().await;
            if let Some(entry) = g.get(id) {
                if entry.created_at.elapsed() <= self.ttl {
                    return Some(entry.messages.clone());
                }
                drop(g);
                self.inner.write().await.remove(id);
                self.delete_disk_files(id);
                return None;
            }
        }
        self.load_messages_from_disk(id).await
    }

    /// Delete a single entry.
    pub async fn delete(&self, id: &str) -> bool {
        let existed = self.inner.write().await.remove(id).is_some();
        self.delete_disk_files(id);
        existed
    }

    /// Remove all expired entries.
    pub async fn sweep_expired(&self) {
        let expired: Vec<String> = {
            let g = self.inner.read().await;
            g.iter()
                .filter(|(_, v)| v.created_at.elapsed() > self.ttl)
                .map(|(k, _)| k.clone())
                .collect()
        };
        if expired.is_empty() {
            return;
        }
        for k in &expired {
            self.inner.write().await.remove(k);
            self.delete_disk_files(k);
        }
        tracing::info!(count = expired.len(), "Swept expired entries from store");
    }

    pub fn start_sweep_task(&self) {
        let store = self.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(300));
            loop {
                interval.tick().await;
                store.sweep_expired().await;
            }
        });
    }

    // ── Cancellation ──────────────────────────────────────────────────────

    pub async fn register_cancel_token(&self, id: &str) -> tokio::sync::watch::Receiver<bool> {
        let (tx, rx) = tokio::sync::watch::channel(false);
        self.cancel_tokens.write().await.insert(id.to_string(), tx);
        rx
    }

    pub async fn cancel_in_flight(&self, id: &str) -> bool {
        if let Some(tx) = self.cancel_tokens.write().await.remove(id) {
            let _ = tx.send(true);
            true
        } else {
            false
        }
    }

    pub async fn unregister_cancel_token(&self, id: &str) {
        self.cancel_tokens.write().await.remove(id);
    }

    // ── Disk persistence ──────────────────────────────────────────────────

    fn delete_disk_files(&self, id: &str) {
        if let Some(ref dir) = self.dir {
            delete_messages_file(dir, id);
        }
    }

    async fn load_messages_from_disk(&self, id: &str) -> Option<Vec<MessageRequest>> {
        let dir = self.dir.as_ref()?;
        let path = messages_path(dir, id);
        let msgs = read_messages(&path).await?;
        self.inner.write().await.insert(
            id.to_string(),
            StoredMessages {
                messages: msgs.clone(),
                created_at: Instant::now(),
            },
        );
        Some(msgs)
    }
}

impl Default for Store {
    fn default() -> Self {
        Self::new()
    }
}

// ── Persistence helpers ──────────────────────────────────────────────────────

fn file_stem(id: &str) -> &str {
    id.strip_prefix("resp_").unwrap_or(id)
}

fn messages_path(dir: &Path, id: &str) -> PathBuf {
    dir.join("messages")
        .join(format!("{}.jsonl", file_stem(id)))
}

fn delete_messages_file(dir: &Path, id: &str) {
    let path = messages_path(dir, id);
    tokio::spawn(async move {
        let _ = tokio::fs::remove_file(&path).await;
    });
}

fn serde_to_io_err(e: serde_json::Error) -> std::io::Error {
    std::io::Error::other(e)
}

async fn write_messages(path: &Path, items: &[MessageRequest]) -> Result<(), std::io::Error> {
    let mut file = tokio::fs::File::create(path).await?;
    for item in items {
        let json = serde_json::to_string(item).map_err(serde_to_io_err)?;
        file.write_all(format!("{}\n", json).as_bytes()).await?;
    }
    file.flush().await?;
    tracing::debug!("saved {} messages to {:?}", items.len(), path);
    Ok(())
}

async fn read_messages(path: &Path) -> Option<Vec<MessageRequest>> {
    if !path.exists() {
        return Some(vec![]);
    }
    let file = tokio::fs::File::open(path).await.ok()?;
    let mut lines = BufReader::new(file).lines();
    let mut messages = Vec::new();
    while let Ok(Some(line)) = lines.next_line().await {
        if line.is_empty() {
            continue;
        }
        if let Ok(message) = serde_json::from_str::<MessageRequest>(&line) {
            messages.push(message);
        }
    }
    Some(messages)
}
