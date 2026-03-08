CREATE TABLE IF NOT EXISTS memory (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    namespace TEXT    NOT NULL,
    key       TEXT    NOT NULL,
    value     TEXT    NOT NULL,
    updated_at TEXT   NOT NULL,
    UNIQUE(namespace, key)
);

CREATE TABLE IF NOT EXISTS hitl_flags (
    id              TEXT PRIMARY KEY,
    agent_name      TEXT NOT NULL,
    category        TEXT NOT NULL,
    severity        TEXT NOT NULL DEFAULT 'review',
    summary         TEXT NOT NULL,
    context         TEXT NOT NULL,
    proposed_action TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',
    user_response   TEXT,
    created_at      TEXT NOT NULL,
    resolved_at     TEXT
);
