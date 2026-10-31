"""Local, transactional metadata and append-only task events."""
import hashlib
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('QUANT_DATA_DIR', ROOT / 'var'))
DATA.mkdir(parents=True, exist_ok=True)


def now():
    return datetime.now(timezone.utc).isoformat()


def uid(prefix=''):
    return prefix + uuid.uuid4().hex[:16]


def encode(obj):
    return json.dumps(obj, ensure_ascii=False, allow_nan=False, default=str)


def connect():
    db = sqlite3.connect(DATA / 'platform.sqlite3', timeout=30)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA busy_timeout=30000')
    return db


def init():
    with connect() as db:
        db.executescript('''
        CREATE TABLE IF NOT EXISTS records(kind TEXT, id TEXT PRIMARY KEY, body TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS record_kind ON records(kind);
        CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, body TEXT);
        CREATE INDEX IF NOT EXISTS event_run ON events(run_id,id);
        CREATE TABLE IF NOT EXISTS requests(key TEXT PRIMARY KEY, fingerprint TEXT, result TEXT);
        ''')


def put(kind, obj):
    with connect() as db:
        db.execute('INSERT INTO records VALUES(?,?,?) ON CONFLICT(id) DO UPDATE SET body=excluded.body',
                   (kind, obj['id'], encode(obj)))
    return obj


def get(id, kind=None):
    with connect() as db:
        row = db.execute('SELECT kind,body FROM records WHERE id=?', (id,)).fetchone()
    if not row or (kind and row[0] != kind):
        raise ValueError('记录不存在：' + id)
    return json.loads(row[1])


def update(id, **changes):
    with connect() as db:
        db.execute('BEGIN IMMEDIATE')
        kind, body = db.execute('SELECT kind,body FROM records WHERE id=?', (id,)).fetchone()
        obj = json.loads(body); obj.update(changes); obj['updated_at'] = now()
        db.execute('UPDATE records SET body=? WHERE id=?', (encode(obj), id))
    return obj


def listing(kind, include_deleted=False):
    with connect() as db:
        rows = db.execute('SELECT body FROM records WHERE kind=? ORDER BY rowid DESC', (kind,)).fetchall()
    objects = [json.loads(row[0]) for row in rows]
    return objects if include_deleted else [obj for obj in objects if not obj.get('deleted_at')]


def event(run_id, type, payload):
    with connect() as db:
        body = {'run_id': run_id, 'timestamp': now(), 'type': type, 'schema_version': 1, 'payload': payload}
        cur = db.execute('INSERT INTO events(run_id,body) VALUES(?,?)', (run_id, encode(body)))
        body['event_id'] = cur.lastrowid
    return body


def events(run_id, after=0):
    with connect() as db:
        rows = db.execute('SELECT id,body FROM events WHERE run_id=? AND id>? ORDER BY id LIMIT 1000',
                          (run_id, after)).fetchall()
    return [dict(json.loads(body), event_id=id) for id, body in rows]


def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp'); temp.write_text(encode(obj)); temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


init()
