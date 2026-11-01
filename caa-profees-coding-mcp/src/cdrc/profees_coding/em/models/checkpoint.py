from __future__ import annotations

import importlib
import json
import random
import sqlite3
import threading
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import closing, contextmanager
from dataclasses import asdict, is_dataclass
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    SerializerProtocol,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.utils import search_where

try:
    # Only import if you actually use Pydantic; safe to leave as optional
    from pydantic import BaseModel
except Exception:  # pragma: no cover
    BaseModel = None  # type: ignore[assignment]

_AIO_ERROR_MSG = (
    "The SqliteSaver does not support async methods. "
    "Consider using AsyncSqliteSaver instead.\n"
    "from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver\n"
    "Note: AsyncSqliteSaver requires the aiosqlite package to use.\n"
    "Install with:\n`pip install aiosqlite`\n"
    "See https://langchain-ai.github.io/langgraph/reference/checkpoints/#langgraph.checkpoint.sqlite.aio.AsyncSqliteSaver"
    "for more information."
)


class SqliteCheckpointer(BaseCheckpointSaver[str]):
    """A checkpoint saver that stores checkpoints in a SQLite database."""

    conn: sqlite3.Connection
    is_setup: bool

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        serde: SerializerProtocol | None = None,
    ) -> None:
        # serde is used for checkpoints + writes
        super().__init__(serde=serde)
        self.jsonplus_serde = JsonPlusSerializer()
        self.conn = conn
        self.is_setup = False
        self.lock = threading.Lock()

    # ---------------------------------------------------------------------
    # JSON <-> Python conversion helpers for metadata (and anything else
    # you want to be JSON-serializable, including EncounterTypeDetails).
    # ---------------------------------------------------------------------
    _TYPE_TAG = "__lg_type__"
    _DATA_KEY = "data"

    def _to_primitive(self, obj: Any) -> Any:
        """Recursively convert non-JSON objects into JSON-serializable primitives.

        Non-JSON objects are encoded as:
            {"__lg_type__": "module:QualName", "data": <primitive form>}
        """

        # Basic JSON types
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj

        # Sequences
        if isinstance(obj, (list, tuple, set)):
            return [self._to_primitive(x) for x in obj]

        # Dict
        if isinstance(obj, dict):
            return {k: self._to_primitive(v) for k, v in obj.items()}

        # Pydantic model
        if BaseModel is not None and isinstance(obj, BaseModel):
            cls = obj.__class__
            class_path = f"{cls.__module__}:{cls.__qualname__}"
            data = obj.model_dump()
            return {
                self._TYPE_TAG: class_path,
                self._DATA_KEY: self._to_primitive(data),
            }

        # Dataclass
        if is_dataclass(obj):
            cls = obj.__class__
            class_path = f"{cls.__module__}:{cls.__qualname__}"
            data = asdict(obj)
            return {
                self._TYPE_TAG: class_path,
                self._DATA_KEY: self._to_primitive(data),
            }

        # Fallback: any other object with __dict__
        if hasattr(obj, "__dict__"):
            cls = obj.__class__
            class_path = f"{cls.__module__}:{cls.__qualname__}"
            data = vars(obj)
            return {
                self._TYPE_TAG: class_path,
                self._DATA_KEY: self._to_primitive(data),
            }

        # Last resort: stringify
        return str(obj)

    def _from_primitive(self, obj: Any) -> Any:
        """Reverse _to_primitive: reconstruct objects from tagged dicts."""
        # Basic types
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj

        # List
        if isinstance(obj, list):
            return [self._from_primitive(x) for x in obj]

        # Dict
        if isinstance(obj, dict):
            # Tagged object?
            if self._TYPE_TAG in obj and self._DATA_KEY in obj:
                class_path = obj[self._TYPE_TAG]
                data = self._from_primitive(obj[self._DATA_KEY])

                try:
                    module_name, qualname = class_path.split(":", 1)
                    module = importlib.import_module(module_name)
                    cls = module
                    for part in qualname.split("."):
                        cls = getattr(cls, part)

                    # Try Pydantic-style constructor first
                    if BaseModel is not None and issubclass(cls, BaseModel):
                        return cls.model_validate(data)  # type: ignore[attr-defined]

                    # Otherwise assume standard **kwargs
                    if isinstance(data, dict):
                        return cls(**data)
                    return cls(data)
                except Exception:
                    # If anything fails, just return the decoded data dict
                    return data

            # Plain dict
            return {k: self._from_primitive(v) for k, v in obj.items()}

        # Fallback
        return obj

    # ---------------------------------------------------------------------
    # SqliteSaver implementation
    # ---------------------------------------------------------------------

    @classmethod
    @contextmanager
    def from_conn_string(cls, conn_string: str) -> Iterator[SqliteCheckpointer]:
        with closing(
            sqlite3.connect(
                conn_string,
                check_same_thread=False,
            )
        ) as conn:
            yield cls(conn)

    def setup(self) -> None:
        if self.is_setup:
            return

        self.conn.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS checkpoints (
                thread_id TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL DEFAULT '',
                checkpoint_id TEXT NOT NULL,
                parent_checkpoint_id TEXT,
                type TEXT,
                checkpoint BLOB,
                metadata BLOB,
                PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
            );
            CREATE TABLE IF NOT EXISTS writes (
                thread_id TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL DEFAULT '',
                checkpoint_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                idx INTEGER NOT NULL,
                channel TEXT NOT NULL,
                type TEXT,
                value BLOB,
                PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
            );
            """)

        self.is_setup = True

    @contextmanager
    def cursor(self, transaction: bool = True) -> Iterator[sqlite3.Cursor]:
        with self.lock:
            self.setup()
            cur = self.conn.cursor()
            try:
                yield cur
            finally:
                if transaction:
                    self.conn.commit()
                cur.close()

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        with self.cursor(transaction=False) as cur:
            if checkpoint_id := get_checkpoint_id(config):
                cur.execute(
                    """
                    SELECT thread_id, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata
                     FROM checkpoints WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
                    """,
                    (
                        str(config["configurable"]["thread_id"]),
                        checkpoint_ns,
                        checkpoint_id,
                    ),
                )
            else:
                cur.execute(
                    """
                    SELECT thread_id, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata
                     FROM checkpoints WHERE thread_id = ? AND checkpoint_ns = ? ORDER BY checkpoint_id DESC LIMIT 1
                    """,
                    (str(config["configurable"]["thread_id"]), checkpoint_ns),
                )
            if value := cur.fetchone():
                (
                    thread_id,
                    checkpoint_id,
                    parent_checkpoint_id,
                    type,
                    checkpoint,
                    metadata,
                ) = value
                if not get_checkpoint_id(config):
                    config = {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": checkpoint_id,
                        }
                    }
                cur.execute(
                    """
                    SELECT task_id, channel, type, value FROM writes
                     WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ? ORDER BY task_id, idx
                    """,
                    (
                        str(config["configurable"]["thread_id"]),
                        checkpoint_ns,
                        str(config["configurable"]["checkpoint_id"]),
                    ),
                )

                # --- decode metadata via primitive -> object round trip
                if metadata is not None:
                    raw_meta = json.loads(metadata)
                    decoded_meta = self._from_primitive(raw_meta)
                else:
                    decoded_meta = {}

                return CheckpointTuple(
                    config,
                    self.serde.loads_typed((type, checkpoint)),
                    cast(CheckpointMetadata, decoded_meta),
                    (
                        {
                            "configurable": {
                                "thread_id": thread_id,
                                "checkpoint_ns": checkpoint_ns,
                                "checkpoint_id": parent_checkpoint_id,
                            }
                        }
                        if parent_checkpoint_id
                        else None
                    ),
                    [(task_id, channel, self.serde.loads_typed((type, value))) for task_id, channel, type, value in cur],
                )

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        where, param_values = search_where(config, filter, before)
        query = f"""SELECT thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata
        FROM checkpoints
        {where}
        ORDER BY checkpoint_id DESC"""
        if limit:
            query += f" LIMIT {limit}"
        with self.cursor(transaction=False) as cur, closing(self.conn.cursor()) as wcur:
            cur.execute(query, param_values)
            for (
                thread_id,
                checkpoint_ns,
                checkpoint_id,
                parent_checkpoint_id,
                type,
                checkpoint,
                metadata,
            ) in cur:
                wcur.execute(
                    """
                    SELECT task_id, channel, type, value FROM writes
                     WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ? ORDER BY task_id, idx
                    """,
                    (thread_id, checkpoint_ns, checkpoint_id),
                )

                if metadata is not None:
                    raw_meta = json.loads(metadata)
                    decoded_meta = self._from_primitive(raw_meta)
                else:
                    decoded_meta = {}

                yield CheckpointTuple(
                    {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": checkpoint_id,
                        }
                    },
                    self.serde.loads_typed((type, checkpoint)),
                    cast(CheckpointMetadata, decoded_meta),
                    (
                        {
                            "configurable": {
                                "thread_id": thread_id,
                                "checkpoint_ns": checkpoint_ns,
                                "checkpoint_id": parent_checkpoint_id,
                            }
                        }
                        if parent_checkpoint_id
                        else None
                    ),
                    [(task_id, channel, self.serde.loads_typed((type, value))) for task_id, channel, type, value in wcur],
                )

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        type_, serialized_checkpoint = self.serde.dumps_typed(checkpoint)

        # Convert metadata to JSON-serializable form, including custom types
        metadata_dict = get_checkpoint_metadata(config, metadata)
        primitive_metadata = self._to_primitive(metadata_dict)
        serialized_metadata = json.dumps(primitive_metadata, ensure_ascii=False).encode("utf-8", "ignore")

        with self.cursor() as cur:
            cur.execute(
                """
                INSERT OR REPLACE INTO checkpoints
                 (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(config["configurable"]["thread_id"]),
                    checkpoint_ns,
                    checkpoint["id"],
                    config["configurable"].get("checkpoint_id"),
                    type_,
                    serialized_checkpoint,
                    serialized_metadata,
                ),
            )
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        query = (
            """
            INSERT OR REPLACE INTO writes
              (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, value) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """
            if all(w[0] in WRITES_IDX_MAP for w in writes)
            else """
            INSERT OR IGNORE INTO writes
              (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, value) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """
        )
        with self.cursor() as cur:
            cur.executemany(
                query,
                [
                    (
                        str(config["configurable"]["thread_id"]),
                        str(config["configurable"]["checkpoint_ns"]),
                        str(config["configurable"]["checkpoint_id"]),
                        task_id,
                        WRITES_IDX_MAP.get(channel, idx),
                        channel,
                        *self.serde.dumps_typed(value),
                    )
                    for idx, (channel, value) in enumerate(writes)
                ],
            )

    def delete_thread(self, thread_id: str) -> None:
        with self.cursor() as cur:
            cur.execute(
                "DELETE FROM checkpoints WHERE thread_id = ?",
                (str(thread_id),),
            )
            cur.execute(
                "DELETE FROM writes WHERE thread_id = ?",
                (str(thread_id),),
            )

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        raise NotImplementedError(_AIO_ERROR_MSG)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        raise NotImplementedError(_AIO_ERROR_MSG)
        yield

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        raise NotImplementedError(_AIO_ERROR_MSG)

    def get_next_version(self, current: str | None, channel: None) -> str:
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(current.split(".")[0])
        next_v = current_v + 1
        next_h = random.random()
        return f"{next_v:032}.{next_h:016}"
