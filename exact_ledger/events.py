"""Bitemporal measurement event storage (SQLite).

A measurement *event* is one revision of a device reading.  It carries two
independent integer-hour time axes:

* ``event_time``     -- when the measurement physically happened;
* ``knowable_time``  -- the first hour at which *this revision* could have
  been known.  A correction that arrives later has a later ``knowable_time``
  and must never leak into a prediction made before that hour.

Every event is keyed by a stable ``event_id`` plus a positive ``revision`` and
carries either a **sparse** set of planned inputs (a missing input is absent,
never zero) or an explicit withdrawal marker (a tombstone).  The store
enforces the ingestion rules:

* an ``event_id`` is permanently bound to one ``(device_id, event_time)``
  pair -- a later claim that contradicts the binding rejects the batch;
* revisions may arrive in any order (out-of-order ingestion is normal);
* re-importing the same ``(event_id, revision)`` with identical canonical
  content is an idempotent replay; different content rejects the whole batch
  and writes nothing;
* the first knowable time is part of the revision's canonical content, so a
  redelivery that tries to change it is a conflict rather than a silent
  rewrite;
* nothing is ever deleted or overwritten: a received higher revision cannot
  erase a lower revision's historical visibility.

Point-in-time selection (see :meth:`EventStore.window_cut`) first restricts
the visible revisions to ``knowable_time <= cutoff`` and only then takes the
highest revision.  A selected tombstone means the measurement did not exist
at that cutoff.

The store knows nothing about plan outputs -- it only owns event identity,
versions and the two time axes.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import sqlite3
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .facts import canonical_inputs, require_name, require_revision
from .rational import as_fraction, to_str

__all__ = [
    "EventError",
    "InvalidEventError",
    "EventIdentityError",
    "EventConflictError",
    "StoredEvent",
    "WindowedEvent",
    "EventStore",
    "event_content_hash",
]


class EventError(ValueError):
    """Base class for rejected measurement events."""


class InvalidEventError(EventError):
    pass


class EventIdentityError(EventError):
    """An ``event_id`` was rebound to a different device or event time."""


class EventConflictError(EventError):
    """Same ``(event_id, revision)`` redelivered with different content."""


def require_hour(value: object, what: str) -> int:
    """Times are integer hours; booleans and floats are rejected."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidEventError(f"{what} must be an integer hour, got {value!r}")
    return value


def require_positive_int(value: object, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise InvalidEventError(f"{what} must be a positive integer, got {value!r}")
    return value


def event_content_hash(
    device_id: str,
    event_time: int,
    knowable_time: int,
    withdrawn: bool,
    inputs: Mapping[str, Fraction],
) -> str:
    """Hash of the canonical revision content.

    The knowable time is included on purpose: the first hour at which a
    revision became knowable is an intrinsic property of that revision, so
    redelivery with a different knowable time conflicts instead of silently
    moving the visibility boundary.
    """
    payload = {
        "device_id": device_id,
        "event_time": event_time,
        "knowable_time": knowable_time,
        "withdrawn": withdrawn,
        "inputs": canonical_inputs(inputs),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class StoredEvent:
    event_id: str
    revision: int
    device_id: str
    event_time: int
    knowable_time: int
    withdrawn: bool
    inputs: Mapping[str, Fraction]

    @property
    def content_hash(self) -> str:
        return event_content_hash(
            self.device_id,
            self.event_time,
            self.knowable_time,
            self.withdrawn,
            self.inputs,
        )


@dataclass(frozen=True)
class WindowedEvent:
    """One in-window event and the version selected at a knowledge cutoff.

    ``selected`` is ``None`` when no revision of this event was knowable by
    the cutoff; otherwise it is the highest knowable revision, which may be a
    withdrawal tombstone.
    """

    event_id: str
    device_id: str
    event_time: int
    selected: Optional[StoredEvent]


class EventStore:
    """Append-only, SQLite-backed bitemporal measurement store."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._conn = connection
        self._conn.row_factory = sqlite3.Row
        self._create_schema()

    def _create_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS event_identity (
                event_id TEXT PRIMARY KEY,
                device_id TEXT NOT NULL,
                event_time INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS measurement_events (
                event_id TEXT NOT NULL,
                revision INTEGER NOT NULL,
                device_id TEXT NOT NULL,
                event_time INTEGER NOT NULL,
                knowable_time INTEGER NOT NULL,
                is_withdrawn INTEGER NOT NULL,
                content_hash TEXT NOT NULL,
                imported_seq INTEGER NOT NULL,
                PRIMARY KEY (event_id, revision),
                FOREIGN KEY (event_id) REFERENCES event_identity (event_id)
            );
            CREATE TABLE IF NOT EXISTS event_inputs (
                event_id TEXT NOT NULL,
                revision INTEGER NOT NULL,
                name TEXT NOT NULL,
                num TEXT NOT NULL,
                den TEXT NOT NULL,
                PRIMARY KEY (event_id, revision, name),
                FOREIGN KEY (event_id, revision)
                    REFERENCES measurement_events (event_id, revision)
                    ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS event_identity_device_time_idx
                ON event_identity (device_id, event_time);
            CREATE INDEX IF NOT EXISTS measurement_events_knowable_idx
                ON measurement_events (event_id, knowable_time, revision);
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------- reading
    def _load(self, row: sqlite3.Row) -> StoredEvent:
        inputs = {
            item["name"]: Fraction(int(item["num"]), int(item["den"]))
            for item in self._conn.execute(
                "SELECT name, num, den FROM event_inputs "
                "WHERE event_id = ? AND revision = ? ORDER BY name",
                (row["event_id"], row["revision"]),
            )
        }
        return StoredEvent(
            event_id=row["event_id"],
            revision=row["revision"],
            device_id=row["device_id"],
            event_time=row["event_time"],
            knowable_time=row["knowable_time"],
            withdrawn=bool(row["is_withdrawn"]),
            inputs=inputs,
        )

    def identity(self, event_id: str) -> Optional[Tuple[str, int]]:
        """Return the bound ``(device_id, event_time)`` for an event, if any."""
        row = self._conn.execute(
            "SELECT device_id, event_time FROM event_identity WHERE event_id = ?",
            (event_id,),
        ).fetchone()
        if row is None:
            return None
        return row["device_id"], row["event_time"]

    def identities(
        self, device_id: Optional[str] = None
    ) -> Tuple[Tuple[str, str, int], ...]:
        if device_id is None:
            rows = self._conn.execute(
                "SELECT event_id, device_id, event_time FROM event_identity "
                "ORDER BY event_id"
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT event_id, device_id, event_time FROM event_identity "
                "WHERE device_id = ? ORDER BY event_id",
                (device_id,),
            ).fetchall()
        return tuple(
            (row["event_id"], row["device_id"], row["event_time"]) for row in rows
        )

    def get_version(self, event_id: str, revision: int) -> Optional[StoredEvent]:
        row = self._conn.execute(
            "SELECT * FROM measurement_events WHERE event_id = ? AND revision = ?",
            (event_id, revision),
        ).fetchone()
        return self._load(row) if row else None

    def all_versions(self, event_id: str) -> List[StoredEvent]:
        rows = self._conn.execute(
            "SELECT * FROM measurement_events WHERE event_id = ? ORDER BY revision",
            (event_id,),
        ).fetchall()
        return [self._load(row) for row in rows]

    def point_in_time(
        self, cutoff: int, device_id: Optional[str] = None
    ) -> List[StoredEvent]:
        """Select the highest revision knowable at ``cutoff`` for every event.

    Revisions with ``knowable_time > cutoff`` are ignored *before* taking the
    maximum revision, so a late correction never reaches an earlier cutoff.
    Withdrawal tombstones are returned like any other selected version; the
    caller decides that the measurement then did not exist.
    """
        sql = [
            "SELECT m.* FROM measurement_events m",
            "JOIN (",
            "    SELECT event_id, MAX(revision) AS revision",
            "    FROM measurement_events WHERE knowable_time <= ?",
            "    GROUP BY event_id",
            ") latest ON m.event_id = latest.event_id",
            "          AND m.revision = latest.revision",
        ]
        params: List[object] = [cutoff]
        if device_id is not None:
            sql.append("WHERE m.device_id = ?")
            params.append(device_id)
        sql.append("ORDER BY m.event_id")
        rows = self._conn.execute("\n".join(sql), params).fetchall()
        return [self._load(row) for row in rows]

    def window_cut(
        self, device_id: str, start: int, end: int, cutoff: int
    ) -> List[WindowedEvent]:
        """All of one device's events with ``start <= event_time < end``.

        Each row carries the version selected under knowledge cutoff
        ``cutoff`` (highest revision with ``knowable_time <= cutoff``), or no
        selected version when the event was not yet knowable.  The window is
        half-open: an event occurring exactly at ``end`` is excluded.
        """
        rows = self._conn.execute(
            """
            SELECT i.event_id AS event_id, i.device_id AS device_id,
                   i.event_time AS event_time,
                   m.event_id AS selected_event, m.revision AS revision
            FROM event_identity i
            LEFT JOIN (
                SELECT v.event_id AS event_id, v.revision AS revision
                FROM measurement_events v
                JOIN (
                    SELECT event_id, MAX(revision) AS revision
                    FROM measurement_events
                    WHERE knowable_time <= ?
                    GROUP BY event_id
                ) pick
                  ON v.event_id = pick.event_id AND v.revision = pick.revision
            ) m ON m.event_id = i.event_id
            WHERE i.device_id = ?
              AND i.event_time >= ?
              AND i.event_time < ?
            ORDER BY i.event_time, i.event_id
            """,
            (cutoff, device_id, start, end),
        ).fetchall()
        result: List[WindowedEvent] = []
        for row in rows:
            selected = (
                self.get_version(row["event_id"], row["revision"])
                if row["selected_event"] is not None
                else None
            )
            result.append(
                WindowedEvent(
                    event_id=row["event_id"],
                    device_id=row["device_id"],
                    event_time=row["event_time"],
                    selected=selected,
                )
            )
        return result

    # ------------------------------------------------------------- writing
    def import_events(
        self, raw_events: Sequence[object], declared_inputs: Sequence[str]
    ) -> int:
        """Validate and atomically import an event batch; rows added."""
        prepared = self.prepare(raw_events, declared_inputs)
        return self.write(prepared)

    def prepare(
        self, raw_events: Sequence[object], declared_inputs: Sequence[str]
    ) -> List[StoredEvent]:
        """Validate the whole batch without writing anything.

        Malformed rows, identity rebindings and same-revision conflicts all
        abort the batch before a single row lands.
        """
        if not isinstance(raw_events, (list, tuple)):
            raise InvalidEventError("event batch must be a list")
        declared = set(declared_inputs)
        prepared: List[StoredEvent] = []
        batch_versions: Dict[Tuple[str, int], StoredEvent] = {}
        batch_identity: Dict[str, Tuple[str, int]] = {}

        for index, raw in enumerate(raw_events):
            event = self._parse(raw, declared, index)
            key = (event.event_id, event.revision)
            previous = batch_versions.get(key)
            if previous is not None:
                if previous.content_hash != event.content_hash:
                    raise EventConflictError(
                        f"batch contains revision {event.revision} of event "
                        f"{event.event_id!r} with conflicting content"
                    )
                continue
            batch_versions[key] = event
            prepared.append(event)

            binding = (event.device_id, event.event_time)
            existing_binding = batch_identity.get(event.event_id)
            if existing_binding is not None and existing_binding != binding:
                raise EventIdentityError(
                    f"event {event.event_id!r} is bound to "
                    f"{existing_binding[0]!r} at hour {existing_binding[1]}; "
                    f"the batch rebinds it to {binding[0]!r} at hour {binding[1]}"
                )
            batch_identity[event.event_id] = binding

        for event in prepared:
            stored_binding = self.identity(event.event_id)
            if stored_binding is not None and stored_binding != (
                event.device_id,
                event.event_time,
            ):
                raise EventIdentityError(
                    f"event {event.event_id!r} is already bound to "
                    f"{stored_binding[0]!r} at hour {stored_binding[1]}; "
                    f"cannot rebind to {event.device_id!r} at hour "
                    f"{event.event_time}"
                )
            existing = self.get_version(event.event_id, event.revision)
            if existing is not None and existing.content_hash != event.content_hash:
                raise EventConflictError(
                    f"event {event.event_id!r} revision {event.revision} "
                    "already exists with different canonical content "
                    "(including its first knowable time); the entire batch "
                    "is rejected"
                )
        return prepared

    def _parse(
        self, raw: object, declared: set, index: int
    ) -> StoredEvent:
        if not isinstance(raw, Mapping):
            raise InvalidEventError(f"event {index} must be an object")
        event_id = require_name(raw.get("event_id"), f"event {index} event_id")
        device_id = require_name(raw.get("device_id"), f"event {index} device_id")
        event_time = require_hour(raw.get("event_time"), f"event {index} event_time")
        knowable_time = require_hour(
            raw.get("knowable_time"), f"event {index} knowable_time"
        )
        if knowable_time < event_time:
            raise InvalidEventError(
                f"event {index} ({event_id!r}): knowable_time {knowable_time} "
                f"is earlier than event_time {event_time}"
            )
        revision = require_revision(raw.get("revision"))

        withdrawn = False
        if "withdrawn" in raw:
            if not isinstance(raw["withdrawn"], bool):
                raise InvalidEventError(
                    f"event {index} 'withdrawn' must be a boolean"
                )
            withdrawn = raw["withdrawn"]

        raw_inputs = raw.get("inputs", {})
        if not isinstance(raw_inputs, Mapping):
            raise InvalidEventError(f"event {index} inputs must be an object")
        if withdrawn and raw_inputs:
            raise InvalidEventError(
                f"event {index} ({event_id!r}): a withdrawal tombstone cannot "
                "also carry plan inputs"
            )
        inputs: Dict[str, Fraction] = {}
        for name, value in raw_inputs.items():
            name = require_name(name, f"event {index} input name")
            if name not in declared:
                raise InvalidEventError(
                    f"event {index} ({event_id!r}) supplies input {name!r}, "
                    "which is not declared by the plan"
                )
            try:
                inputs[name] = as_fraction(value)
            except Exception as exc:  # noqa: BLE001
                raise InvalidEventError(
                    f"event {index} input {name!r}: {exc}"
                ) from exc
        return StoredEvent(
            event_id=event_id,
            revision=revision,
            device_id=device_id,
            event_time=event_time,
            knowable_time=knowable_time,
            withdrawn=withdrawn,
            inputs=inputs,
        )

    def write(self, events: Sequence[StoredEvent]) -> int:
        """Write validated rows atomically; returns the number added."""
        added = 0
        with self._conn:
            seq = self._conn.execute(
                "SELECT COALESCE(MAX(imported_seq), 0) FROM measurement_events"
            ).fetchone()[0]
            for event in events:
                if self.get_version(event.event_id, event.revision) is not None:
                    continue  # idempotent replay of an identical revision
                if self.identity(event.event_id) is None:
                    self._conn.execute(
                        "INSERT INTO event_identity (event_id, device_id, "
                        "event_time) VALUES (?, ?, ?)",
                        (event.event_id, event.device_id, event.event_time),
                    )
                seq += 1
                self._conn.execute(
                    "INSERT INTO measurement_events (event_id, revision, "
                    "device_id, event_time, knowable_time, is_withdrawn, "
                    "content_hash, imported_seq) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        event.event_id,
                        event.revision,
                        event.device_id,
                        event.event_time,
                        event.knowable_time,
                        int(event.withdrawn),
                        event.content_hash,
                        seq,
                    ),
                )
                for name, value in sorted(event.inputs.items()):
                    self._conn.execute(
                        "INSERT INTO event_inputs (event_id, revision, name, "
                        "num, den) VALUES (?, ?, ?, ?, ?)",
                        (
                            event.event_id,
                            event.revision,
                            name,
                            str(value.numerator),
                            str(value.denominator),
                        ),
                    )
                added += 1
        return added
