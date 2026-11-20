"""Point-in-time feature sample construction for offline backtesting.

The sample warehouse consumes the same two independent products as the replay
warehouse -- published plan contracts and a fact/event store -- but never
touches the ordinary current-value reporting path:

* :class:`~exact_ledger.facts.FactStore` /
  :meth:`~exact_ledger.replay.ReplayWarehouse.import_facts` and
  :meth:`~exact_ledger.replay.ReplayWarehouse.publish` keep working exactly as
  before on their own tables;
* measurement *events* (with their two time axes) live in
  :class:`~exact_ledger.events.EventStore` and are ingested through the
  separate :meth:`SampleWarehouse.import_events` entry point.

A sample spec fixes a prediction anchor ``T``, a knowledge cutoff ``C <= T``
(the earliest-shift perspective: what could have been known when predicting),
a half-open history window ``[T - L, T)`` and a minimum event count ``m``.
Only the sample's own device participates.  For every in-window event the
version selected at cutoff ``C`` is used -- highest revision whose
``knowable_time <= C``; a selected tombstone means the measurement did not
exist at that cutoff and is neither counted nor averaged.  The same event can
therefore serve several anchors through *different* revisions.

The two failure modes stay separate and both are reported instead of being
papered over:

* ``insufficient_events`` -- fewer than ``m`` surviving events;
* ``missing_inputs``      -- a surviving event lacks an input that the
  selected output's reduced, non-zero contract requires.  The event is *not*
  skipped and the remaining values are *not* averaged as if nothing happened;
  no mean is produced.

Samples are immutable: each build gets its own sample id, its own status, its
own per-event sources (event id, revision, knowable time, computed value) and
the plan digest/contract it used.  Events arriving later never rewrite an old
sample, and a sample can be reopened across connections.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import json
import sqlite3
from types import MappingProxyType
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .events import EventStore, StoredEvent, WindowedEvent, require_hour
from .plan import PlanService
from .rational import to_str

__all__ = [
    "SampleError",
    "InvalidSampleError",
    "DuplicateSampleError",
    "UnknownSampleError",
    "SampleSpec",
    "SampleSource",
    "Sample",
    "SampleWarehouse",
    "SAMPLE_COMPLETE",
    "SAMPLE_INSUFFICIENT",
    "SAMPLE_MISSING_INPUTS",
    "SAMPLE_INSUFFICIENT_AND_MISSING_INPUTS",
    "REASON_NOT_KNOWABLE",
    "REASON_WITHDRAWN",
]

SAMPLE_COMPLETE = "complete"
SAMPLE_INSUFFICIENT = "insufficient_events"
SAMPLE_MISSING_INPUTS = "missing_inputs"
SAMPLE_INSUFFICIENT_AND_MISSING_INPUTS = "insufficient_events+missing_inputs"

REASON_NOT_KNOWABLE = "not_knowable_at_cutoff"
REASON_WITHDRAWN = "withdrawn"


class SampleError(ValueError):
    """Base class for sample construction errors."""


class InvalidSampleError(SampleError):
    pass


class DuplicateSampleError(SampleError):
    pass


class UnknownSampleError(SampleError):
    pass


@dataclass(frozen=True)
class SampleSpec:
    sample_id: str
    device_id: str
    plan_id: str
    output: str
    prediction_time: int
    cutoff: int
    window_length: int
    min_events: int

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> "SampleSpec":
        if not isinstance(raw, Mapping):
            raise InvalidSampleError("sample spec must be an object")
        for key in (
            "sample_id",
            "device_id",
            "plan_id",
            "output",
            "prediction_time",
            "cutoff",
            "window_length",
            "min_events",
        ):
            if key not in raw:
                raise InvalidSampleError(f"sample spec is missing {key!r}")
        sample_id = raw["sample_id"]
        device_id = raw["device_id"]
        plan_id = raw["plan_id"]
        output = raw["output"]
        for key, value in (
            ("sample_id", sample_id),
            ("device_id", device_id),
            ("plan_id", plan_id),
            ("output", output),
        ):
            if not isinstance(value, str) or not value.strip():
                raise InvalidSampleError(
                    f"sample spec {key} must be a non-empty string, got {value!r}"
                )
        prediction_time = require_hour(
            raw["prediction_time"], "sample spec prediction_time"
        )
        cutoff = require_hour(raw["cutoff"], "sample spec cutoff")
        window_length = require_hour(
            raw["window_length"], "sample spec window_length"
        )
        if window_length < 1:
            raise InvalidSampleError(
                f"window_length must be positive, got {window_length}"
            )
        if cutoff > prediction_time:
            raise InvalidSampleError(
                f"cutoff {cutoff} must not exceed prediction_time "
                f"{prediction_time}"
            )
        min_events = raw["min_events"]
        if isinstance(min_events, bool) or not isinstance(min_events, int):
            raise InvalidSampleError(
                f"min_events must be a positive integer, got {min_events!r}"
            )
        if min_events < 1:
            raise InvalidSampleError(
                f"min_events must be >= 1, got {min_events}"
            )
        return cls(
            sample_id=sample_id,
            device_id=device_id,
            plan_id=plan_id,
            output=output,
            prediction_time=prediction_time,
            cutoff=cutoff,
            window_length=window_length,
            min_events=min_events,
        )

    def as_dict(self) -> Dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "device_id": self.device_id,
            "plan_id": self.plan_id,
            "output": self.output,
            "prediction_time": self.prediction_time,
            "cutoff": self.cutoff,
            "window_length": self.window_length,
            "min_events": self.min_events,
        }


@dataclass(frozen=True)
class SampleSource:
    """One in-window event and how the cutoff view treated it."""

    event_id: str
    event_time: int
    revision: Optional[int]
    knowable_time: Optional[int]
    withdrawn: bool
    included: bool
    excluded_reason: Optional[str]
    value: Optional[Fraction]

    def as_dict(self) -> Dict[str, object]:
        return {
            "event_id": self.event_id,
            "event_time": self.event_time,
            "revision": self.revision,
            "knowable_time": self.knowable_time,
            "withdrawn": self.withdrawn,
            "included": self.included,
            "excluded_reason": self.excluded_reason,
            "value": None if self.value is None else to_str(self.value),
        }


@dataclass(frozen=True)
class Sample:
    """An immutable backtest sample with full traceability."""

    sample_id: str
    device_id: str
    plan_id: str
    plan_digest: str
    output: str
    prediction_time: int
    cutoff: int
    window_start: int
    window_end: int
    window_length: int
    min_events: int
    status: str
    event_count: int
    mean: Optional[Fraction]
    reasons: Tuple[Mapping[str, object], ...]
    sources: Tuple[SampleSource, ...]
    contributions: Tuple[Mapping[str, object], ...]
    contract: Mapping[str, object]
    created_seq: int

    @property
    def complete(self) -> bool:
        return self.status == SAMPLE_COMPLETE

    def as_dict(self) -> Dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "device_id": self.device_id,
            "plan_id": self.plan_id,
            "plan_digest": self.plan_digest,
            "output": self.output,
            "prediction_time": self.prediction_time,
            "cutoff": self.cutoff,
            "window": [self.window_start, self.window_end],
            "window_length": self.window_length,
            "min_events": self.min_events,
            "status": self.status,
            "event_count": self.event_count,
            "mean": None if self.mean is None else to_str(self.mean),
            "reasons": [dict(item) for item in self.reasons],
            "sources": [source.as_dict() for source in self.sources],
            "contributions": [dict(item) for item in self.contributions],
            "contract": dict(self.contract),
            "created_seq": self.created_seq,
        }

    def trace(self, event_id: Optional[str] = None) -> Dict[str, object]:
        """Trace the sample back to the exact event revisions it adopted."""
        sources = [
            source.as_dict()
            for source in self.sources
            if event_id is None or source.event_id == event_id
        ]
        contributions = [
            dict(item)
            for item in self.contributions
            if event_id is None or item["event_id"] == event_id
        ]
        return {
            "sample_id": self.sample_id,
            "plan_id": self.plan_id,
            "plan_digest": self.plan_digest,
            "output": self.output,
            "prediction_time": self.prediction_time,
            "cutoff": self.cutoff,
            "window": [self.window_start, self.window_end],
            "status": self.status,
            "sources": sources,
            "contributions": contributions,
        }


class SampleWarehouse:
    """Builds and stores immutable point-in-time feature samples."""

    def __init__(
        self,
        connection: sqlite3.Connection,
        plans: PlanService,
        events: Optional[EventStore] = None,
    ) -> None:
        self._conn = connection
        self._conn.row_factory = sqlite3.Row
        self.plans = plans
        self.events = events or EventStore(connection)
        self._create_schema()

    def _create_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS samples (
                sample_id TEXT PRIMARY KEY,
                device_id TEXT NOT NULL,
                plan_id TEXT NOT NULL,
                plan_digest TEXT NOT NULL,
                output TEXT NOT NULL,
                prediction_time INTEGER NOT NULL,
                cutoff INTEGER NOT NULL,
                window_start INTEGER NOT NULL,
                window_end INTEGER NOT NULL,
                window_length INTEGER NOT NULL,
                min_events INTEGER NOT NULL,
                status TEXT NOT NULL,
                event_count INTEGER NOT NULL,
                mean_num TEXT,
                mean_den TEXT,
                reasons TEXT NOT NULL,
                contract TEXT NOT NULL,
                created_seq INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sample_sources (
                sample_id TEXT NOT NULL,
                event_id TEXT NOT NULL,
                event_time INTEGER NOT NULL,
                revision INTEGER,
                knowable_time INTEGER,
                is_withdrawn INTEGER NOT NULL,
                included INTEGER NOT NULL,
                excluded_reason TEXT,
                value_num TEXT,
                value_den TEXT,
                PRIMARY KEY (sample_id, event_id),
                FOREIGN KEY (sample_id) REFERENCES samples (sample_id)
            );
            CREATE TABLE IF NOT EXISTS sample_contributions (
                sample_id TEXT NOT NULL,
                event_id TEXT NOT NULL,
                revision INTEGER NOT NULL,
                output TEXT NOT NULL,
                input TEXT NOT NULL,
                is_constant INTEGER NOT NULL,
                coefficient TEXT NOT NULL,
                value TEXT NOT NULL,
                contribution TEXT NOT NULL,
                PRIMARY KEY (sample_id, event_id, output, is_constant, input)
            );
            CREATE INDEX IF NOT EXISTS sample_sources_device_idx
                ON sample_sources (sample_id, event_time);
            """
        )
        self._conn.commit()

    # ------------------------------------------------------ event ingestion
    def import_events(
        self, plan_id: str, raw_events: Sequence[object]
    ) -> int:
        """Ingest a bitemporal measurement event batch for a plan (atomic)."""
        plan = self.plans.get(plan_id)
        return self.events.import_events(raw_events, plan.inputs)

    def import_events_declared(
        self, declared_inputs: Sequence[str], raw_events: Sequence[object]
    ) -> int:
        return self.events.import_events(raw_events, declared_inputs)

    # ------------------------------------------------------------- building
    def build_sample(
        self, raw_spec: Mapping[str, object]
    ) -> Sample:
        """Build, persist and return one sample."""
        return self.build_samples([raw_spec])[0]

    def build_samples(
        self, raw_specs: Sequence[Mapping[str, object]]
    ) -> List[Sample]:
        """Validate and build a whole request atomically.

        Structural problems (bad specs, unknown plan/output, duplicated or
        already-used sample ids) reject the request before anything is
        evaluated or stored.  Data-level outcomes such as a short window or a
        missing input are individual sample statuses, never request errors.
        """
        specs = [SampleSpec.from_mapping(raw) for raw in raw_specs]
        seen: Dict[str, SampleSpec] = {}
        for spec in specs:
            if spec.sample_id in seen:
                raise DuplicateSampleError(
                    f"sample id {spec.sample_id!r} is used twice in one request"
                )
            if self._sample_exists(spec.sample_id):
                raise DuplicateSampleError(
                    f"sample {spec.sample_id!r} already exists; samples are "
                    "immutable"
                )
            seen[spec.sample_id] = spec
            # Fails early on an unknown plan or output.
            self.plans.get(spec.plan_id).contract(spec.output)

        samples = [self._evaluate(spec) for spec in specs]
        with self._conn:
            for sample in samples:
                self._persist(sample)
        return [self.reopen(sample.sample_id) for sample in samples]

    def _evaluate(self, spec: SampleSpec) -> Sample:
        plan = self.plans.get(spec.plan_id)
        contract = plan.contract(spec.output)
        contract_payload = {
            "constant": to_str(contract.constant),
            "input_coefficients": {
                name: to_str(coef)
                for name, coef in contract.input_coefficients.items()
            },
        }
        window_start = spec.prediction_time - spec.window_length
        window_end = spec.prediction_time
        windowed = self.events.window_cut(
            spec.device_id, window_start, window_end, spec.cutoff
        )

        sources: List[SampleSource] = []
        reasons: List[Mapping[str, object]] = []
        contributions: List[Mapping[str, object]] = []
        included: List[Tuple[WindowedEvent, StoredEvent, Fraction]] = []

        for item in windowed:
            selected = item.selected
            if selected is None:
                sources.append(
                    SampleSource(
                        event_id=item.event_id,
                        event_time=item.event_time,
                        revision=None,
                        knowable_time=None,
                        withdrawn=False,
                        included=False,
                        excluded_reason=REASON_NOT_KNOWABLE,
                        value=None,
                    )
                )
                reasons.append(
                    {
                        "kind": REASON_NOT_KNOWABLE,
                        "event_id": item.event_id,
                        "event_time": item.event_time,
                        "cutoff": spec.cutoff,
                    }
                )
                continue
            if selected.withdrawn:
                sources.append(
                    SampleSource(
                        event_id=item.event_id,
                        event_time=item.event_time,
                        revision=selected.revision,
                        knowable_time=selected.knowable_time,
                        withdrawn=True,
                        included=False,
                        excluded_reason=REASON_WITHDRAWN,
                        value=None,
                    )
                )
                reasons.append(
                    {
                        "kind": REASON_WITHDRAWN,
                        "event_id": item.event_id,
                        "event_time": item.event_time,
                        "revision": selected.revision,
                        "knowable_time": selected.knowable_time,
                    }
                )
                continue

            missing = [
                name
                for name in contract.input_coefficients
                if name not in selected.inputs
            ]
            if missing:
                sources.append(
                    SampleSource(
                        event_id=item.event_id,
                        event_time=item.event_time,
                        revision=selected.revision,
                        knowable_time=selected.knowable_time,
                        withdrawn=False,
                        included=False,
                        excluded_reason="missing_required_inputs",
                        value=None,
                    )
                )
                for name in missing:
                    reasons.append(
                        {
                            "kind": "missing_required_input",
                            "event_id": item.event_id,
                            "event_time": item.event_time,
                            "revision": selected.revision,
                            "input": name,
                            "output": spec.output,
                            "coefficient": to_str(
                                contract.coefficient_of(name)
                            ),
                        }
                    )
                continue

            value = contract.constant
            contributions.append(
                {
                    "event_id": item.event_id,
                    "revision": selected.revision,
                    "output": spec.output,
                    "input": "",
                    "is_constant": True,
                    "coefficient": "1",
                    "value": to_str(contract.constant),
                    "contribution": to_str(contract.constant),
                }
            )
            for input_name, coefficient in contract.input_coefficients.items():
                input_value = selected.inputs[input_name]
                contribution = coefficient * input_value
                value += contribution
                contributions.append(
                    {
                        "event_id": item.event_id,
                        "revision": selected.revision,
                        "output": spec.output,
                        "input": input_name,
                        "is_constant": False,
                        "coefficient": to_str(coefficient),
                        "value": to_str(input_value),
                        "contribution": to_str(contribution),
                    }
                )
            included.append((item, selected, value))
            sources.append(
                SampleSource(
                    event_id=item.event_id,
                    event_time=item.event_time,
                    revision=selected.revision,
                    knowable_time=selected.knowable_time,
                    withdrawn=False,
                    included=True,
                    excluded_reason=None,
                    value=value,
                )
            )

        event_count = len(included)
        has_missing = any(
            reason["kind"] == "missing_required_input" for reason in reasons
        )
        insufficient = event_count < spec.min_events
        if insufficient and has_missing:
            status = SAMPLE_INSUFFICIENT_AND_MISSING_INPUTS
        elif insufficient:
            status = SAMPLE_INSUFFICIENT
        elif has_missing:
            status = SAMPLE_MISSING_INPUTS
        else:
            status = SAMPLE_COMPLETE
        if insufficient:
            reasons.append(
                {
                    "kind": "below_min_events",
                    "event_count": event_count,
                    "min_events": spec.min_events,
                }
            )

        mean = (
            sum((value for _, _, value in included), Fraction(0)) / event_count
            if not has_missing and event_count > 0
            else None
        )
        # A short window is reported via the status but the equal-weight mean
        # of every fully evaluable event is still given honestly.  A missing
        # required input suppresses the mean instead: silently averaging the
        # other events would present a skipped value as a complete window.

        return Sample(
            sample_id=spec.sample_id,
            device_id=spec.device_id,
            plan_id=spec.plan_id,
            plan_digest=plan.digest,
            output=spec.output,
            prediction_time=spec.prediction_time,
            cutoff=spec.cutoff,
            window_start=window_start,
            window_end=window_end,
            window_length=spec.window_length,
            min_events=spec.min_events,
            status=status,
            event_count=event_count,
            mean=mean,
            reasons=tuple(reasons),
            sources=tuple(sources),
            contributions=tuple(contributions),
            contract=MappingProxyType(contract_payload),
            created_seq=0,
        )

    # ------------------------------------------------------------- reading
    def reopen(self, sample_id: str) -> Sample:
        row = self._conn.execute(
            "SELECT * FROM samples WHERE sample_id = ?", (sample_id,)
        ).fetchone()
        if row is None:
            raise UnknownSampleError(f"unknown sample {sample_id!r}")

        sources: List[SampleSource] = []
        for item in self._conn.execute(
            "SELECT * FROM sample_sources WHERE sample_id = ? "
            "ORDER BY event_time, event_id",
            (sample_id,),
        ):
            value = (
                Fraction(int(item["value_num"]), int(item["value_den"]))
                if item["value_num"] is not None
                else None
            )
            sources.append(
                SampleSource(
                    event_id=item["event_id"],
                    event_time=item["event_time"],
                    revision=item["revision"],
                    knowable_time=item["knowable_time"],
                    withdrawn=bool(item["is_withdrawn"]),
                    included=bool(item["included"]),
                    excluded_reason=item["excluded_reason"],
                    value=value,
                )
            )

        contributions: List[Mapping[str, object]] = []
        for item in self._conn.execute(
            "SELECT * FROM sample_contributions WHERE sample_id = ? "
            "ORDER BY event_id, revision, is_constant DESC, input",
            (sample_id,),
        ):
            contributions.append(
                {
                    "event_id": item["event_id"],
                    "revision": item["revision"],
                    "output": item["output"],
                    "input": item["input"],
                    "is_constant": bool(item["is_constant"]),
                    "coefficient": item["coefficient"],
                    "value": item["value"],
                    "contribution": item["contribution"],
                }
            )

        mean = (
            Fraction(int(row["mean_num"]), int(row["mean_den"]))
            if row["mean_num"] is not None
            else None
        )
        return Sample(
            sample_id=row["sample_id"],
            device_id=row["device_id"],
            plan_id=row["plan_id"],
            plan_digest=row["plan_digest"],
            output=row["output"],
            prediction_time=row["prediction_time"],
            cutoff=row["cutoff"],
            window_start=row["window_start"],
            window_end=row["window_end"],
            window_length=row["window_length"],
            min_events=row["min_events"],
            status=row["status"],
            event_count=row["event_count"],
            mean=mean,
            reasons=tuple(json.loads(row["reasons"])),
            sources=tuple(sources),
            contributions=tuple(contributions),
            contract=MappingProxyType(json.loads(row["contract"])),
            created_seq=row["created_seq"],
        )

    def list_samples(
        self, device_id: Optional[str] = None
    ) -> List[Dict[str, object]]:
        if device_id is None:
            rows = self._conn.execute(
                "SELECT sample_id, device_id, plan_id, output, status, "
                "event_count, prediction_time, cutoff, created_seq "
                "FROM samples ORDER BY created_seq"
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT sample_id, device_id, plan_id, output, status, "
                "event_count, prediction_time, cutoff, created_seq "
                "FROM samples WHERE device_id = ? ORDER BY created_seq",
                (device_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    # ----------------------------------------------------------- internals
    def _sample_exists(self, sample_id: str) -> bool:
        return (
            self._conn.execute(
                "SELECT 1 FROM samples WHERE sample_id = ?", (sample_id,)
            ).fetchone()
            is not None
        )

    def _next_seq(self) -> int:
        row = self._conn.execute(
            "SELECT COALESCE(MAX(created_seq), 0) + 1 FROM samples"
        ).fetchone()
        return int(row[0])

    def _persist(self, sample: Sample) -> None:
        seq = self._next_seq()
        self._conn.execute(
            "INSERT INTO samples (sample_id, device_id, plan_id, plan_digest, "
            "output, prediction_time, cutoff, window_start, window_end, "
            "window_length, min_events, status, event_count, mean_num, "
            "mean_den, reasons, contract, created_seq) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                sample.sample_id,
                sample.device_id,
                sample.plan_id,
                sample.plan_digest,
                sample.output,
                sample.prediction_time,
                sample.cutoff,
                sample.window_start,
                sample.window_end,
                sample.window_length,
                sample.min_events,
                sample.status,
                sample.event_count,
                None if sample.mean is None else str(sample.mean.numerator),
                None if sample.mean is None else str(sample.mean.denominator),
                json.dumps(list(sample.reasons), sort_keys=True),
                json.dumps(dict(sample.contract), sort_keys=True),
                seq,
            ),
        )
        for source in sample.sources:
            self._conn.execute(
                "INSERT INTO sample_sources (sample_id, event_id, event_time, "
                "revision, knowable_time, is_withdrawn, included, "
                "excluded_reason, value_num, value_den) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    sample.sample_id,
                    source.event_id,
                    source.event_time,
                    source.revision,
                    source.knowable_time,
                    int(source.withdrawn),
                    int(source.included),
                    source.excluded_reason,
                    None if source.value is None else str(source.value.numerator),
                    None if source.value is None else str(source.value.denominator),
                ),
            )
        for item in sample.contributions:
            self._conn.execute(
                "INSERT INTO sample_contributions (sample_id, event_id, "
                "revision, output, input, is_constant, coefficient, value, "
                "contribution) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    sample.sample_id,
                    item["event_id"],
                    item["revision"],
                    item["output"],
                    item["input"],
                    int(bool(item["is_constant"])),
                    item["coefficient"],
                    item["value"],
                    item["contribution"],
                ),
            )
