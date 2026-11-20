# exact_ledger — exact linear plans + grouped fact replay

Python 3.11, standard library only (SQLite is used for persistence).  Two
deliberately separate components:

| Component | Module | Owns |
| --- | --- | --- |
| Plan service | `exact_ledger/plan.py` | Plans, DAG validation, exact algebra, published per-output contracts. **Knows nothing about facts.** |
| Replay warehouse | `exact_ledger/replay.py`, `exact_ledger/facts.py` | Fact versions, groups, import rules, eligibility, immutable results and provenance. **Consumes the published plan products.** |
| Backtest samples | `exact_ledger/events.py`, `exact_ledger/samples.py` | Bitemporal measurement events (event time + first knowable time), cutoff-scoped version selection, immutable feature samples with per-event provenance. Separate tables; ordinary import and current-value reporting are unchanged. |

`exact_ledger/rational.py` is the exact-rational layer: ints, `Fraction`, and
string literals `"3"`, `"-3/4"`, `"0.125"`.  No float ever enters evaluation.

## Plan service

A plan has named inputs and nodes; each node is

```
node = constant + Σ coefficientᵢ · refᵢ          (refᵢ = input name or node id)
```

Terms may repeat, coefficients may be zero or negative, and a node may be
referenced from many places.  The graph must be acyclic; unknown refs and
empty ids are rejected.  Merged coefficients that reduce to zero are
**eliminated** from every published product, and names are compared literally
(`"a"`, `"aa"`, `"a_a"` are distinct).

```python
from exact_ledger import PlanService
plans = PlanService()
p1 = plans.publish(plan_definition)          # immutable; same content => same object
p1.contract("z").input_coefficients          # {'a': Fraction(1)}  (b cancelled)
plans.explain("p1", "z")                     # contract + raw node products
plans.explain_coefficient("p1", "z", "b")    # why b's final coefficient is 0
p2 = plans.publish_replacement("p2", "p1", replacement_node)  # copy + replace
```

Published plans are immutable: re-publishing a `plan_id` with different
content is rejected.  Each plan carries a content `digest`.

`explain_coefficient` runs one exact reverse-topological pass over the shared
DAG (never path enumeration).  For `z = u − b`, `u = a + b` it reports
`b`'s final coefficient `0` as `cancelled`, with the signed edge contributions
`u->b: +1`, `z->b: −1` that sum to zero — so a shared source is not double
counted and a zero dependency is never mistaken for a value change.

## Replay warehouse

Facts are keyed by `record_id`, with a positive `revision`, a `group_id`, and a
**subset** of declared inputs.  Missing input = **unavailable**, never zero.

* highest revision wins; earlier revisions are retained but ignored;
* same revision + same canonical content = idempotent replay;
* same revision + different content = **whole batch rejected**, nothing written;
* `group_id` may change only in a *higher, explicitly `complete`* fact
  (an explicit complete fact must cover all declared inputs);
* incomplete facts are legal to store — never coerced to zero, never rejected.

```python
from exact_ledger import open_warehouse
warehouse = open_warehouse(plans)                   # in-memory SQLite; or path=...
warehouse.import_facts("p1", facts)                 # atomic batch
report = warehouse.inspect("p1", ["z", "w"])        # eligibility for the requested set
result = warehouse.publish("p1", ["z"], result_id="initial")
result.total("g", "z")                              # Fraction(4)
result.trace("r1")                                  # revision -> nonzero contributions
warehouse.reopen("initial")                         # immutable, survives reconnect
```

### Eligibility is per requested output set

For every record and every **requested** output, every input in that output's
**reduced, non-zero** dependency map must be present.  Therefore:

* an input with coefficient `0` for all requested outputs may be absent;
* a constant-only output is decidable with no inputs at all;
* if one record lacks a needed input for one requested output, the **whole
  request is rejected** (no half group), and a `rejected` result records the
  `record_id` / `input` / `output` / coefficient reasons;
* plan legality is never treated as “every output has a value”.

### Results are immutable

Every publish attempt gets its own identity and records the plan digest and
fact snapshot.  Old results keep their old plan and old facts; a later
rejection never overwrites an earlier published result.  `recompute` is a full
recompute; `incremental_recompute` reuses per-record values only where the
fact snapshot and reduced contract are unchanged, but always rebuilds totals
from the current record set so constants and group migrations are not missed.

Evaluation is a sparse dot product over the reduced dependency map — the
warehouse never “propagates UNKNOWN” through the original expression, so
`(a+b)−b` with `b` missing is correctly a non-dependency, not “uncertain”.

## Point-in-time feature samples (offline backtest)

A correction that a reading received *later* must not leak into a prediction
made *earlier*.  The bitemporal path therefore lives in two further modules
that share the SQLite database but never touch the ordinary fact/result
tables: `exact_ledger/events.py` (measurement event store) and
`exact_ledger/samples.py` (sample builder).  `import_facts`, `effective()`,
`inspect`/`publish` and every existing result keep their exact behaviour.

A measurement event carries a stable `event_id`, a `device_id`, an integer
hour `event_time`, a positive `revision`, the integer hour
`knowable_time` (first hour *that revision* could be known), a sparse set of
plan inputs, or an explicit `withdrawn` tombstone.  Ingestion rules:

* `event_id` is permanently bound to one `(device_id, event_time)` pair;
* revisions may arrive out of order — nothing is ever deleted, so a received
  high revision cannot erase a low revision's historical visibility;
* same `(event_id, revision)` + identical canonical content is an idempotent
  replay; different content rejects the whole batch;
* the knowable time is part of the canonical content, so a redelivery that
  tries to move it conflicts instead of silently rewriting first visibility;
* `knowable_time < event_time` and tombstones carrying inputs are rejected.

```python
from exact_ledger import open_sample_warehouse
samples = open_sample_warehouse(plans)            # same path/connection OK
samples.import_events(“pz”, events)               # atomic bitemporal batch
sample = samples.build_sample({
    “sample_id”: “s1”, “device_id”: “D”, “plan_id”: “pz”, “output”: “z”,
    “prediction_time”: 11, “cutoff”: 11,         # C <= T, integer hours
    “window_length”: 4, “min_events”: 2,
})
sample.status            # complete | insufficient_events | missing_inputs | ...
sample.event_count       # surviving events in [T-L, T)
sample.mean              # exact Fraction equal-weight mean (or None)
samples.reopen(“s1”)     # immutable; later events never rewrite it
sample.trace(“e2”)       # exact event_id / revision / knowable time / plan
```

Cutoff selection first restricts to revisions with `knowable_time <= C` and
only then takes the highest revision; a selected tombstone means the
measurement did not exist at that cutoff.  The window is half-open
`[T−L, T)` — a reading at exactly `T` is excluded — and only the sample's
own device participates.  The same event can serve several anchors through
different revisions.

The two failure modes stay separate.  `insufficient_events` means fewer than
`m` surviving events (the honest mean of the fully evaluable events is still
returned).  `missing_inputs` means a surviving event lacks an input in the
output's reduced non-zero contract: that event is not skipped and the others
are not averaged as if the window were complete, so **no mean is produced**
and every missing `event_id`/`input`/`output` is recorded in `reasons` and
in the per-event `sources`.  `build_samples([...])` validates a whole
request atomically (duplicate or reused sample ids abort before anything is
stored) while giving each sample its own status, sources and provenance.

`python3.11 run_backtest.py` walks the worked numbers: e1 value 2 at hour 8,
e2 value 5 (rev 1, knowable at 9) then 50 (rev 2, knowable at 12).  At
`T=C=11, L=4, m=2` the mean is `7/2`; at `T=C=13, L=6` it is `26`; replaying
the second anchor with `C=11` returns to `7/2`.  It also demonstrates
out-of-order arrival, a high-version withdrawal leaving one survivor (and
therefore an `insufficient_events` sample), independent devices, sparse
inputs that block the mean, and old samples surviving later corrections.

## Scenario (materials/746)

`python3.11 run_scenario.py` executes the input's operations and the prose
follow-ups:

| Step | z (g) | w (g) |
| --- | --- | --- |
| initial (`p1`, r1 rev1) | 4 | 16 |
| r1.b → 10 (rev2) | 4 (b is zero-dep) | 28 |
| new product `p2` (`z = u − 2b`) | −8 | 28 |
| r1 rev3 `{a:3}` — request `{z}` | 4 | — |
| r1 rev3 — request `{z,w}` | rejected: `r1.b → w` | |
| r1 rev3 — `p2`, request `{z}` | rejected: `r1.b → z` | |
| r1 rev4 restores `b=10` — `p2 {z}` | −8 | |

The original `p1` results stay at 4 throughout.

## Tests

```bash
python3.11 -m unittest discover -s tests -v
```

102 tests cover exact rationals, DAG validation, zero-dependency elimination,
shared-node algebra, cancellation explanation on a wide DAG, fact replay rules,
per-output-set eligibility, immutable result identity, incremental recompute
with group migration, persistence across connections, a 50-node / 10-output
/ 5000-fact run, bitemporal event ingestion (out-of-order arrival, identity
binding, knowable-time conflicts, tombstones), cutoff-scoped window
selection, every sample status and failure mode, sample immutability and
provenance, and coexistence of the replay and sample paths in one database.

## Scale

Documented bounds: ≤ 8 inputs, ≤ 50 nodes, ≤ 10 outputs, ≤ 5000 records per
plan; coefficient and value numerator/denominator magnitudes to 1 000 000; no
output truncation.  Out of scope by design: nonlinear functions, cycle solving,
decimal expression type inference, optimal incremental scheduling.
