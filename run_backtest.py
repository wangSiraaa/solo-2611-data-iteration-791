#!/usr/bin/env python3.11
"""Offline backtest demo over bitemporal measurement events.

Shows the point-in-time feature-sample entry point end to end:

* out-of-order revision arrival (a high revision is ingested first);
* a high-version withdrawal tombstone;
* two devices sampled independently;
* the same prediction anchor replayed under different knowledge cutoffs;
* half-open windows (a reading exactly at T is excluded);
* sparse inputs that miss an output's required input;
* immutable samples that later events never rewrite, with full provenance.

The ordinary replay warehouse (current-value reports) runs alongside on the
same database and keeps behaving exactly as before.

Usage:
    python3.11 run_backtest.py
"""

from __future__ import annotations

from fractions import Fraction

from exact_ledger import (
    PlanService,
    ReplayWarehouse,
    SampleWarehouse,
    open_connection,
    to_str,
)


def show(sample):
    print(
        f"  {sample.sample_id:>10} dev={sample.device_id:<3} "
        f"T={sample.prediction_time:>2} C={sample.cutoff:>2} "
        f"window={tuple(sample.as_dict()['window'])} "
        f"status={sample.status:<37} n={sample.event_count} "
        f"mean={'-' if sample.mean is None else to_str(sample.mean)}"
    )
    for source in sample.sources:
        print(
            f"             - {source.event_id} at {source.event_time} "
            f"rev={source.revision} knowable={source.knowable_time} "
            f"included={source.included}"
            + (f" value={to_str(source.value)}" if source.value is not None else "")
            + (f" reason={source.excluded_reason}" if source.excluded_reason else "")
        )


def main():
    plans = PlanService()
    # z = a : the worked example output
    plans.publish({
        "plan_id": "pz",
        "inputs": ["a"],
        "nodes": [{"node_id": "z", "terms": [{"ref": "a", "coefficient": "1"}]}],
        "outputs": ["z"],
    })
    # q = a + b : used to show sparse / missing required inputs
    plans.publish({
        "plan_id": "pq",
        "inputs": ["a", "b"],
        "nodes": [{"node_id": "q", "terms": [
            {"ref": "a", "coefficient": "1"},
            {"ref": "b", "coefficient": "1"},
        ]}],
        "outputs": ["q"],
    })
    # ordinary current-value reporting plan, published up front
    plans.publish({
        "plan_id": "p1",
        "inputs": ["a", "b"],
        "nodes": [
            {"node_id": "u", "terms": [
                {"ref": "a", "coefficient": "1"},
                {"ref": "b", "coefficient": "1"},
            ]},
            {"node_id": "z", "terms": [
                {"ref": "u", "coefficient": "1"},
                {"ref": "b", "coefficient": "-1"},
            ]},
        ],
        "outputs": ["z"],
    })

    conn = open_connection(":memory:")
    samples = SampleWarehouse(conn, plans)
    replay = ReplayWarehouse(conn, plans)  # ordinary reporting, same database

    print("== ingestion: out-of-order arrival (high revision first) ==")
    samples.import_events("pz", [
        # e2's correction arrives before its initial reading is ingested
        {"event_id": "e2", "device_id": "D", "event_time": 9,
         "knowable_time": 12, "revision": 2, "inputs": {"a": "50"}},
    ])
    samples.import_events("pz", [
        {"event_id": "e1", "device_id": "D", "event_time": 8,
         "knowable_time": 8, "revision": 1, "inputs": {"a": "2"}},
        {"event_id": "e2", "device_id": "D", "event_time": 9,
         "knowable_time": 9, "revision": 1, "inputs": {"a": "5"}},
    ])
    print("  stored revisions of e2:",
          [event.revision for event in samples.events.all_versions("e2")])

    print("\n== worked example: T=C=11, L=4, m=2 -> 7/2 ==")
    s1 = samples.build_sample({
        "sample_id": "s1", "device_id": "D", "plan_id": "pz", "output": "z",
        "prediction_time": 11, "cutoff": 11, "window_length": 4, "min_events": 2,
    })
    show(s1)
    assert s1.mean == Fraction(7, 2)

    print("\n== later anchor: T=C=13, L=6, m=2 -> 26 ==")
    s2 = samples.build_sample({
        "sample_id": "s2", "device_id": "D", "plan_id": "pz", "output": "z",
        "prediction_time": 13, "cutoff": 13, "window_length": 6, "min_events": 2,
    })
    show(s2)
    assert s2.mean == Fraction(26)

    print("\n== same anchor, second request with C=11 -> back to 7/2 ==")
    s3 = samples.build_sample({
        "sample_id": "s3", "device_id": "D", "plan_id": "pz", "output": "z",
        "prediction_time": 13, "cutoff": 11, "window_length": 6, "min_events": 2,
    })
    show(s3)
    assert s3.mean == Fraction(7, 2)

    print("\n== reading exactly at prediction time is excluded ==")
    samples.import_events("pz", [
        {"event_id": "edge", "device_id": "D", "event_time": 11,
         "knowable_time": 11, "revision": 1, "inputs": {"a": "100"}},
    ])
    s4 = samples.build_sample({
        "sample_id": "s4", "device_id": "D", "plan_id": "pz", "output": "z",
        "prediction_time": 11, "cutoff": 11, "window_length": 4, "min_events": 2,
    })
    show(s4)
    assert [source.event_id for source in s4.sources] == ["e1", "e2"]

    print("\n== different device, then a high-version withdrawal ==")
    samples.import_events("pz", [
        {"event_id": "w1", "device_id": "W", "event_time": 8,
         "knowable_time": 8, "revision": 1, "inputs": {"a": "2"}},
        {"event_id": "w2", "device_id": "W", "event_time": 9,
         "knowable_time": 9, "revision": 1, "inputs": {"a": "5"}},
    ])
    before = samples.build_sample({
        "sample_id": "w-before", "device_id": "W", "plan_id": "pz", "output": "z",
        "prediction_time": 13, "cutoff": 13, "window_length": 6, "min_events": 2,
    })
    show(before)
    # w1 is withdrawn at hour 14: the measurement did not exist afterwards
    samples.import_events("pz", [
        {"event_id": "w1", "device_id": "W", "event_time": 8,
         "knowable_time": 14, "revision": 2, "withdrawn": True},
    ])
    after = samples.build_sample({
        "sample_id": "w-after", "device_id": "W", "plan_id": "pz", "output": "z",
        "prediction_time": 15, "cutoff": 15, "window_length": 7, "min_events": 2,
    })
    show(after)
    assert after.status == "insufficient_events" and after.event_count == 1
    assert after.mean == Fraction(5)
    # the pre-withdrawal sample is untouched
    assert samples.reopen("w-before").mean == Fraction(7, 2)

    print("\n== sparse input missing a required dependency is not averaged over ==")
    # dedicated device so the sparse events stay out of device D's windows
    samples.import_events("pq", [
        {"event_id": "h1", "device_id": "M", "event_time": 8,
         "knowable_time": 8, "revision": 1,
         "inputs": {"a": "1", "b": "1"}},
        {"event_id": "h2", "device_id": "M", "event_time": 9,
         "knowable_time": 9, "revision": 1, "inputs": {"a": "2"}},  # b absent
    ])
    missing = samples.build_sample({
        "sample_id": "missing", "device_id": "M", "plan_id": "pq", "output": "q",
        "prediction_time": 11, "cutoff": 11, "window_length": 4, "min_events": 1,
    })
    show(missing)
    assert missing.status == "missing_inputs" and missing.mean is None
    assert any(
        reason["kind"] == "missing_required_input" and reason["input"] == "b"
        for reason in missing.reasons
    )

    print("\n== one request builds a batch with independent per-sample state ==")
    batch = samples.build_samples([
        {"sample_id": "b1", "device_id": "D", "plan_id": "pz", "output": "z",
         "prediction_time": 11, "cutoff": 11, "window_length": 4, "min_events": 2},
        {"sample_id": "b2", "device_id": "W", "plan_id": "pz", "output": "z",
         "prediction_time": 15, "cutoff": 15, "window_length": 7, "min_events": 2},
    ])
    for sample in batch:
        show(sample)

    print("\n== provenance: trace a sample back to exact event revisions ==")
    trace = samples.reopen("s2").trace("e2")
    print("  s2 / e2 ->", trace["sources"][0])
    print("  contributions ->", trace["contributions"])
    assert trace["sources"][0]["revision"] == 2
    assert trace["sources"][0]["value"] == "50"

    print("\n== later events never rewrite old samples ==")
    samples.import_events("pz", [
        {"event_id": "e2", "device_id": "D", "event_time": 9,
         "knowable_time": 20, "revision": 3, "inputs": {"a": "500"}},
    ])
    print("  s1 mean still", to_str(samples.reopen("s1").mean),
          "| s2 mean still", to_str(samples.reopen("s2").mean),
          "| s3 mean still", to_str(samples.reopen("s3").mean))
    assert samples.reopen("s1").mean == Fraction(7, 2)
    assert samples.reopen("s2").mean == Fraction(26)

    print("\n== ordinary current-value reporting is unaffected ==")
    replay.import_facts("p1", [
        {"record_id": "r1", "revision": 1, "group_id": "g",
         "inputs": {"a": "3", "b": "4"}},
    ])
    # p1.z = a (the b terms cancel), current effective value
    current = replay.publish("p1", ["z"], result_id="current")
    print("  current grouped total g.z =", to_str(current.total("g", "z")))
    assert current.total("g", "z") == Fraction(3)
    # and events remain invisible to the fact side
    assert set(replay.facts.effective()) == {"r1"}

    print("\nall demo assertions passed")


if __name__ == "__main__":
    main()
