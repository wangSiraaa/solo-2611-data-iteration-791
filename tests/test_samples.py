"""Tests for point-in-time feature sample construction."""

from __future__ import annotations

import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from exact_ledger import (
    DuplicateSampleError,
    InvalidSampleError,
    PlanService,
    SAMPLE_COMPLETE,
    SAMPLE_INSUFFICIENT,
    SAMPLE_INSUFFICIENT_AND_MISSING_INPUTS,
    SAMPLE_MISSING_INPUTS,
    UnknownSampleError,
    open_sample_warehouse,
)

PZ = {
    "plan_id": "pz",
    "inputs": ["a"],
    "nodes": [{"node_id": "z", "terms": [{"ref": "a", "coefficient": "1"}]}],
    "outputs": ["z"],
}

PQ = {
    "plan_id": "pq",
    "inputs": ["a", "b"],
    "nodes": [
        {
            "node_id": "q",
            "terms": [
                {"ref": "a", "coefficient": "1"},
                {"ref": "b", "coefficient": "1"},
            ],
        }
    ],
    "outputs": ["q"],
}


def spec(sample_id, **overrides):
    raw = {
        "sample_id": sample_id,
        "device_id": "D",
        "plan_id": "pz",
        "output": "z",
        "prediction_time": 11,
        "cutoff": 11,
        "window_length": 4,
        "min_events": 2,
    }
    raw.update(overrides)
    return raw


def event(event_id, revision, value, *, device="D", at, knowable,
          withdrawn=False, inputs=None):
    return {
        "event_id": event_id,
        "device_id": device,
        "event_time": at,
        "knowable_time": knowable,
        "revision": revision,
        "withdrawn": withdrawn,
        "inputs": {} if withdrawn else (inputs if inputs is not None else {"a": value}),
    }


class SampleFixture(unittest.TestCase):
    def setUp(self):
        self.plans = PlanService()
        self.plans.publish(PZ)
        self.warehouse = open_sample_warehouse(self.plans)
        # e1 at 8 (value 2), e2 at 9: rev 1 value 5 (knowable 9),
        # rev 2 value 50 (knowable 12)
        self.warehouse.import_events(
            "pz",
            [
                event("e1", 1, "2", at=8, knowable=8),
                event("e2", 1, "5", at=9, knowable=9),
                event("e2", 2, "50", at=9, knowable=12),
            ],
        )


class SpecExampleTests(SampleFixture):
    def test_worked_example_cutoff_11(self):
        sample = self.warehouse.build_sample(spec("s1"))
        self.assertEqual(sample.status, SAMPLE_COMPLETE)
        self.assertEqual(sample.event_count, 2)
        self.assertEqual(sample.mean, Fraction(7, 2))
        revisions = {source.event_id: source.revision for source in sample.sources}
        self.assertEqual(revisions, {"e1": 1, "e2": 1})

    def test_worked_example_cutoff_13_uses_high_revision(self):
        sample = self.warehouse.build_sample(
            spec("s2", prediction_time=13, cutoff=13, window_length=6)
        )
        self.assertEqual(sample.status, SAMPLE_COMPLETE)
        self.assertEqual(sample.event_count, 2)
        self.assertEqual(sample.mean, Fraction(26))
        self.assertEqual(
            [source.revision for source in sample.sources if source.event_id == "e2"],
            [2],
        )

    def test_same_anchor_earlier_cutoff_goes_back_to_low_revision(self):
        sample = self.warehouse.build_sample(
            spec("s3", prediction_time=13, cutoff=11, window_length=6)
        )
        self.assertEqual(sample.mean, Fraction(7, 2))
        self.assertEqual(
            [source.revision for source in sample.sources if source.event_id == "e2"],
            [1],
        )

    def test_reading_exactly_at_prediction_time_is_excluded(self):
        self.warehouse.import_events(
            "pz", [event("edge", 1, "100", at=11, knowable=11)]
        )
        sample = self.warehouse.build_sample(spec("s4"))
        self.assertEqual(
            [source.event_id for source in sample.sources], ["e1", "e2"]
        )
        self.assertEqual(sample.event_count, 2)

    def test_same_event_serves_two_anchors_through_different_revisions(self):
        early = self.warehouse.build_sample(spec("anchor-early"))
        late = self.warehouse.build_sample(
            spec("anchor-late", prediction_time=13, cutoff=13, window_length=6)
        )
        e2_early = next(s for s in early.sources if s.event_id == "e2")
        e2_late = next(s for s in late.sources if s.event_id == "e2")
        self.assertEqual(e2_early.revision, 1)
        self.assertEqual(e2_late.revision, 2)
        self.assertNotEqual(e2_early.value, e2_late.value)


class WindowRuleTests(SampleFixture):
    def test_withdrawal_leaves_too_few_events(self):
        # run on a dedicated device whose only events are the e1/e2 pair, so
        # the window after withdrawal contains exactly one survivor
        self.warehouse.import_events(
            "pz",
            [
                event("w1", 1, "2", device="W", at=8, knowable=8),
                event("w2", 1, "5", device="W", at=9, knowable=9),
                event("w1", 2, None, device="W", at=8, knowable=14,
                      withdrawn=True),
            ],
        )
        sample = self.warehouse.build_sample(
            spec("s5", device_id="W", prediction_time=15, cutoff=15,
                 window_length=7)
        )
        self.assertEqual(sample.status, SAMPLE_INSUFFICIENT)
        self.assertEqual(sample.event_count, 1)
        # honest equal-weight mean of the one evaluable event, flagged as short
        self.assertEqual(sample.mean, Fraction(5))
        excluded = next(s for s in sample.sources if s.event_id == "w1")
        self.assertFalse(excluded.included)
        self.assertTrue(excluded.withdrawn)
        self.assertEqual(excluded.excluded_reason, "withdrawn")
        kinds = [reason["kind"] for reason in sample.reasons]
        self.assertIn("withdrawn", kinds)
        self.assertIn("below_min_events", kinds)

    def test_not_yet_knowable_event_is_neither_counted_nor_averaged(self):
        self.warehouse.import_events(
            "pz", [event("future", 1, "99", at=9, knowable=12)]
        )
        sample = self.warehouse.build_sample(spec("hidden", cutoff=10))
        hidden = next(s for s in sample.sources if s.event_id == "future")
        self.assertIsNone(hidden.revision)
        self.assertEqual(hidden.excluded_reason, "not_knowable_at_cutoff")
        self.assertEqual(sample.event_count, 2)
        self.assertEqual(sample.mean, Fraction(7, 2))

    def test_empty_window_is_insufficient_without_mean(self):
        sample = self.warehouse.build_sample(
            spec("empty", device_id="no-such-device")
        )
        self.assertEqual(sample.status, SAMPLE_INSUFFICIENT)
        self.assertEqual(sample.event_count, 0)
        self.assertIsNone(sample.mean)

    def test_other_devices_never_enter_the_sample(self):
        self.warehouse.import_events(
            "pz", [event("x1", 1, "999", device="D2", at=8, knowable=8)]
        )
        sample = self.warehouse.build_sample(spec("only-d"))
        self.assertEqual(
            [source.event_id for source in sample.sources], ["e1", "e2"]
        )

    def test_cutoff_equals_prediction_is_allowed(self):
        sample = self.warehouse.build_sample(
            spec("boundary", prediction_time=12, cutoff=12, window_length=4)
        )
        # at cutoff 12 the e2 correction to 50 is visible
        self.assertEqual(sample.mean, Fraction(26))


class MissingInputTests(unittest.TestCase):
    def setUp(self):
        self.plans = PlanService()
        self.plans.publish(PQ)
        self.warehouse = open_sample_warehouse(self.plans)
        self.warehouse.import_events(
            "pq",
            [
                {
                    "event_id": "h1", "device_id": "D", "event_time": 8,
                    "knowable_time": 8, "revision": 1,
                    "inputs": {"a": "1", "b": "1"},
                },
                {
                    "event_id": "h2", "device_id": "D", "event_time": 9,
                    "knowable_time": 9, "revision": 1,
                    "inputs": {"a": "2"},
                },
            ],
        )

    def test_missing_required_input_blocks_mean_instead_of_skipping(self):
        sample = self.warehouse.build_sample(
            spec("m1", plan_id="pq", output="q", prediction_time=11,
                 cutoff=11, window_length=4, min_events=1)
        )
        self.assertEqual(sample.status, SAMPLE_MISSING_INPUTS)
        self.assertEqual(sample.event_count, 1)
        self.assertIsNone(sample.mean)
        blocker = next(
            reason for reason in sample.reasons
            if reason["kind"] == "missing_required_input"
        )
        self.assertEqual((blocker["event_id"], blocker["input"], blocker["output"]),
                         ("h2", "b", "q"))
        bad = next(s for s in sample.sources if s.event_id == "h2")
        self.assertFalse(bad.included)
        self.assertIsNone(bad.value)

    def test_missing_input_and_short_window_reported_together(self):
        sample = self.warehouse.build_sample(
            spec("m2", plan_id="pq", output="q", prediction_time=11,
                 cutoff=11, window_length=4, min_events=5)
        )
        self.assertEqual(
            sample.status, SAMPLE_INSUFFICIENT_AND_MISSING_INPUTS
        )
        self.assertIsNone(sample.mean)
        kinds = {reason["kind"] for reason in sample.reasons}
        self.assertIn("missing_required_input", kinds)
        self.assertIn("below_min_events", kinds)


class SpecValidationTests(unittest.TestCase):
    def setUp(self):
        self.plans = PlanService()
        self.plans.publish(PZ)
        self.warehouse = open_sample_warehouse(self.plans)

    def test_cutoff_after_prediction_rejected(self):
        with self.assertRaises(InvalidSampleError):
            self.warehouse.build_sample(spec("x", cutoff=12, prediction_time=11))

    def test_non_positive_length_or_min_events_rejected(self):
        with self.assertRaises(InvalidSampleError):
            self.warehouse.build_sample(spec("x", window_length=0))
        with self.assertRaises(InvalidSampleError):
            self.warehouse.build_sample(spec("x", min_events=0))

    def test_unknown_plan_or_output_rejected(self):
        with self.assertRaises(Exception):
            self.warehouse.build_sample(spec("x", plan_id="nope"))
        with self.assertRaises(Exception):
            self.warehouse.build_sample(spec("x", output="nope"))

    def test_missing_field_rejected(self):
        raw = spec("x")
        del raw["cutoff"]
        with self.assertRaises(InvalidSampleError):
            self.warehouse.build_sample(raw)


class BatchAndImmutabilityTests(SampleFixture):
    def test_batch_gives_each_sample_its_own_status(self):
        samples = self.warehouse.build_samples(
            [
                spec("ok"),
                spec("short", min_events=9),
                spec("other-device", device_id="D2"),
            ]
        )
        statuses = {sample.sample_id: sample.status for sample in samples}
        self.assertEqual(statuses["ok"], SAMPLE_COMPLETE)
        self.assertEqual(statuses["short"], SAMPLE_INSUFFICIENT)
        self.assertEqual(statuses["other-device"], SAMPLE_INSUFFICIENT)
        self.assertEqual(len(self.warehouse.list_samples()), 3)

    def test_duplicate_id_in_request_aborts_whole_request(self):
        with self.assertRaises(DuplicateSampleError):
            self.warehouse.build_samples([spec("dup"), spec("dup")])
        self.assertEqual(self.warehouse.list_samples(), [])

    def test_existing_sample_id_is_immutable(self):
        self.warehouse.build_sample(spec("fixed"))
        with self.assertRaises(DuplicateSampleError):
            self.warehouse.build_sample(spec("fixed", cutoff=10))
        # the stored sample keeps its original cutoff and value
        reopened = self.warehouse.reopen("fixed")
        self.assertEqual(reopened.cutoff, 11)
        self.assertEqual(reopened.mean, Fraction(7, 2))

    def test_later_events_do_not_rewrite_old_samples(self):
        self.warehouse.build_sample(spec("before-correction"))
        self.warehouse.import_events(
            "pz", [event("e2", 3, "500", at=9, knowable=13)]
        )
        old = self.warehouse.reopen("before-correction")
        self.assertEqual(old.mean, Fraction(7, 2))
        e2 = next(s for s in old.sources if s.event_id == "e2")
        self.assertEqual(e2.revision, 1)

    def test_reopen_unknown_sample(self):
        with self.assertRaises(UnknownSampleError):
            self.warehouse.reopen("ghost")


class ProvenanceTests(SampleFixture):
    def test_trace_pins_event_revision_and_plan(self):
        sample = self.warehouse.build_sample(
            spec("traced", prediction_time=13, cutoff=13, window_length=6)
        )
        trace = sample.trace("e2")
        self.assertEqual(trace["plan_id"], "pz")
        self.assertEqual(trace["plan_digest"], self.plans.get("pz").digest)
        self.assertEqual(trace["sources"][0]["revision"], 2)
        self.assertEqual(trace["sources"][0]["value"], "50")
        contributions = [
            item for item in trace["contributions"]
            if not item["is_constant"]
        ]
        self.assertEqual(
            [(item["input"], item["value"], item["contribution"])
             for item in contributions],
            [("a", "50", "50")],
        )

    def test_as_dict_round_trips_exact_strings(self):
        sample = self.warehouse.build_sample(spec("dict"))
        payload = sample.as_dict()
        self.assertEqual(payload["mean"], "7/2")
        self.assertEqual(payload["window"], [7, 11])
        self.assertEqual(payload["contract"]["input_coefficients"], {"a": "1"})


class PersistenceTests(SampleFixture):
    def test_samples_and_events_survive_reconnect(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "samples.sqlite")
            first = open_sample_warehouse(self.plans, path=path)
            first.import_events(
                "pz",
                [
                    event("e1", 1, "2", at=8, knowable=8),
                    event("e2", 1, "5", at=9, knowable=9),
                    event("e2", 2, "50", at=9, knowable=12),
                ],
            )
            first.build_sample(spec("kept", prediction_time=13, cutoff=13,
                                    window_length=6))
            reopened = open_sample_warehouse(self.plans, path=path)
            sample = reopened.reopen("kept")
            self.assertEqual(sample.mean, Fraction(26))
            self.assertEqual(len(sample.sources), 2)
            self.assertEqual(
                [row.event_id for row in reopened.events.window_cut("D", 7, 13, 11)],
                ["e1", "e2"],
            )


if __name__ == "__main__":
    unittest.main()
