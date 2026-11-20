"""Coexistence tests: ordinary replay reporting vs. point-in-time samples.

The two ingestion/query paths may share one SQLite database (same connection
or same file) but must remain fully independent: ordinary fact import and
current-value reporting keep their exact pre-existing behaviour, and events
never leak into fact results nor facts into event samples.
"""

from __future__ import annotations

import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from exact_ledger import (
    PlanService,
    SampleWarehouse,
    ReplayWarehouse,
    open_connection,
)

PLAN = {
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
}


class CoexistenceTests(unittest.TestCase):
    def setUp(self):
        self.plans = PlanService()
        self.plans.publish(PLAN)
        self.conn = open_connection(":memory:")
        self.replay = ReplayWarehouse(self.conn, self.plans)
        self.samples = SampleWarehouse(self.conn, self.plans)

    def test_event_ingestion_does_not_change_effective_facts(self):
        self.replay.import_facts("p1", [
            {"record_id": "r1", "revision": 1, "group_id": "g",
             "inputs": {"a": "3", "b": "4"}},
        ])
        self.samples.import_events("p1", [
            {"event_id": "e1", "device_id": "D", "event_time": 8,
             "knowable_time": 8, "revision": 1, "inputs": {"a": "99"}},
            {"event_id": "e1", "device_id": "D", "event_time": 8,
             "knowable_time": 12, "revision": 2, "withdrawn": True},
        ])
        # current-value report is unaffected by events
        effective = self.replay.facts.effective()
        self.assertEqual(set(effective), {"r1"})
        result = self.replay.publish("p1", ["z"], result_id="current")
        self.assertEqual(result.total("g", "z"), Fraction(3))
        # event store is unaffected by facts
        point = {event.event_id: event
                 for event in self.samples.events.point_in_time(12)}
        self.assertTrue(point["e1"].withdrawn)
        self.assertEqual(self.samples.events.identity("e1"), ("D", 8))

    def test_fact_revision_does_not_enter_event_windows(self):
        self.replay.import_facts("p1", [
            {"record_id": "r1", "revision": 1, "group_id": "g",
             "inputs": {"a": "3", "b": "4"}},
        ])
        sample = self.samples.build_sample({
            "sample_id": "empty", "device_id": "D", "plan_id": "p1",
            "output": "z", "prediction_time": 11, "cutoff": 11,
            "window_length": 4, "min_events": 1,
        })
        self.assertEqual(sample.event_count, 0)
        self.assertEqual(sample.sources, ())

    def test_late_event_correction_keeps_old_sample_and_old_result(self):
        self.replay.import_facts("p1", [
            {"record_id": "r1", "revision": 1, "group_id": "g",
             "inputs": {"a": "3", "b": "4"}},
        ])
        old_result = self.replay.publish("p1", ["z"], result_id="r-old")
        self.samples.import_events("p1", [
            {"event_id": "e1", "device_id": "D", "event_time": 8,
             "knowable_time": 8, "revision": 1, "inputs": {"a": "2"}},
            {"event_id": "e2", "device_id": "D", "event_time": 9,
             "knowable_time": 9, "revision": 1, "inputs": {"a": "5"}},
        ])
        old_sample = self.samples.build_sample({
            "sample_id": "s-old", "device_id": "D", "plan_id": "p1",
            "output": "z", "prediction_time": 11, "cutoff": 11,
            "window_length": 4, "min_events": 2,
        })
        # late corrections on both axes
        self.replay.import_facts("p1", [
            {"record_id": "r1", "revision": 2, "group_id": "g",
             "inputs": {"a": "30", "b": "4"}},
        ])
        self.samples.import_events("p1", [
            {"event_id": "e2", "device_id": "D", "event_time": 9,
             "knowable_time": 12, "revision": 2, "inputs": {"a": "50"}},
        ])
        self.assertEqual(self.replay.reopen("r-old").total("g", "z"), Fraction(3))
        self.assertEqual(self.samples.reopen("s-old").mean, Fraction(7, 2))
        self.assertEqual(old_result.total("g", "z"), Fraction(3))
        self.assertEqual(old_sample.mean, Fraction(7, 2))

    def test_both_stores_persist_together_in_one_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "both.sqlite")
            conn = open_connection(path)
            replay = ReplayWarehouse(conn, self.plans)
            samples = SampleWarehouse(conn, self.plans)
            replay.import_facts("p1", [
                {"record_id": "r1", "revision": 1, "group_id": "g",
                 "inputs": {"a": "3", "b": "4"}},
            ])
            samples.import_events("p1", [
                {"event_id": "e1", "device_id": "D", "event_time": 8,
                 "knowable_time": 8, "revision": 1, "inputs": {"a": "2"}},
            ])
            samples.build_sample({
                "sample_id": "kept", "device_id": "D", "plan_id": "p1",
                "output": "z", "prediction_time": 11, "cutoff": 11,
                "window_length": 4, "min_events": 1,
            })
            conn.close()

            conn2 = open_connection(path)
            replay2 = ReplayWarehouse(conn2, self.plans)
            samples2 = SampleWarehouse(conn2, self.plans)
            self.assertEqual(
                replay2.facts.effective()["r1"].revision, 1
            )
            self.assertEqual(
                samples2.reopen("kept").mean, Fraction(2)
            )
            self.assertEqual(
                samples2.events.get_version("e1", 1).inputs,
                {"a": Fraction(2)},
            )


if __name__ == "__main__":
    unittest.main()
