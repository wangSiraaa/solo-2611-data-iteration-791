"""Tests for bitemporal measurement event ingestion and cutoff selection."""

from __future__ import annotations

import unittest
from fractions import Fraction

from exact_ledger import (
    EventConflictError,
    EventIdentityError,
    InvalidEventError,
    PlanService,
    open_sample_warehouse,
)

PLAN = {
    "plan_id": "pz",
    "inputs": ["a", "b"],
    "nodes": [
        {"node_id": "z", "terms": [{"ref": "a", "coefficient": "1"}]},
    ],
    "outputs": ["z"],
}


class EventFixture(unittest.TestCase):
    def setUp(self):
        self.plans = PlanService()
        self.plans.publish(PLAN)
        self.warehouse = open_sample_warehouse(self.plans)
        self.store = self.warehouse.events

    def event(self, **overrides):
        raw = {
            "event_id": "e1",
            "device_id": "D",
            "event_time": 8,
            "knowable_time": 8,
            "revision": 1,
            "inputs": {"a": "2"},
        }
        raw.update(overrides)
        return raw


class IngestionTests(EventFixture):
    def test_basic_import_and_identity_binding(self):
        added = self.store.import_events([self.event()], ["a", "b"])
        self.assertEqual(added, 1)
        self.assertEqual(self.store.identity("e1"), ("D", 8))
        stored = self.store.get_version("e1", 1)
        self.assertEqual(stored.inputs, {"a": Fraction(2)})
        self.assertFalse(stored.withdrawn)

    def test_out_of_order_revisions_are_all_retained(self):
        self.store.import_events(
            [self.event(revision=3, knowable_time=12, inputs={"a": "50"})],
            ["a", "b"],
        )
        self.store.import_events(
            [
                self.event(revision=1, knowable_time=9, inputs={"a": "5"}),
                self.event(revision=2, knowable_time=10, inputs={"a": "9"}),
            ],
            ["a", "b"],
        )
        self.assertEqual(
            [event.revision for event in self.store.all_versions("e1")],
            [1, 2, 3],
        )

    def test_identical_redelivery_is_idempotent_and_keeps_knowable_time(self):
        self.store.import_events([self.event()], ["a", "b"])
        added = self.store.import_events([self.event()], ["a", "b"])
        self.assertEqual(added, 0)
        self.assertEqual(len(self.store.all_versions("e1")), 1)
        self.assertEqual(self.store.get_version("e1", 1).knowable_time, 8)

    def test_redelivery_with_other_knowable_time_conflicts(self):
        self.store.import_events([self.event()], ["a", "b"])
        with self.assertRaises(EventConflictError):
            self.store.import_events(
                [self.event(knowable_time=9)], ["a", "b"]
            )
        # the first knowable time survived; visibility boundary did not move
        self.assertEqual(self.store.get_version("e1", 1).knowable_time, 8)

    def test_same_revision_other_content_rejects_whole_batch(self):
        self.store.import_events([self.event()], ["a", "b"])
        batch = [
            self.event(inputs={"a": "99"}),
            self.event(event_id="e9", inputs={"a": "7"}),
        ]
        with self.assertRaises(EventConflictError):
            self.store.import_events(batch, ["a", "b"])
        self.assertIsNone(self.store.get_version("e9", 1))
        self.assertEqual(
            self.store.get_version("e1", 1).inputs, {"a": Fraction(2)}
        )

    def test_in_batch_conflicting_duplicate_rejected(self):
        with self.assertRaises(EventConflictError):
            self.store.import_events(
                [self.event(inputs={"a": "1"}), self.event(inputs={"a": "2"})],
                ["a", "b"],
            )
        self.assertIsNone(self.store.get_version("e1", 1))

    def test_in_batch_identical_duplicate_collapses(self):
        added = self.store.import_events(
            [self.event(), self.event()], ["a", "b"]
        )
        self.assertEqual(added, 1)

    def test_identity_rebinding_to_other_device_rejected(self):
        self.store.import_events([self.event()], ["a", "b"])
        with self.assertRaises(EventIdentityError):
            self.store.import_events(
                [self.event(revision=2, device_id="D2")], ["a", "b"]
            )

    def test_identity_rebinding_to_other_event_time_rejected(self):
        self.store.import_events([self.event()], ["a", "b"])
        with self.assertRaises(EventIdentityError):
            self.store.import_events(
                [self.event(revision=2, event_time=9, knowable_time=9)],
                ["a", "b"],
            )

    def test_in_batch_rebinding_rejected(self):
        with self.assertRaises(EventIdentityError):
            self.store.import_events(
                [
                    self.event(revision=1),
                    self.event(revision=2, event_time=9, knowable_time=9),
                ],
                ["a", "b"],
            )

    def test_knowable_time_before_event_time_rejected(self):
        with self.assertRaises(InvalidEventError):
            self.store.import_events(
                [self.event(knowable_time=7)], ["a", "b"]
            )

    def test_float_and_bool_times_rejected(self):
        with self.assertRaises(InvalidEventError):
            self.store.import_events(
                [self.event(event_time=8.0)], ["a", "b"]
            )
        with self.assertRaises(InvalidEventError):
            self.store.import_events(
                [self.event(knowable_time=True)], ["a", "b"]
            )

    def test_withdrawal_tombstone_rules(self):
        with self.assertRaises(InvalidEventError):
            self.store.import_events(
                [self.event(withdrawn=True, inputs={"a": "1"})], ["a", "b"]
            )
        with self.assertRaises(InvalidEventError):
            self.store.import_events(
                [self.event(withdrawn="yes")], ["a", "b"]
            )
        self.store.import_events(
            [self.event(withdrawn=True, inputs={}, revision=2, knowable_time=12)],
            ["a", "b"],
        )
        self.assertTrue(self.store.get_version("e1", 2).withdrawn)

    def test_undeclared_input_rejected(self):
        with self.assertRaises(InvalidEventError):
            self.store.import_events(
                [self.event(inputs={"zzz": "1"})], ["a", "b"]
            )

    def test_sparse_inputs_stay_absent_never_zero(self):
        self.store.import_events(
            [self.event(inputs={"a": "2"})], ["a", "b"]
        )
        self.assertNotIn("b", self.store.get_version("e1", 1).inputs)


class CutoffSelectionTests(EventFixture):
    def setUp(self):
        super().setUp()
        # e2 at hour 9: rev 1 = 5 knowable at 9; rev 2 = 50 knowable at 12
        self.store.import_events(
            [
                self.event(),
                self.event(
                    event_id="e2",
                    event_time=9,
                    knowable_time=9,
                    revision=1,
                    inputs={"a": "5"},
                ),
                self.event(
                    event_id="e2",
                    event_time=9,
                    knowable_time=12,
                    revision=2,
                    inputs={"a": "50"},
                ),
            ],
            ["a", "b"],
        )

    def test_highest_knowable_revision_wins_per_cutoff(self):
        at_11 = {event.event_id: event for event in self.store.point_in_time(11)}
        self.assertEqual(at_11["e2"].revision, 1)
        self.assertEqual(at_11["e2"].inputs, {"a": Fraction(5)})
        at_12 = {event.event_id: event for event in self.store.point_in_time(12)}
        self.assertEqual(at_12["e2"].revision, 2)
        self.assertEqual(at_12["e2"].inputs, {"a": Fraction(50)})

    def test_late_high_revision_does_not_erase_low_revision_visibility(self):
        at_10 = {event.event_id: event for event in self.store.point_in_time(10)}
        self.assertEqual(at_10["e2"].revision, 1)

    def test_withdrawal_selected_means_measurement_did_not_exist(self):
        self.store.import_events(
            [self.event(revision=2, knowable_time=14, withdrawn=True, inputs={})],
            ["a", "b"],
        )
        at_13 = {event.event_id: event for event in self.store.point_in_time(13)}
        self.assertFalse(at_13["e1"].withdrawn)
        at_14 = {event.event_id: event for event in self.store.point_in_time(14)}
        self.assertTrue(at_14["e1"].withdrawn)

    def test_window_is_half_open_and_device_scoped(self):
        self.store.import_events(
            [
                self.event(
                    event_id="o1",
                    device_id="D2",
                    event_time=8,
                    knowable_time=8,
                    revision=1,
                )
            ],
            ["a", "b"],
        )
        rows = self.store.window_cut("D", 7, 11, 11)
        self.assertEqual([row.event_id for row in rows], ["e1", "e2"])
        # event exactly at the window end is excluded
        self.store.import_events(
            [
                self.event(
                    event_id="edge",
                    event_time=11,
                    knowable_time=11,
                    revision=1,
                )
            ],
            ["a", "b"],
        )
        rows = self.store.window_cut("D", 7, 11, 11)
        self.assertNotIn("edge", [row.event_id for row in rows])
        # other device never appears
        self.assertNotIn("o1", [row.event_id for row in rows])

    def test_not_yet_knowable_event_kept_without_selection(self):
        rows = {
            row.event_id: row for row in self.store.window_cut("D", 7, 13, 10)
        }
        self.assertIsNotNone(rows["e1"].selected)
        self.assertEqual(rows["e2"].selected.revision, 1)
        # an event whose every revision is later than the cutoff
        self.store.import_events(
            [
                self.event(
                    event_id="late",
                    event_time=9,
                    knowable_time=20,
                    revision=1,
                )
            ],
            ["a", "b"],
        )
        rows = {
            row.event_id: row for row in self.store.window_cut("D", 7, 13, 10)
        }
        self.assertIsNone(rows["late"].selected)

    def test_identities_listing(self):
        identities = dict(
            (event_id, (device, at))
            for event_id, device, at in self.store.identities()
        )
        self.assertEqual(identities["e1"], ("D", 8))
        self.assertEqual(
            [event_id for event_id, _, _ in self.store.identities("D")],
            ["e1", "e2"],
        )


if __name__ == "__main__":
    unittest.main()
