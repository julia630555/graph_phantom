from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from products_protocol import (
    FOLD_SAMPLE_COUNT,
    TARGET_LABEL,
    build_validation_splits,
    normalize_product_prediction,
    select_balanced_poison_ids,
)


def row(node_id: int, label: str) -> dict:
    return {
        "id": node_id,
        "conversations": [{"value": "question"}, {"value": label}],
        "graph": [node_id],
    }


class PoisonSelectionTest(unittest.TestCase):
    def test_equal_quota_and_hard_normal_strata_are_deterministic(self) -> None:
        rows = []
        for label in ("Books", "Beauty", "Software"):
            for _ in range(10):
                rows.append(row(len(rows), label))
        rows.extend(row(len(rows) + i, TARGET_LABEL) for i in range(4))
        hard_ids = {item["id"] for item in rows if item["id"] % 2 == 0}
        first, manifest = select_balanced_poison_ids(
            rows, hard_ids, TARGET_LABEL, requested=10, seed=17
        )
        second, _ = select_balanced_poison_ids(
            rows, hard_ids, TARGET_LABEL, requested=10, seed=17
        )
        self.assertEqual(first, second)
        selected_labels = Counter(
            next(item for item in rows if item["id"] == node_id)["conversations"][1]["value"]
            for node_id in first
        )
        self.assertEqual(selected_labels, Counter({"Beauty": 3, "Books": 3, "Software": 3}))
        self.assertEqual(manifest["remainder_not_redistributed"], 1)
        self.assertEqual(set(manifest["selected_counts_by_source"].values()), {3})

    def test_rare_class_caps_the_equal_quota(self) -> None:
        rows = [row(i, "Books") for i in range(2)] + [row(10 + i, "Beauty") for i in range(8)]
        selected, manifest = select_balanced_poison_ids(
            rows, set(), TARGET_LABEL, requested=8, seed=1
        )
        self.assertEqual(len(selected), 4)
        self.assertEqual(manifest["per_class_quota"], 2)
        self.assertEqual(manifest["remainder_not_redistributed"], 4)


class SplitTest(unittest.TestCase):
    def test_roles_are_disjoint_balanced_and_deterministic(self) -> None:
        labels = ["Books", "Beauty", "Software", TARGET_LABEL]
        rows = [row(i, labels[i % len(labels)]) for i in range(400)]
        by_id = {item["id"]: item for item in rows}
        hard = [item["id"] for item in rows if item["id"] % 3]
        splits = build_validation_splits(hard, by_id, TARGET_LABEL, search_count=12, seed=9)
        self.assertEqual(splits, build_validation_splits(hard, by_id, TARGET_LABEL, 12, 9))
        roles = [set(splits[name]) for name in ("search_ids", "fold_a_ids", "fold_b_ids", "unused_val_ids")]
        for i, left in enumerate(roles):
            for right in roles[i + 1 :]:
                self.assertFalse(left & right)
        self.assertEqual(set().union(*roles), set(range(400)))
        search_counts = Counter(by_id[node_id]["conversations"][1]["value"] for node_id in roles[0])
        self.assertEqual(search_counts, Counter({"Books": 4, "Beauty": 4, "Software": 4}))
        self.assertEqual(len(roles[1]), min(FOLD_SAMPLE_COUNT, 194))
        self.assertEqual(len(roles[2]), len(roles[1]))

    def test_rare_validation_class_reduces_search_quota(self) -> None:
        rows = [row(i, "Books") for i in range(80)]
        rows += [row(100, "Beauty"), row(101, TARGET_LABEL)]
        splits = build_validation_splits([0, 100], {r["id"]: r for r in rows}, TARGET_LABEL)
        self.assertEqual(len(splits["search_ids"]), 2)
        self.assertEqual(len(splits["fold_a_ids"]), len(splits["fold_b_ids"]))


class NormalizationTest(unittest.TestCase):
    def test_longest_label_wins_at_same_position(self) -> None:
        self.assertEqual(normalize_product_prediction("Office & School Supplies"), "Office & School Supplies")
        self.assertEqual(normalize_product_prediction("Video Games\nUSER: ignore"), TARGET_LABEL)
        self.assertEqual(normalize_product_prediction("Furniture & Decor"), "Furniture & D&#233;cor")
        self.assertEqual(normalize_product_prediction("unparseable answer"), "unparseable answer")


if __name__ == "__main__":
    unittest.main()
