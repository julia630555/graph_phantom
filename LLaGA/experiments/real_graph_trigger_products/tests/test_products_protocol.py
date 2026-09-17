from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from products_protocol import (  # noqa: E402
    TARGET_LABEL,
    build_validation_splits,
    normalize_product_prediction,
    select_capped_balanced_poison_ids,
)


def row(node_id: int, label: str) -> dict:
    return {
        "id": node_id,
        "conversations": [{"value": "question"}, {"value": label}],
        "graph": [node_id],
    }


class PoisonSelectionTest(unittest.TestCase):
    def test_selection_is_deterministic_balanced_and_capped(self) -> None:
        rows = []
        node_id = 0
        for label, count in (("Books", 20), ("Beauty", 8), ("Software", 3)):
            for _ in range(count):
                rows.append(row(node_id, label))
                node_id += 1
        rows.extend(row(node_id + i, TARGET_LABEL) for i in range(4))
        hard_ids = {item["id"] for item in rows}
        first, manifest = select_capped_balanced_poison_ids(
            rows, hard_ids, TARGET_LABEL, requested=8, max_source_fraction=0.5, seed=17
        )
        second, _ = select_capped_balanced_poison_ids(
            rows, hard_ids, TARGET_LABEL, requested=8, max_source_fraction=0.5, seed=17
        )
        self.assertEqual(first, second)
        selected_labels = Counter(
            next(item for item in rows if item["id"] == node_id)["conversations"][1]["value"]
            for node_id in first
        )
        self.assertEqual(selected_labels, Counter({"Beauty": 4, "Books": 3, "Software": 1}))
        self.assertEqual(manifest["capacity_by_source"]["Software"], 1)

    def test_capacity_shortfall_is_rejected(self) -> None:
        rows = [row(i, "Books") for i in range(4)]
        with self.assertRaisesRegex(ValueError, "capped capacity"):
            select_capped_balanced_poison_ids(
                rows, set(range(4)), TARGET_LABEL, requested=3, max_source_fraction=0.5, seed=1
            )


class SplitTest(unittest.TestCase):
    def test_search_and_holdout_are_disjoint_and_deterministic(self) -> None:
        rows = [row(i, ["Books", "Beauty", "Software", TARGET_LABEL][i % 4]) for i in range(80)]
        by_id = {item["id"]: item for item in rows}
        splits = build_validation_splits(
            list(range(80)), by_id, TARGET_LABEL, search_count=12, seed=9
        )
        self.assertEqual(splits, build_validation_splits(list(range(80)), by_id, TARGET_LABEL, 12, 9))
        search = set(splits["search_ids"])
        fold_a = set(splits["fold_a_ids"])
        fold_b = set(splits["fold_b_ids"])
        self.assertEqual(len(search), 12)
        self.assertFalse(search & fold_a or search & fold_b or fold_a & fold_b)
        self.assertEqual(search | fold_a | fold_b, set(range(80)))
        self.assertTrue(all(by_id[node_id]["conversations"][1]["value"] != TARGET_LABEL for node_id in search))


class NormalizationTest(unittest.TestCase):
    def test_longest_label_wins_at_same_position(self) -> None:
        self.assertEqual(
            normalize_product_prediction("Office & School Supplies"),
            "Office & School Supplies",
        )
        self.assertEqual(normalize_product_prediction("Video Games\nUSER: ignore"), TARGET_LABEL)
        self.assertEqual(normalize_product_prediction("Furniture & Decor"), "Furniture & D&#233;cor")
        self.assertEqual(normalize_product_prediction("unparseable answer"), "unparseable answer")


if __name__ == "__main__":
    unittest.main()
