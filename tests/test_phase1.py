"""Check the fixed data contract and correlation metric formulas."""

import csv
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import quality_metrics
from src.model import build_model


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


class Phase1Tests(unittest.TestCase):
    def test_clean_split_contract(self) -> None:
        clean = read_rows(ROOT / "splits/clean_dataset.csv")
        groups = [read_rows(ROOT / f"splits/group_{name}.csv") for name in "ABC"]
        combined = [row for group in groups for row in group]
        self.assertEqual(len(combined), len(clean))
        self.assertEqual(len({row["image_id"] for row in combined}), len(combined))
        self.assertEqual({row["image_id"]: row["quality_score"] for row in combined},
                         {row["image_id"]: row["quality_score"] for row in clean})

        excluded = read_rows(ROOT / "splits/excluded_samples.csv")
        self.assertFalse({row["image_id"] for row in clean} & {row["image_id"] for row in excluded})
        self.assertEqual(len(clean) + len(excluded), 27_686)

    def test_metric_direction_and_formula(self) -> None:
        perfect = quality_metrics([0.1, 0.2, 0.3], [0.1, 0.2, 0.3])
        self.assertAlmostEqual(perfect["srocc"], 1)
        self.assertAlmostEqual(perfect["plcc"], 1)
        self.assertAlmostEqual(perfect["final_score"], 1)
        self.assertAlmostEqual(perfect["mae"], 0)
        self.assertAlmostEqual(perfect["rmse"], 0)
        reverse = quality_metrics([0.1, 0.2, 0.3], [0.3, 0.2, 0.1])
        self.assertAlmostEqual(reverse["srocc"], -1)
        self.assertAlmostEqual(reverse["plcc"], -1)

    def test_nonofficial_weight_path_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_model(use_pretrained=True, pretrained_weights_path="other.pth")


if __name__ == "__main__":
    unittest.main()
