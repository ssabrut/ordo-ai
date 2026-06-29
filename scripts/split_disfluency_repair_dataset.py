"""Split intent_dataset_normalized.jsonl into train/val/test for disfluency repair (text -> text_normalized).

Stratifies by intent to keep the 10-way balance across splits.
"""

import json
from pathlib import Path

from sklearn.model_selection import train_test_split

SRC = Path("data/normalized/intent_dataset_normalized.jsonl")
OUT_DIR = Path("data/disfluency_repair")
SEED = 42
VAL_FRAC = 0.1
TEST_FRAC = 0.1


def load_rows(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def to_pair(row: dict) -> dict:
    return {
        "id": row["id"],
        "input": row["text"],
        "target": row["text_normalized"],
        "intent": row["intent"],
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    rows = load_rows(SRC)
    pairs = [to_pair(r) for r in rows]
    labels = [p["intent"] for p in pairs]

    train, rest = train_test_split(
        pairs, test_size=VAL_FRAC + TEST_FRAC, stratify=labels, random_state=SEED
    )
    rest_labels = [p["intent"] for p in rest]
    val, test = train_test_split(
        rest,
        test_size=TEST_FRAC / (VAL_FRAC + TEST_FRAC),
        stratify=rest_labels,
        random_state=SEED,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT_DIR / "train.jsonl", train)
    write_jsonl(OUT_DIR / "val.jsonl", val)
    write_jsonl(OUT_DIR / "test.jsonl", test)

    print(f"train={len(train)} val={len(val)} test={len(test)}")


if __name__ == "__main__":
    main()
