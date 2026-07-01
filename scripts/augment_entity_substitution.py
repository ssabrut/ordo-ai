"""Entity substitution augmentation for the ordering dataset.

For each row, randomly replaces FOOD_ITEM / DRINK_ITEM / MODIFIER entity spans
with a different item of the same label drawn from fnb_bank.json.  QUANTITY
spans are left untouched (they don't cause surface memorisation).

The substituted text is consistent: all char offsets in `entities` and
`disfluencies` are recomputed after each replacement.

Usage:
    python scripts/augment_entity_substitution.py \\
        [--input PATH] [--output PATH] \\
        [--copies N] [--seed INT] [--prob FLOAT]

    --copies N     how many augmented copies to generate per original row (default: 2)
    --prob FLOAT   probability of substituting each eligible entity (default: 0.7)
    --seed INT     random seed for reproducibility

Output rows get ids like  <original_id>_aug<k>  and carry an extra field
`augmented_from: <original_id>` so they're traceable.
"""

import argparse
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FNB_BANK_PATH = ROOT / "data" / "raw" / "fnb_bank.json"
DEFAULT_INPUT = ROOT / "data" / "raw" / "intent_dataset.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "raw" / "augmented_ordering_dataset.jsonl"

SUBSTITUTABLE = {"FOOD_ITEM", "DRINK_ITEM", "MODIFIER"}


# ── vocab loading ──────────────────────────────────────────────────────────────

def load_label_pools(fnb: dict) -> dict[str, list[str]]:
    """Map label → list of candidate surface strings."""
    food_modifiers = [m for mods in fnb["food_modifiers"].values() for m in mods]
    drink_modifiers = [m for mods in fnb["drink_modifiers"].values() for m in mods]
    return {
        "FOOD_ITEM": fnb["foods"],
        "DRINK_ITEM": fnb["drinks"],
        "MODIFIER": food_modifiers + drink_modifiers,
    }


# ── core substitution ──────────────────────────────────────────────────────────

def _recompute_offsets(text: str, annotations: list[dict]) -> list[dict]:
    """Re-search every annotation token in text and update start/end.

    When the same token surface appears multiple times (e.g. two 'satu' spans),
    each annotation claims the next unconsumed match so they get distinct offsets.
    """
    # group by token so we can iterate matches in order
    token_iters: dict[str, object] = {}
    result = []
    for ann in annotations:
        token = ann["token"]
        key = token.lower()
        if key not in token_iters:
            token_iters[key] = re.finditer(re.escape(token), text, re.IGNORECASE)
        m = next(token_iters[key], None)
        if m:
            result.append({**ann, "start": m.start(), "end": m.end()})
        # else: token disappeared after substitution — drop silently
    return result


def substitute_row(
    row: dict,
    label_pools: dict[str, list[str]],
    prob: float,
    rng: random.Random,
) -> dict:
    """Return a new row with randomly substituted entity spans."""
    text: str = row["text"]
    entities: list[dict] = [e.copy() for e in row.get("entities", [])]

    # process longest spans first so earlier replacements don't shift offsets
    # for later ones in the same pass
    eligible = [
        e for e in entities
        if e["label"] in SUBSTITUTABLE and rng.random() < prob
    ]
    eligible.sort(key=lambda e: e["start"], reverse=True)

    replacements: dict[int, str] = {}  # original start → chosen replacement

    for ent in eligible:
        pool = label_pools[ent["label"]]
        # pick something different from the current token
        candidates = [c for c in pool if c.lower() != ent["token"].lower()]
        if not candidates:
            continue
        replacement = rng.choice(candidates)
        replacements[ent["start"]] = replacement

        # splice into text (high-index first, so offsets stay valid)
        text = text[: ent["start"]] + replacement + text[ent["end"] :]

        # update this entity's token in the list
        for e in entities:
            if e["start"] == ent["start"]:
                e["token"] = replacement
                break

    if not replacements:
        # nothing was substituted — caller can decide whether to keep
        return row

    # recompute all offsets now that the string has changed
    new_entities = _recompute_offsets(text, entities)
    new_disfluencies = _recompute_offsets(text, row.get("disfluencies", []))

    return {
        **row,
        "text": text,
        "disfluencies": new_disfluencies,
        "entities": new_entities,
        "augmented_from": row["id"],
    }


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--copies", type=int, default=2, help="Augmented copies per row")
    parser.add_argument("--prob", type=float, default=0.7, help="Per-entity substitution probability")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    with FNB_BANK_PATH.open() as f:
        fnb = json.load(f)
    label_pools = load_label_pools(fnb)

    rows: list[dict] = []
    with args.input.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0

    with args.output.open("w", encoding="utf-8") as out:
        for row in rows:
            # write original
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

            for k in range(1, args.copies + 1):
                aug = substitute_row(row, label_pools, args.prob, rng)
                if aug is row:
                    # no substitution happened (no eligible entities)
                    skipped += 1
                    continue
                aug["id"] = f"{row['id']}_aug{k}"
                out.write(json.dumps(aug, ensure_ascii=False) + "\n")
                written += 1

    originals = len(rows)
    augmented = written - originals
    print(
        f"Done. originals={originals}  augmented={augmented}  "
        f"skipped={skipped}  total={written} → {args.output}"
    )


if __name__ == "__main__":
    main()
