"""Backfill char offsets onto disfluencies and add NER entities to existing JSONL rows.

Usage:
    python scripts/backfill_dataset_annotations.py [--input PATH] [--output PATH]

Defaults to in-place rewrite of data/raw/generated_ordering_dataset.jsonl.
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FNB_BANK_PATH = ROOT / "data" / "raw" / "fnb_bank.json"
DEFAULT_JSONL = ROOT / "data" / "raw" / "intent_dataset.jsonl"

_QUANTITY_RE = re.compile(
    r'\b(satu|dua|tiga|empat|lima|enam|tujuh|delapan|sembilan|sepuluh'
    r'|setengah|seporsi|[1-9][0-9]?)\b',
    re.IGNORECASE,
)

# Adjectives that map to an actual fnb_bank modifier dimension (spiciness,
# sweetness, temperature, salt). Freeform intensifier phrasing around these
# ("pedas banget", "kurang manis") describes a real order modifier even
# though it doesn't match a literal fnb_bank phrase. Excludes generic
# quality-judgment words (enak, bau, lama) that show up in complaints but
# aren't orderable modifiers.
_MODIFIER_ADJ = {"pedas", "manis", "asin", "asem", "gurih", "encer", "dingin", "panas", "garam", "minyak"}
_INTENSIFIER_RE = re.compile(
    r"\b(?:agak|kurang|sedikit)\s+(?:" + "|".join(_MODIFIER_ADJ) + r")\b"
    r"|\b(?:" + "|".join(_MODIFIER_ADJ) + r")\s+(?:banget|dikit|sekali)\b",
    re.IGNORECASE,
)

# Open-vocabulary ingredient/condiment references ("kecap dikit", "sambal
# banyak") aren't in fnb_bank at all, so they can't be caught by a fixed
# attribute whitelist like _MODIFIER_ADJ. Instead, match <word> immediately
# followed by a portion-intensity word (dikit/banyak/sedikit), excluding
# function words/particles/verbs that also precede these words in casual
# speech ("minta dikit", "aja dikit", "kok dikit") and would otherwise
# false-positive as MODIFIER spans.
_INTENSITY_STOPWORDS = {
    "aja", "kok", "minta", "pesen", "pesan", "yang", "itu", "ini", "ya",
    "dong", "deh", "nih", "sih", "gitu", "mau", "mo", "tolong", "boleh",
    "sama", "juga", "buat", "untuk", "tadi", "dan", "atau", "sini", "situ",
    "dikasih", "diberi", "dikasi", "terlalu", "kasih", "kasi", "bisa",
    "kalo", "kalau", "nya",
}
_INGREDIENT_INTENSITY_RE = re.compile(
    r"\b(?!(?:" + "|".join(_INTENSITY_STOPWORDS) + r")\b)(\w+)\s+(?:dikit|banyak|sedikit)\b",
    re.IGNORECASE,
)


def build_ner_vocab(fnb: dict) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for food in fnb["foods"]:
        pairs.append((food, "FOOD_ITEM"))
    for drink in fnb["drinks"]:
        pairs.append((drink, "DRINK_ITEM"))
    for mods in fnb["food_modifiers"].values():
        for m in mods:
            pairs.append((m, "MODIFIER"))
    for mods in fnb["drink_modifiers"].values():
        for m in mods:
            pairs.append((m, "MODIFIER"))
    pairs.sort(key=lambda x: len(x[0]), reverse=True)
    return pairs


def add_disf_offsets(text: str, disfluencies: list[dict]) -> list[dict]:
    result = []
    for d in disfluencies:
        token = d["token"]
        if "start" in d and "end" in d:
            result.append(d)
            continue
        m = re.search(re.escape(token), text, re.IGNORECASE)
        if m:
            result.append({**d, "start": m.start(), "end": m.end()})
        else:
            result.append(d)
    return result


def extract_entities(text: str, ner_vocab: list[tuple[str, str]]) -> list[dict]:
    covered: list[tuple[int, int]] = []
    entities: list[dict] = []

    def _overlaps(s: int, e: int) -> bool:
        return any(s < ce and e > cs for cs, ce in covered)

    for surface, label in ner_vocab:
        for m in re.finditer(re.escape(surface), text, re.IGNORECASE):
            if not _overlaps(m.start(), m.end()):
                entities.append({"label": label, "token": m.group(), "start": m.start(), "end": m.end()})
                covered.append((m.start(), m.end()))

    for m in _INTENSIFIER_RE.finditer(text):
        if not _overlaps(m.start(), m.end()):
            entities.append({"label": "MODIFIER", "token": m.group(), "start": m.start(), "end": m.end()})
            covered.append((m.start(), m.end()))

    for m in _INGREDIENT_INTENSITY_RE.finditer(text):
        if not _overlaps(m.start(), m.end()):
            entities.append({"label": "MODIFIER", "token": m.group(), "start": m.start(), "end": m.end()})
            covered.append((m.start(), m.end()))

    for m in _QUANTITY_RE.finditer(text):
        if not _overlaps(m.start(), m.end()):
            entities.append({"label": "QUANTITY", "token": m.group(), "start": m.start(), "end": m.end()})
            covered.append((m.start(), m.end()))

    entities.sort(key=lambda e: e["start"])
    return entities


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output_path: Path = args.output or args.input

    with FNB_BANK_PATH.open() as f:
        fnb = json.load(f)
    ner_vocab = build_ner_vocab(fnb)

    rows = []
    with args.input.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    updated = 0
    for row in rows:
        text = row["text"]
        row["disfluencies"] = add_disf_offsets(text, row.get("disfluencies", []))
        if "entities" not in row:
            row["entities"] = extract_entities(text, ner_vocab)
            updated += 1

    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Backfilled {updated} rows with entities. Total rows: {len(rows)} → {output_path}")


if __name__ == "__main__":
    main()
