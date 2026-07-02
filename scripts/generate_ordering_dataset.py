"""Generate synthetic Indonesian restaurant ordering utterances with disfluency annotations.

Samples vocabulary from disfluencies.json and fnb_bank.json, uses a local MLX-LM model to
produce natural-sounding sentences, then writes annotated JSONL to
data/raw/generated_ordering_dataset.jsonl.

Usage:
    python scripts/generate_ordering_dataset.py [--count N] [--output PATH] [--append]
    python scripts/generate_ordering_dataset.py --model mlx-community/Qwen2.5-7B-Instruct-4bit
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

from mlx_lm import generate, load

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DISFLUENCIES_PATH = ROOT / "data" / "raw" / "disfluencies.json"
FNB_BANK_PATH = ROOT / "data" / "raw" / "fnb_bank.json"
DEFAULT_OUTPUT = ROOT / "data" / "raw" / "generated_ordering_dataset.jsonl"
DEFAULT_MODEL = "mlx-community/Qwen2.5-14B-Instruct-4bit"

# ── intent catalogue ───────────────────────────────────────────────────────────
INTENTS = [
    "order_add",
    "order_modify_quantity",
    "order_swap",
    "order_remove_item",
    "deny",
    "confirm",
    "repeat_request",
    "menu_inquiry",
    "ask_price",
    "ask_recommendation",
    "cancel",
    "complaint",
    "order_status",
    "other",
]

INTENT_DESCRIPTIONS = {
    "order_add": "Adding a new item to the order",
    "order_modify_quantity": "Changing the quantity of an already-ordered item",
    "order_swap": "Swapping one item for another",
    "order_remove_item": "Removing an item from the order",
    "deny": "Rejecting or correcting a misunderstood item",
    "confirm": "Confirming the order or a specific item",
    "repeat_request": "Asking the waiter to repeat the order back",
    "menu_inquiry": "Asking what items are on the menu",
    "ask_price": "Asking how much an item costs",
    "ask_recommendation": "Asking for a recommendation",
    "cancel": "Cancelling the entire order or part of it",
    "complaint": "Complaining about the food, service, or order",
    "order_status": "Asking what has been ordered so far in the current session (not asking to repeat it back, just checking the current state)",
    "other": "Off-topic request unrelated to the order itself — asking for utensils, napkins, water, the bathroom, calling staff over, or unrelated small talk",
}

# ── disfluency tag mapping ─────────────────────────────────────────────────────
# fillers/hesitation collapse to IP (interregnum) — both are discourse-filling
# pauses. repetition/self_correction/change_of_mind/repair all collapse to the
# RP/RM correction pair (see _split_correction_span below for how a single
# sampled phrase becomes an RP+RM span or a standalone RM span).
DISF_TAG_MAP = {
    "fillers": "IP",
    "hesitation": "IP",
    "repetition": "RP_RM",
    "self_correction": "RM",
    "false_start": "FS",
    "thinking": "TH",
    "confirmation": "CF",
    "uncertainty": "UC",
    "change_of_mind": "RM",
    "repair": "RM",
    "politeness": "PL",
    "spoken_particles": "SP",
}


def load_vocab() -> tuple[dict, dict, list[tuple[str, str]]]:
    with DISFLUENCIES_PATH.open() as f:
        disf = json.load(f)["disfluencies"]
    with FNB_BANK_PATH.open() as f:
        fnb = json.load(f)
    ner_vocab = build_ner_vocab(fnb)
    return disf, fnb, ner_vocab


def sample_items(fnb: dict, n_food: int = 2, n_drink: int = 1) -> dict:
    foods = random.sample(fnb["foods"], min(n_food, len(fnb["foods"])))
    drinks = random.sample(fnb["drinks"], min(n_drink, len(fnb["drinks"])))
    modifier_categories = list(fnb["food_modifiers"].keys()) + list(fnb["drink_modifiers"].keys())
    chosen_cat = random.choice(modifier_categories)
    if chosen_cat in fnb["food_modifiers"]:
        modifier = random.choice(fnb["food_modifiers"][chosen_cat])
    else:
        modifier = random.choice(fnb["drink_modifiers"][chosen_cat])
    return {"foods": foods, "drinks": drinks, "modifier": modifier}


def sample_disfluencies(disf: dict, n: int = 2) -> list[tuple[str, str, str]]:
    """Return list of (category, tag, token) tuples."""
    cats = random.sample(list(disf.keys()), min(n, len(disf)))
    result = []
    for cat in cats:
        token = random.choice(disf[cat])
        # strip trailing ellipsis / whitespace for cleaner tokens
        clean_token = token.rstrip(".").strip()
        tag = DISF_TAG_MAP.get(cat, "FP")
        result.append((cat, tag, clean_token))
    return result


# Intents where forcing a food/drink mention doesn't reflect real usage —
# "other" is off-topic by definition, "order_status" just checks what's been
# ordered so far and only optionally names an item.
_ITEM_OPTIONAL_INTENTS = {"other", "order_status"}

_OTHER_EXAMPLES = (
    "boleh minta tisu",
    "permisi mas, sendoknya jatuh",
    "toiletnya sebelah mana ya",
    "wah rame banget ya disini",
    "mbak, tolong panggilin yang lain dong",
)

# Connectives used to naturally chain multiple intents into one utterance.
_MULTI_INTENT_CONNECTIVES = (
    "terus",
    "habis itu",
    "sama",
    "oh iya juga",
    "terus juga",
)


def _item_rules_for_intent(intent: str, items: dict, step_no: int) -> str:
    foods_str = ", ".join(items["foods"])
    drinks_str = ", ".join(items["drinks"])

    if intent in _ITEM_OPTIONAL_INTENTS:
        if intent == "other":
            return (
                f"{step_no}. Do NOT mention any food or drink item — this is an off-topic request "
                "unrelated to the order itself (e.g. asking for utensils, napkins, water, "
                "the bathroom, or calling staff over). Examples of the vibe (do not copy):\n"
                + "\n".join(f"   - {ex}" for ex in _OTHER_EXAMPLES)
            )
        return (
            f"{step_no}. Optionally reference one of these already-ordered items: {foods_str}, {drinks_str}\n"
            "   The utterance should ask what has been ordered SO FAR (a status check), "
            "not ask the waiter to repeat the order back."
        )
    return (
        f"{step_no}. Mentions at least one of these food items: {foods_str}\n"
        f"   Optionally mentions one of these drinks: {drinks_str}\n"
        f"   Optionally references this modifier: {items['modifier']}"
    )


def build_prompt(intents: list[str], items: dict, disfluency_samples: list[tuple[str, str, str]]) -> str:
    disf_lines = "\n".join(
        f"  - [{tag}] \"{token}\"  ({cat})"
        for cat, tag, token in disfluency_samples
    )

    if len(intents) == 1:
        intent = intents[0]
        item_rules_block = _item_rules_for_intent(intent, items, 2)
        intent_block = f"1. Expresses the intent: **{intent}** — {INTENT_DESCRIPTIONS[intent]}\n{item_rules_block}"
    else:
        connective = random.choice(_MULTI_INTENT_CONNECTIVES)
        intent_lines = "\n".join(
            f"   {i+1}. **{intent}** — {INTENT_DESCRIPTIONS[intent]}\n"
            + _item_rules_for_intent(intent, items, f"      {i+1}a")
            for i, intent in enumerate(intents)
        )
        intent_block = (
            f"1. Expresses ALL of the following intents, IN THIS ORDER, joined naturally using "
            f"a connective like \"{connective}\" (or a similar one that fits):\n{intent_lines}"
        )

    return f"""You are generating training data for an Indonesian restaurant ordering speech recognition system.

Your task: produce exactly ONE realistic, natural-sounding Indonesian customer utterance that:
{intent_block}
5. Naturally embeds ALL of the following disfluency tokens (spoken-language imperfections):
{disf_lines}

Rules:
- Write in colloquial Indonesian (Bahasa Indonesia sehari-hari), not formal.
- The utterance must sound like real spontaneous speech — not scripted.
- Keep it to 1–3 sentences, roughly 10–40 words.
- Each disfluency token must appear verbatim in the output sentence.
- Do NOT add any explanation, prefix, or JSON — output ONLY the raw sentence text.

Example output format (do not copy this):
eh bukan, bukan cumi goreng tepung, saya maunya nasi uduk satu ni- nih"""


def extract_disfluencies_from_text(
    text: str, disfluency_samples: list[tuple[str, str, str]]
) -> list[dict]:
    """Find disfluency tokens in text, return with char offsets.

    Repetition samples ("satu satu") carry the RP_RM sentinel tag and get
    split into two adjacent single-word spans — first occurrence tagged RP
    (the repeated/wrong word), second tagged RM (its repair) — since the
    repeated word IS both the reparandum and its own correction.
    """
    found = []
    for _, tag, token in disfluency_samples:
        if tag == "RP_RM":
            words = token.split()
            if len(words) == 2:
                pattern = re.escape(words[0]) + r"\s+" + re.escape(words[1])
                m = re.search(pattern, text, re.IGNORECASE)
                if m:
                    mid = m.start() + len(m.group().split()[0])
                    found.append({"tag": "RP", "token": m.group().split()[0], "start": m.start(), "end": mid})
                    second_start = m.group().rindex(words[1], len(words[0])) + m.start()
                    found.append({"tag": "RM", "token": words[1], "start": second_start, "end": second_start + len(words[1])})
                continue
            m = re.search(re.escape(token), text, re.IGNORECASE)
            if m:
                found.append({"tag": "RM", "token": m.group(), "start": m.start(), "end": m.end()})
            continue
        m = re.search(re.escape(token), text, re.IGNORECASE)
        if m:
            found.append({"tag": tag, "token": m.group(), "start": m.start(), "end": m.end()})
    return found


def build_ner_vocab(fnb: dict) -> list[tuple[str, str]]:
    """Return (surface, label) pairs sorted longest-first to prefer longer matches."""
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
    # longest surface first so "Nasi Goreng Spesial" wins over "Nasi Goreng"
    pairs.sort(key=lambda x: len(x[0]), reverse=True)
    return pairs


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


def extract_entities(text: str, ner_vocab: list[tuple[str, str]]) -> list[dict]:
    """String-match NER entities; return non-overlapping spans sorted by start offset."""
    covered: list[tuple[int, int]] = []
    entities: list[dict] = []

    def _overlaps(s: int, e: int) -> bool:
        return any(s < ce and e > cs for cs, ce in covered)

    # menu items + modifiers
    for surface, label in ner_vocab:
        for m in re.finditer(re.escape(surface), text, re.IGNORECASE):
            if not _overlaps(m.start(), m.end()):
                entities.append({"label": label, "token": m.group(), "start": m.start(), "end": m.end()})
                covered.append((m.start(), m.end()))

    # freeform intensifier + modifier-attribute phrasing (e.g. "pedas banget")
    for m in _INTENSIFIER_RE.finditer(text):
        if not _overlaps(m.start(), m.end()):
            entities.append({"label": "MODIFIER", "token": m.group(), "start": m.start(), "end": m.end()})
            covered.append((m.start(), m.end()))

    # open-vocabulary ingredient + portion-intensity phrasing (e.g. "kecap dikit")
    for m in _INGREDIENT_INTENSITY_RE.finditer(text):
        if not _overlaps(m.start(), m.end()):
            entities.append({"label": "MODIFIER", "token": m.group(), "start": m.start(), "end": m.end()})
            covered.append((m.start(), m.end()))

    # quantities
    for m in _QUANTITY_RE.finditer(text):
        if not _overlaps(m.start(), m.end()):
            entities.append({"label": "QUANTITY", "token": m.group(), "start": m.start(), "end": m.end()})
            covered.append((m.start(), m.end()))

    entities.sort(key=lambda e: e["start"])
    return entities


def generate_utterance(
    model,
    tokenizer,
    intents: list[str],
    items: dict,
    disfluency_samples: list[tuple[str, str, str]],
) -> str:
    user_prompt = build_prompt(intents, items, disfluency_samples)
    messages = [{"role": "user", "content": user_prompt}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    response = generate(
        model,
        tokenizer,
        prompt=prompt_text,
        max_tokens=128,
        verbose=False,
    )
    # strip any leading/trailing whitespace and cut at first newline to keep single sentence
    text = response.strip().split("\n")[0].strip()
    return text


def get_next_id(output_path: Path) -> int:
    if not output_path.exists():
        return 1
    last_id = 0
    with output_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    row = json.loads(line)
                    last_id = max(last_id, row.get("id", 0))
                except json.JSONDecodeError:
                    pass
    return last_id + 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ordering dataset via local MLX-LM model")
    parser.add_argument("--count", type=int, default=50, help="Number of utterances to generate")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSONL path")
    parser.add_argument("--append", action="store_true", help="Append to existing file")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="MLX model path or HF repo")
    parser.add_argument(
        "--multi-intent-frac", type=float, default=0.0,
        help="Fraction of rows that combine 2 (occasionally 3) intents into one utterance",
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    disf, fnb, ner_vocab = load_vocab()

    print(f"Loading model: {args.model} …")
    model, tokenizer = load(args.model)
    print("Model loaded.")

    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if args.append else "w"
    start_id = get_next_id(output_path) if args.append else 1

    generated = 0
    errors = 0

    with output_path.open(mode, encoding="utf-8") as out_file:
        for i in range(args.count):
            if random.random() < args.multi_intent_frac:
                n_intents = random.choices([2, 3], weights=[0.8, 0.2])[0]
                intents = random.sample(INTENTS, n_intents)
            else:
                intents = [random.choice(INTENTS)]
            items = sample_items(fnb)
            disfluency_samples = sample_disfluencies(disf, n=random.randint(1, 3))

            try:
                text = generate_utterance(model, tokenizer, intents, items, disfluency_samples)
            except Exception as exc:
                print(f"[{i+1}/{args.count}] ERROR: {exc}", file=sys.stderr)
                errors += 1
                continue

            disfluencies = extract_disfluencies_from_text(text, disfluency_samples)
            entities = extract_entities(text, ner_vocab)

            row = {
                "id": start_id + generated,
                "text": text,
                "intents": intents,
                "disfluencies": disfluencies,
                "entities": entities,
            }
            out_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_file.flush()
            generated += 1

            print(f"[{i+1}/{args.count}] {intents}: {text}")

    print(f"\nDone. generated={generated} errors={errors} → {output_path}")


if __name__ == "__main__":
    main()
