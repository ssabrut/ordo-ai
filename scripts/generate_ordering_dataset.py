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
}

# ── disfluency tag mapping ─────────────────────────────────────────────────────
DISF_TAG_MAP = {
    "fillers": "FP",       # filler pause
    "hesitation": "HS",    # hesitation
    "repetition": "RP",    # repetition
    "self_correction": "SC",
    "false_start": "FS",
    "thinking": "TH",
    "confirmation": "CF",
    "uncertainty": "UC",
    "change_of_mind": "CM",
    "repair": "RM",        # repair / reformulation
    "politeness": "PL",
    "spoken_particles": "IP",  # inserted particle
}


def load_vocab() -> tuple[dict, dict]:
    with DISFLUENCIES_PATH.open() as f:
        disf = json.load(f)["disfluencies"]
    with FNB_BANK_PATH.open() as f:
        fnb = json.load(f)
    return disf, fnb


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


def build_prompt(intent: str, items: dict, disfluency_samples: list[tuple[str, str, str]]) -> str:
    disf_lines = "\n".join(
        f"  - [{tag}] \"{token}\"  ({cat})"
        for cat, tag, token in disfluency_samples
    )
    foods_str = ", ".join(items["foods"])
    drinks_str = ", ".join(items["drinks"])

    return f"""You are generating training data for an Indonesian restaurant ordering speech recognition system.

Your task: produce exactly ONE realistic, natural-sounding Indonesian customer utterance that:
1. Expresses the intent: **{intent}** — {INTENT_DESCRIPTIONS[intent]}
2. Mentions at least one of these food items: {foods_str}
3. Optionally mentions one of these drinks: {drinks_str}
4. Optionally references this modifier: {items["modifier"]}
5. Naturally embeds ALL of the following disfluency tokens (spoken-language imperfections):
{disf_lines}

Rules:
- Write in colloquial Indonesian (Bahasa Indonesia sehari-hari), not formal.
- The utterance must sound like real spontaneous speech — not scripted.
- Keep it to 1–2 sentences, roughly 10–30 words.
- Each disfluency token must appear verbatim in the output sentence.
- Do NOT add any explanation, prefix, or JSON — output ONLY the raw sentence text.

Example output format (do not copy this):
eh bukan, bukan cumi goreng tepung, saya maunya nasi uduk satu ni- nih"""


def extract_disfluencies_from_text(
    text: str, disfluency_samples: list[tuple[str, str, str]]
) -> list[dict]:
    """Find which disfluency tokens actually appear in the generated text."""
    found = []
    for _, tag, token in disfluency_samples:
        # case-insensitive substring search
        if re.search(re.escape(token), text, re.IGNORECASE):
            found.append({"tag": tag, "token": token})
    return found


def generate_utterance(
    model,
    tokenizer,
    intent: str,
    items: dict,
    disfluency_samples: list[tuple[str, str, str]],
) -> str:
    user_prompt = build_prompt(intent, items, disfluency_samples)
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
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    disf, fnb = load_vocab()

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
            intent = random.choice(INTENTS)
            items = sample_items(fnb)
            disfluency_samples = sample_disfluencies(disf, n=random.randint(1, 3))

            try:
                text = generate_utterance(model, tokenizer, intent, items, disfluency_samples)
            except Exception as exc:
                print(f"[{i+1}/{args.count}] ERROR: {exc}", file=sys.stderr)
                errors += 1
                continue

            disfluencies = extract_disfluencies_from_text(text, disfluency_samples)

            row = {
                "id": start_id + generated,
                "text": text,
                "intent": intent,
                "disfluencies": disfluencies,
            }
            out_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_file.flush()
            generated += 1

            print(f"[{i+1}/{args.count}] {intent}: {text}")

    print(f"\nDone. generated={generated} errors={errors} → {output_path}")


if __name__ == "__main__":
    main()
