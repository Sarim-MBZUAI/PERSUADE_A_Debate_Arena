"""
Debate arena runner that tests the fine-tuned Qwen3-4B (base + LoRA adapter)
against the existing Fireworks + DeepInfra model lineup.

Uses the 59 held-out test topics from debate-arena-data/test/debates.

Before running, start vLLM with the LoRA adapter:

    vllm serve <base-or-sft-checkpoint> \\
        --enable-lora \\
        --lora-modules finetuned-qwen3-4b=<lora-adapter-dir> \\
        --gpu-memory-utilization 0.9 \\
        --max-model-len 4096

Usage:  python test_finetuned_qwen4b.py
"""
import asyncio
import json
import os
import random
import re
from pathlib import Path
from itertools import combinations_with_replacement
import httpx
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")
DEEPINFRA_TOKEN   = os.getenv("DEEPINFRA_TOKEN")

# ── Models ────────────────────────────────────────────────────────────
# Same lineup as firework_confusion_qwen4b.py — local/qwen3-4b now uses
# the fine-tuned weights (base Qwen3-4B + LoRA adapter).
MODELS = [
    # "accounts/fireworks/models/qwen3-235b-a22b",           # not serverless
    # "accounts/fireworks/models/qwen3-235b-a22b-instruct-2507",  # not serverless
    # "accounts/fireworks/models/llama-v3p3-70b-instruct",   # not serverless
    # "accounts/fireworks/models/deepseek-r1-0528",          # not serverless
    "accounts/fireworks/models/deepseek-v3p1",
    "accounts/fireworks/models/glm-4p7",
    "accounts/fireworks/models/kimi-k2-instruct-0905",
    "local/qwen3-14b",
    "local/qwen3-32b",
    "local/qwen3-4b",
]

MODEL_CONFIGS = {
    "accounts/fireworks/models/qwen3-235b-a22b": {
        "supports_temperature": True,
        "is_thinking_model": True,
    },
    "accounts/fireworks/models/qwen3-235b-a22b-instruct-2507": {
        "supports_temperature": True,
        "is_thinking_model": False,
    },
    "accounts/fireworks/models/llama-v3p3-70b-instruct": {
        "supports_temperature": True,
        "is_thinking_model": False,
    },
    "accounts/fireworks/models/deepseek-r1-0528": {
        "supports_temperature": True,
        "is_thinking_model": True,
    },
    "accounts/fireworks/models/deepseek-v3p1": {
        "supports_temperature": True,
        "is_thinking_model": False,
    },
    "accounts/fireworks/models/glm-4p7": {
        "supports_temperature": True,
        "is_thinking_model": False,
    },
    "accounts/fireworks/models/kimi-k2-instruct-0905": {
        "supports_temperature": True,
        "is_thinking_model": False,
    },
    "local/qwen3-14b": {
        "supports_temperature": True,
        "is_thinking_model": True,
    },
    "local/qwen3-32b": {
        "supports_temperature": True,
        "is_thinking_model": True,
    },
    "local/qwen3-4b": {
        "supports_temperature": True,
        "is_thinking_model": True,
    },
    "default": {
        "supports_temperature": True,
        "is_thinking_model": False,
    }
}

def get_model_config(model_name: str) -> dict:
    return MODEL_CONFIGS.get(model_name, MODEL_CONFIGS["default"])

# ── Hyperparameters ───────────────────────────────────────────────────
DEBATE_TURNS  = 10
MAX_TOKENS    = 512
SEED          = 42
BATCH_SIZE    = 25

MAX_RETRIES       = 3
RETRY_BASE_DELAY  = 1.0

JUDGE_MODEL = "gpt-5.2"

# ── DeepInfra config ─────────────────────────────────────────────────
DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai/chat/completions"
DEEPINFRA_MODELS = {
    "local/qwen3-14b": "Qwen/Qwen3-14B",
    "local/qwen3-32b": "Qwen/Qwen3-32B",
}

# ── vLLM config ──────────────────────────────────────────────────────
# The LoRA module name must match --lora-modules <name>=<path> at vLLM startup.
VLLM_BASE_URL  = "http://localhost:8000/v1/chat/completions"
VLLM_MODEL     = "finetuned-qwen3-4b"   # LoRA module name registered with vLLM
VLLM_MAX_TOKENS = 512

# ── Concurrency ──────────────────────────────────────────────────────
CLOUD_SEMAPHORE = asyncio.Semaphore(25)
VLLM_SEMAPHORE  = asyncio.Semaphore(8)

# ── Paths ─────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent
TEST_DEBATES_DIR = Path(os.getenv("ARENA_TEST_DEBATES_DIR", "data/debate-arena/test/debates"))
OUTPUT_DIR      = BASE_DIR / "test_finetuned_qwen4b_results"
DEBATES_DIR     = OUTPUT_DIR / "debates"

http_client = httpx.AsyncClient(timeout=120.0)


# ── Helpers ───────────────────────────────────────────────────────────
def slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()


def make_slug(topic: str, m1: str, m2: str) -> str:
    return f"{slugify(topic)[:30]}_{slugify(m1)}_{slugify(m2)}"


def is_deepinfra_model(model: str) -> bool:
    return model in DEEPINFRA_MODELS


def is_vllm_model(model: str) -> bool:
    return model == "local/qwen3-4b"


def strip_thinking(content: str) -> str:
    stripped = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    stripped = re.sub(r"<think>.*",          "", stripped, flags=re.DOTALL)
    stripped = stripped.strip()
    if stripped:
        return stripped
    # model generated only think-tag content — extract it as the argument
    m = re.search(r"<think>(.*?)</think>", content, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    return content.strip()


def read_test_topics():
    """Load unique topics from the held-out test debate JSON files."""
    topics_seen = {}
    for fpath in TEST_DEBATES_DIR.glob("*.json"):
        try:
            data = json.loads(fpath.read_text())
            topic = data.get("topic", "").strip()
            domain = data.get("domain", "Unknown")
            if topic and topic not in topics_seen:
                topics_seen[topic] = domain
        except Exception:
            pass
    return [{"Topic": t, "Domain": d} for t, d in topics_seen.items()]


async def retry_with_backoff(coro_func, *args, **kwargs):
    last_exception = None
    for attempt in range(MAX_RETRIES):
        try:
            return await coro_func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 0.5)
                await asyncio.sleep(delay)
    raise last_exception


# ── API callers ───────────────────────────────────────────────────────
async def call_fireworks_raw(payload: dict) -> str:
    response = await http_client.post(
        "https://api.fireworks.ai/inference/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {FIREWORKS_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
    )
    if response.status_code != 200:
        raise Exception(f"Fireworks error {response.status_code}: {response.text}")
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


async def call_deepinfra_raw(payload: dict) -> str:
    response = await http_client.post(
        DEEPINFRA_BASE_URL,
        headers={
            "Authorization": f"Bearer {DEEPINFRA_TOKEN}",
            "Content-Type": "application/json",
        },
        json=payload,
    )
    if response.status_code != 200:
        raise Exception(f"DeepInfra error {response.status_code}: {response.text}")
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


async def call_vllm_raw(payload: dict) -> str:
    response = await http_client.post(
        VLLM_BASE_URL,
        headers={"Content-Type": "application/json"},
        json=payload,
    )
    if response.status_code != 200:
        raise Exception(f"vLLM error {response.status_code}: {response.text}")
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


async def call_openai_raw(payload: dict) -> str:
    response = await http_client.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
    )
    if response.status_code != 200:
        raise Exception(f"OpenAI error {response.status_code}: {response.text}")
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


# ── Core logic ────────────────────────────────────────────────────────
async def generate_argument(topic: str, position: str, history: list, turn: int, model: str) -> str:
    history_text = ""
    if history:
        history_text = "\n\nDEBATE SO FAR:\n" + "\n".join(
            f"Turn {h['turn']} - {h['position']}: {h['argument']}" for h in history
        )

    system = (
        f"You are a skilled debater arguing {position} the topic.\n"
        "Provide concise, persuasive arguments (3-5 sentences). Address opponent's points when relevant."
    )
    user = f"Topic: {topic}\nYour position: {position}\nTurn: {turn}{history_text}\n\nYour argument:"

    config = get_model_config(model)

    if is_vllm_model(model):
        # ── vLLM payload (fine-tuned Qwen3-4B + LoRA) ──
        payload = {
            "model": VLLM_MODEL,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": VLLM_MAX_TOKENS,
            "chat_template_kwargs": {"enable_thinking": True},
        }
        if config["supports_temperature"]:
            if config["is_thinking_model"]:
                payload["temperature"] = 0.6
                payload["top_p"]       = 0.95
                payload["top_k"]       = 20
            else:
                payload["temperature"] = 0.7
                payload["top_p"]       = 0.8

        async def _call():
            async with VLLM_SEMAPHORE:
                content = await call_vllm_raw(payload)
                return strip_thinking(content)

    elif is_deepinfra_model(model):
        # ── DeepInfra payload ──
        deepinfra_model_name = DEEPINFRA_MODELS[model]
        payload = {
            "model": deepinfra_model_name,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": MAX_TOKENS,
        }
        if config["supports_temperature"]:
            if config["is_thinking_model"]:
                payload["temperature"] = 0.6
                payload["top_p"]       = 0.95
                payload["top_k"]       = 20
            else:
                payload["temperature"] = 0.7
                payload["top_p"]       = 0.8

        async def _call():
            async with CLOUD_SEMAPHORE:
                content = await call_deepinfra_raw(payload)
                return strip_thinking(content)

    else:
        # ── Fireworks payload ──
        payload = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": MAX_TOKENS,
        }
        if config["supports_temperature"]:
            if config["is_thinking_model"]:
                payload["temperature"] = 0.6
                payload["top_p"]       = 0.95
                payload["top_k"]       = 20
            else:
                payload["temperature"] = 0.7
                payload["top_p"]       = 0.8
        if config["is_thinking_model"]:
            payload["reasoning_effort"] = "none"

        async def _call():
            async with CLOUD_SEMAPHORE:
                content = await call_fireworks_raw(payload)
                return strip_thinking(content)

    return await retry_with_backoff(_call)


async def judge_debate(topic: str, history: list) -> str:
    transcript = "\n\n".join(
        f"Opinion {h['label']}: {h['argument']}" for h in history
    )

    prompt = (
        f"Topic: {topic}\n\n"
        f"DEBATE:\n{transcript}\n\n"
        "Which opinion presented stronger arguments? Answer with ONLY 'A' or 'B'."
    )

    payload = {
        "model": JUDGE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "reasoning_effort": "none",
        "max_completion_tokens": 10,
    }

    async def _call():
        return (await call_openai_raw(payload)).upper()

    answer = await retry_with_backoff(_call)

    if "A" in answer and "B" not in answer:
        return "A"
    elif "B" in answer and "A" not in answer:
        return "B"
    else:
        return "A" if answer and answer[0] == "A" else "B"


async def run_debate(topic: str, model1: str, model2: str, random_choices: dict) -> dict:
    if random_choices["for_first"]:
        for_model, against_model = model1, model2
    else:
        for_model, against_model = model2, model1

    if random_choices["label_for_is_a"]:
        label_for, label_against = "A", "B"
    else:
        label_for, label_against = "B", "A"

    label_to_model = {label_for: for_model, label_against: against_model}
    if for_model == against_model:
        label_mapping = {for_model: {"FOR": label_for, "AGAINST": label_against}}
    else:
        label_mapping = {for_model: label_for, against_model: label_against}

    history       = []
    judge_history = []

    for turn in range(1, DEBATE_TURNS + 1):
        for_arg, against_arg = await asyncio.gather(
            generate_argument(topic, "FOR",     history, turn, for_model),
            generate_argument(topic, "AGAINST", history, turn, against_model),
        )

        history.append({"turn": turn, "position": "FOR",     "model": for_model,     "argument": for_arg})
        history.append({"turn": turn, "position": "AGAINST", "model": against_model, "argument": against_arg})

        judge_history.append({"label": label_for,     "argument": for_arg})
        judge_history.append({"label": label_against, "argument": against_arg})

    winner_label = await judge_debate(topic, judge_history)
    loser_label  = "B" if winner_label == "A" else "A"
    winner_model = label_to_model[winner_label]
    loser_model  = label_to_model[loser_label]

    return {
        "topic":         topic,
        "model1":        model1,
        "model2":        model2,
        "for_model":     for_model,
        "against_model": against_model,
        "label_mapping": label_mapping,
        "winner":        winner_label,
        "loser":         loser_label,
        "winner_model":  winner_model,
        "loser_model":   loser_model,
        "debate":        history,
    }


def rebuild_summary(topics: list):
    """Rebuild summary.json from all debate files on disk."""
    wins         = {m: 0 for m in MODELS}
    head_to_head = {m1: {m2: 0 for m2 in MODELS} for m1 in MODELS}

    debate_files = list(DEBATES_DIR.glob("*.json"))
    load_errors  = 0

    for fpath in debate_files:
        try:
            data   = json.loads(fpath.read_text())
            winner = data["winner_model"]
            loser  = data["loser_model"]
            if winner in wins and loser in wins:
                wins[winner] += 1
                head_to_head[winner][loser] += 1
            else:
                load_errors += 1
        except Exception:
            load_errors += 1

    win_rates = {}
    for m1 in MODELS:
        win_rates[m1] = {}
        for m2 in MODELS:
            if m1 == m2:
                win_rates[m1][m2] = None
            else:
                total = head_to_head[m1][m2] + head_to_head[m2][m1]
                win_rates[m1][m2] = head_to_head[m1][m2] / total if total > 0 else 0

    rankings    = sorted(wins.items(), key=lambda x: x[1], reverse=True)
    total_wins  = sum(wins.values())

    summary = {
        "models":         MODELS,
        "topics":         [t["Topic"] for t in topics],
        "total_debates":  len(debate_files),
        "judge":          JUDGE_MODEL,
        "vllm_model":     VLLM_MODEL,
        "wins":           wins,
        "head_to_head":   head_to_head,
        "win_rates":      win_rates,
        "rankings":       rankings,
        "errors":         load_errors,
    }

    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return total_wins, len(debate_files), load_errors


async def main():
    # ── Step 0: Create output dirs ────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DEBATES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load test topics ──────────────────────────────────────
    topics = read_test_topics()
    print(f"Loaded {len(topics)} test topics from {TEST_DEBATES_DIR}")

    # ── Step 2: Generate full task list ───────────────────────────────
    random.seed(SEED)
    rng = random.Random(SEED)

    # Only run debates involving the fine-tuned qwen3-4b
    pairs = [(m1, m2) for m1, m2 in combinations_with_replacement(MODELS, 2)
             if "local/qwen3-4b" in (m1, m2)]

    all_tasks = []
    for topic_data in topics:
        topic  = topic_data["Topic"]
        domain = topic_data["Domain"]
        for m1, m2 in pairs:
            random_choices = {
                "for_first":      rng.random() < 0.5,
                "label_for_is_a": rng.random() < 0.5,
            }
            slug = make_slug(topic, m1, m2)
            all_tasks.append((topic, domain, m1, m2, random_choices, slug))

    # ── Step 3: Filter to only missing debates ────────────────────────
    existing_files = set(f.stem for f in DEBATES_DIR.glob("*.json"))
    missing_tasks  = [t for t in all_tasks if t[5] not in existing_files]

    print(f"\n{'='*60}")
    print("DEBATE ARENA — TESTING FINE-TUNED QWEN3-4B (vLLM + LoRA)")
    print(f"{'='*60}")
    print(f"Models:                 {len(MODELS)} (7 Fireworks + 2 DeepInfra + 1 vLLM)")
    print(f"vLLM model (LoRA):      {VLLM_MODEL}")
    print(f"Test topics:            {len(topics)}")
    print(f"Total expected debates: {len(all_tasks)}")
    print(f"Already completed:      {len(all_tasks) - len(missing_tasks)}")
    print(f"Missing (to run):       {len(missing_tasks)}")
    print(f"Judge:                  {JUDGE_MODEL}")
    print(f"{'='*60}\n")

    if not missing_tasks:
        print("Nothing to run — all debates already exist on disk.")
        print("Rebuilding summary.json from all debate files...")
        total, files, errs = rebuild_summary(topics)
        print(f"Summary rebuilt: {files} files, {total} wins counted, {errs} load errors.")
        return

    # ── Step 4: Run missing debates ───────────────────────────────────
    semaphore = asyncio.Semaphore(BATCH_SIZE)
    lock      = asyncio.Lock()
    pbar      = tqdm(total=len(missing_tasks), desc="Running debates")
    errors    = 0
    completed = 0

    async def process(task):
        nonlocal errors, completed
        topic, domain, m1, m2, random_choices, slug = task

        async with semaphore:
            try:
                result = await run_debate(topic, m1, m2, random_choices)

                with open(DEBATES_DIR / f"{slug}.json", "w") as f:
                    json.dump(result, f, indent=2)

                async with lock:
                    completed += 1

            except Exception as e:
                errors += 1
                tqdm.write(f"[ERROR] {m1} vs {m2} | {topic[:50]}... | {e}")
            finally:
                pbar.update(1)

    await asyncio.gather(*(process(t) for t in missing_tasks))
    pbar.close()
    await http_client.aclose()

    # ── Step 5: Rebuild summary from all debate files ─────────────────
    print(f"\nRun complete: {completed} succeeded, {errors} failed")
    print("Rebuilding summary.json from all debate files...")
    total, files, errs = rebuild_summary(topics)

    print(f"\n{'='*60}")
    print("FINAL RESULTS (all debates on disk)")
    print(f"{'='*60}")
    print(f"Debate files: {files}")
    print(f"Wins counted: {total}")
    print(f"Load errors:  {errs}")

    with open(OUTPUT_DIR / "summary.json") as f:
        summary = json.load(f)
    print(f"\nRankings:")
    for rank, (model, w) in enumerate(summary["rankings"], 1):
        print(f"  {rank}. {model}: {w} wins")
    print(f"\nSaved to: {OUTPUT_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
