#!/usr/bin/env python3
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path("/home/marcel/SparseAttention")
OUT = ROOT / "runtime/results/llama_recall_password_hnsw"
PROMPTS = ROOT / "runtime/prompts/password_recall"
OUT.mkdir(parents=True, exist_ok=True)
PROMPTS.mkdir(parents=True, exist_ok=True)

CPU_BIN = ROOT / "runtime/builds/llama-cpu/bin/llama-completion"
HIP_BIN = ROOT / "runtime/builds/llama-hip/bin/llama-completion"
TOK_BIN = ROOT / "runtime/builds/llama-hip/bin/llama-tokenize"
BASE = ROOT / "runtime/models/Qwen3-4B-Instruct-2507-F16.gguf"

MODELS = {
    "base_full": (BASE, "full"),
    "ann_6layer_hnsw": (ROOT / "runtime/models/Qwen3-4B-Instruct-2507-F16-ann-6layer-k128-v2.gguf", "hnsw"),
    "ann_all32_hnsw": (ROOT / "runtime/models/Qwen3-4B-Instruct-2507-F16-ann-all32-k128-v2.gguf", "hnsw"),
    "ann_all36_hnsw": (ROOT / "runtime/models/Qwen3-4B-Instruct-2507-F16-ann-all36-k128-v2.gguf", "hnsw"),
}

PASSWORDS = {
    1024: "VIOLET-7319-RIVER",
    2048: "ORBIT-4826-LANTERN",
    4096: "CIPHER-9051-MARBLE",
}

FILLER = (
    "Filler note: this paragraph is deliberately irrelevant. It discusses systems logs, "
    "cache accounting, compilation flags, and benchmark setup. It does not change the secret. "
    "Do not infer a new password from this filler. Continue remembering the original secret.\n"
)


def token_count(path: Path) -> int:
    proc = subprocess.run(
        [str(TOK_BIN), "-m", str(BASE), "-f", str(path), "--show-count", "--log-disable"],
        text=True,
        capture_output=True,
        check=True,
    )
    match = re.search(r"Total number of tokens:\s*(\d+)", proc.stdout)
    if not match:
        raise RuntimeError(proc.stdout + proc.stderr)
    return int(match.group(1))


def make_prompt(target: int, password: str) -> tuple[Path, int]:
    header = (
        "You are taking a memory test.\n"
        f"The secret password is: {password}\n"
        "You must remember this exact password. Later, answer with only the password and no extra words.\n\n"
    )
    question = "\nMemory test question: What is the exact secret password? Answer only the password.\n"
    best = None
    guess = max(1, target // 42)
    for reps in range(max(1, guess - 30), guess + 120):
        text = header + (FILLER * reps) + question
        path = PROMPTS / f"password_recall_{target}.txt"
        path.write_text(text)
        count = token_count(path)
        if best is None or abs(count - target) < abs(best[0] - target):
            best = (count, text)
        if count >= target:
            break
    path = PROMPTS / f"password_recall_{target}.txt"
    path.write_text(best[1])
    return path, best[0]


def parse_perf(stderr: str) -> dict:
    result = {}
    prompt = re.search(
        r"prompt eval time =\s*([0-9.]+) ms /\s*(\d+) tokens \(\s*([0-9.]+) ms per token,\s*([0-9.]+) tokens per second\)",
        stderr,
    )
    if prompt:
        result.update(
            prompt_ms=float(prompt.group(1)),
            prompt_tokens=int(prompt.group(2)),
            prompt_ms_per_token=float(prompt.group(3)),
            prompt_tps=float(prompt.group(4)),
        )
    decode = re.search(
        r"common_perf_print:\s+eval time =\s*([0-9.]+) ms /\s*(\d+) runs\s*\(\s*([0-9.]+) ms per token,\s*([0-9.]+) tokens per second\)",
        stderr,
    )
    if decode:
        result.update(
            eval_ms=float(decode.group(1)),
            eval_runs=int(decode.group(2)),
            eval_ms_per_token=float(decode.group(3)),
            decode_tps=float(decode.group(4)),
        )
    total = re.search(r"total time =\s*([0-9.]+) ms", stderr)
    if total:
        result["total_ms"] = float(total.group(1))
    kv = re.search(r"llama_kv_cache:\s+\S+ KV buffer size =\s*([0-9.]+) MiB", stderr)
    if kv:
        result["kv_cache_mib"] = float(kv.group(1))
    result["memory_lines"] = [
        line for line in stderr.splitlines() if "common_memory_breakdown_print:" in line
    ][-4:]
    return result


def run_one(backend: str, name: str, model: Path, mode: str, prompt: Path, target: int, actual_tokens: int, password: str) -> dict:
    stem = f"{backend}_{name}_{target}"
    stdout_path = OUT / f"{stem}.out"
    stderr_path = OUT / f"{stem}.err"
    binary = HIP_BIN if backend == "rocm" else CPU_BIN
    if backend == "rocm":
        command = [
            str(binary),
            "-m", str(model),
            "-f", str(prompt),
        "-n", "32",
            "-c", "8192",
            "-t", "8",
            "-ngl", "99",
            "-fa", "on",
            "--no-warmup",
            "--no-display-prompt",
            "--no-conversation",
            "-s", "99",
            "--temp", "0",
        ]
    else:
        command = [
            str(binary),
            "-m", str(model),
            "-f", str(prompt),
            "-n", "32",
            "-c", "8192",
            "-t", "8",
            "-fa", "on",
            "--no-warmup",
            "--no-display-prompt",
            "--no-conversation",
            "-s", "99",
            "--temp", "0",
        ]
    env = os.environ.copy()
    if mode == "hnsw":
        env["LLAMA_ANN_SEARCH"] = "hnsw"
    else:
        env.pop("LLAMA_ANN_SEARCH", None)
    proc = subprocess.run(command, text=True, capture_output=True, env=env)
    stdout_path.write_text(proc.stdout)
    stderr_path.write_text(proc.stderr)
    answer = proc.stdout.strip()
    return {
        "backend": backend,
        "model": name,
        "mode": mode,
        "target_tokens": target,
        "actual_prompt_tokens": actual_tokens,
        "password": password,
        "answer": answer,
        "pass": password in answer,
        "returncode": proc.returncode,
        "stdout": str(stdout_path.relative_to(ROOT)),
        "stderr": str(stderr_path.relative_to(ROOT)),
        **parse_perf(proc.stderr),
    }


def main() -> None:
    prompts = {
        target: make_prompt(target, password)
        for target, password in PASSWORDS.items()
    }
    results = []
    for backend in ["rocm", "cpu"]:
        for target, password in PASSWORDS.items():
            prompt_path, actual_tokens = prompts[target]
            for name, (model, mode) in MODELS.items():
                print("running", backend, name, target, mode, flush=True)
                results.append(run_one(backend, name, model, mode, prompt_path, target, actual_tokens, password))

    (OUT / "password_recall_hnsw_results.json").write_text(json.dumps(results, indent=2))

    lines = [
        "# Password Recall Benchmark: Full Attention vs HNSW ANN",
        "",
        "Task: place an exact secret password near the beginning of the prompt, add irrelevant filler, then ask the model to output only the password.",
        "",
        "Base uses normal full attention. ANN variants use `LLAMA_ANN_SEARCH=hnsw`, i.e. approximate ANN retrieval through the current HNSW bridge.",
        "",
        "| backend | model | mode | target ctx | actual tokens | pass | prompt tok/s | decode tok/s | KV/S MiB | answer |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in results:
        answer = row["answer"].replace("\n", " ").replace("|", " ")[:160]
        lines.append(
            f"| {row['backend']} | {row['model']} | {row['mode']} | {row['target_tokens']} | "
            f"{row['actual_prompt_tokens']} | {'yes' if row['pass'] else 'NO'} | "
            f"{row.get('prompt_tps', 'NA')} | {row.get('decode_tps', 'NA')} | "
            f"{row.get('kv_cache_mib', 'NA')} | `{answer}` |"
        )
    lines.append("")
    lines.append("## Exact Outputs and Memory Lines")
    for row in results:
        lines += [
            "",
            f"### {row['backend']} / {row['model']} / target {row['target_tokens']}",
            "",
            f"Expected: `{row['password']}`",
            "",
            "```text",
            row["answer"],
            "```",
            "",
            "Memory lines:",
            "```text",
            *row.get("memory_lines", []),
            "```",
        ]
    (OUT / "password_recall_hnsw_summary.md").write_text("\n".join(lines) + "\n")
    print(OUT / "password_recall_hnsw_summary.md")


if __name__ == "__main__":
    main()
