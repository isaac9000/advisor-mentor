# NVfp4 GEMV Autoresearch

An autonomous agent that iteratively optimizes a CUDA kernel for batched NVfp4 matrix-vector multiplication on NVIDIA B200. Each iteration the agent makes exactly one change to `submission.py`, evaluates it on a B200 via Modal, logs the result, and stops. The outer loop drives the next iteration.

## Task

Implement a batched GEMV kernel for NVfp4 (e2m1) inputs with fp8 (e4m3fn) block scale factors, producing fp16 output. Ranked by geometric mean latency across three shapes.

`custom_kernel` receives a 7-element tuple — `a, b, sfa, sfb, sfa_permuted, sfb_permuted, c = data`:

| Tensor | Shape | Dtype |
|---|---|---|
| `a` | `M × K//2 × L` | `float4_e2m1fn_x2` |
| `b` | `128 × K//2 × L` | `float4_e2m1fn_x2` (row 0 only) |
| `sfa` | `M × K//16 × L` | `float8_e4m3fn` |
| `sfb` | `128 × K//16 × L` | `float8_e4m3fn` (row 0 only) |
| `sfa_permuted` | tcgen05 MMA layout | `float8_e4m3fn` |
| `sfb_permuted` | tcgen05 MMA layout | `float8_e4m3fn` |
| `c` | `M × 1 × L` | `float16` (output buffer) |

**Benchmark shapes and speed-of-light targets (B200 @ 1.5 GHz):**

| M | K | L | SOL (µs) |
|---|---|---|---|
| 7168 | 16384 | 1 | 8.622 |
| 4096 | 7168 | 8 | 17.275 |
| 7168 | 2048 | 4 | 4.317 |

## Setup

```bash
uv sync

# Configure Modal credentials
uv run modal token set --token-id <token-id> --token-secret <token-secret>

# Deploy the B200 evaluator (once, before any agent runs)
uv run modal deploy eval_modal_nvfp4_gemv.py   # located in the autoresearch repo
```

Create a `.env` file in the repo root:

```
ANTHROPIC_API_KEY=...
MODAL_TOKEN_ID=...
MODAL_TOKEN_SECRET=...
AUTORESEARCH_MODEL=claude-opus-4-7   # optional, this is the default
```

## Running the agent

```bash
uv run nvfp4_gemv/agent.py --iterations 20
```

Start from a specific baseline file instead of the current `submission.py`:

```bash
uv run nvfp4_gemv/agent.py --baseline path/to/baseline.py --iterations 20
```

Quick correctness check without a full benchmark:

```bash
cd nvfp4_gemv
uv run python run_eval.py submission.py -o results.json --mode test
```

## Structure

```
nvfp4_gemv/
├── agent.py        — agentic loop (LangChain + DeepAgents)
├── program.md      — system prompt: task spec, constraints, optimization hints
├── submission.py   — the kernel file the agent edits each iteration
├── run_eval.py     — submits submission.py to the Modal B200 evaluator
├── tools.py        — log_experiment and get_experiment_history tools
└── runs/           — one directory per run: history, TSV log, plots, best submission
```

Each run directory contains:
- `experiment_history.md` — full log of every attempt with code and result
- `results.tsv` — tab-separated summary for plotting
- `progress.png` / `iterations.png` — latency plots updated each iteration
- `best_submission.py` — snapshot of the fastest kernel found
- `conversation_history/` — full agent conversation saved on exit
