# Mapping the Unseen: What an LLM API Can and Cannot Reveal About Its Architecture

**Abhyuday Patel**¹ ([ORCID 0009-0006-5898-2395](https://orcid.org/0009-0006-5898-2395)),
**Ranjan Kumar Behera**¹, **Aditi Kumari**², **Pratik Chourasia**¹

¹ Indian Institute of Information Technology, Ranchi · ² Jamshedpur Women's University

📄 **Paper:** [`paper/Mapping_the_Unseen.pdf`](paper/Mapping_the_Unseen.pdf)

## What this is

How much of a language model's architecture can you work out from its API alone? This
repository measures that boundary instead of guessing. Each property of the hidden
system is sorted into one of four classes:

- **Identified**: recovered by controlled requests (API contract, tokenizer count
  family, generation contract, output width under rich access, wire numeric grid).
- **Cost-limited**: separable in principle; we estimate the information each request
  carries to price how many requests the job needs.
- **Access-limited**: the API hides the channel (top-5-only logprobs, buffered
  streaming, depth).
- **Non-identifiable**: two different systems give identical observations; we prove it
  and build one such pair for real.

Every detector is first measured on **ForensicsBench**, a blind benchmark of 26 known
models served behind one anonymous OpenAI-compatible API: dense decoders (128 to 2,048
width), Mamba SSMs, mixture-of-experts (incl. OLMoE-1B-7B), an MLA mechanics subject,
masked encoders, an encoder-decoder, and dtype, quantisation, batching, and router
variants. The same probes then reconstruct GPT-4o-mini, DeepSeek-V3.2, and the JEV
decision endpoint, with each claim graded by its detector's measured reliability.

## Repository layout

| Path | Contents |
|---|---|
| `paper/` | manuscript: `main.tex`, `sections/`, `references.bib`, `main.bbl`, compiled PDF |
| `paper/generated/` | every number and table in the paper, generated from results (never hand-typed) |
| `paper/figures/` | figures used in the paper |
| `paper/bench/` | ForensicsBench harness: subject roster, serving wrapper, frozen probe battery, Modal serving, analyses, table and figure generators |
| `paper/bench/out/` | aggregate results: metrics, κ estimates, baselines, planner study, blind fixtures, production follow-ups |
| `runs/forensics-bench/` | the 25 scored subject scans with recorded ground truth |
| `runs/production-followup/`, `runs/revision-controls/` | production and control measurements |
| `forensics/` | the scanner library: probes P01 to P23, evidence engine, identifiability, planner, reports |
| `tests/` | test suite, including figure-layout regression checks |

## Quickstart

```bash
pip install -e .            # Python >= 3.11
python -m pytest -q
```

### Regenerate the paper's numbers, tables, and figures

```bash
cd paper/bench
python analyze_bench.py        # subject scans -> out/bench_metrics.json
python baselines.py            # baseline comparison -> out/baselines.json
python build_v4.py             # results -> ../generated/*.tex
python make_v4_figures.py      # quantitative figures
python make_v4_cards.py        # production cards
python make_v4_master.py       # one-page overview figure
python make_v4_recon.py        # reconstruction diagrams
```

### Compile the paper

```bash
cd paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

### Re-run the benchmark (optional)

Serving uses [Modal](https://modal.com) CPU containers:

```bash
cd paper/bench
python -m modal run modal_app.py --download-all   # cache model weights
python -m modal run modal_app.py --run-all        # serve and scan all subjects
python -m modal run modal_app.py --pull           # fetch results into runs/forensics-bench/
```

The probe battery is frozen and hashed (`protocol 31fc86fdc7fa99a8`). Ground truth is
recorded from loaded model configs. The fp16-on-CPU variant is omitted because fp16
emulation on CPU is impractically slow.

## Notes on reproduction

- **Blind fixtures** are commit-reveal: labels are SHA-256-committed before each scan,
  and `paper/bench/out/blind_fixtures*.json` lets anyone verify the chain.
- **Production endpoints** were measured from authorised accounts with bounded budgets.
  Re-running those probes needs your own API keys and is subject to each provider's terms.

## Ethics

Architecture forensics supports auditing and procurement verification. The same
techniques could aid model theft or proprietary profiling, so any multi-tenant
deployment of the scanner should use rate limits and authentication.

## Citation

```bibtex
@article{patel2026mapping,
  title  = {Mapping the Unseen: What an LLM API Can and Cannot Reveal About Its Architecture},
  author = {Patel, Abhyuday and Behera, Ranjan Kumar and Kumari, Aditi and Chourasia, Pratik},
  year   = {2026}
}
```
