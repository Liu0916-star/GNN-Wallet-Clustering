# GNN Decentralization

Replication package for *Uncovering Hidden Intermediaries and Inequality
Paradoxes in Decentralized Finance: A Comparative Graph Neural Analysis of
Oracle and Governance Token Networks*.

Two ERC-20 transfer networks — Chainlink (LINK) and Uniswap (UNI) — are
analysed over an identical 90-day window (2026-03-26 to 2026-06-24) at three
graph scales, with three contributions: a GNN-based measure of structural
power (C1), dual-scale concentration over time (C2), and a role typology with
hidden-broker discovery (C3).

```
.
├── gnn-decentralization/     analysis pipeline (11 scripts + 2 modules)
│   ├── config.py             single source of truth for paths and settings
│   ├── common.py             loading, metrics, graph construction, labels
│   ├── 00_data_check.py  …  10_fix_verification.py
│   ├── data/                 ← put the four CSVs here (not in git)
│   ├── labels/               ← put the label snapshot here
│   ├── results/              generated
│   ├── figures/              generated
│   └── results_v16_legacy/   frozen pre-revision results, for comparison
└── paper/
    ├── main.tex              manuscript
    ├── refs.bib              31 entries
    └── figures/              12 PDFs referenced by main.tex
```

---

## Quick start

```bash
cd gnn-decentralization

# 1. Data (about 700 MB, not tracked in git — extract with extract_data.sql)
cp /path/to/{link,uni}_{90d_transfers,balances}.csv data/

# 2. Label snapshot, fetched once and then read locally forever
curl -L -o labels/etherscan_labels_2026-08-26.json \
  https://raw.githubusercontent.com/brianleect/etherscan-labels/main/data/etherscan/combined/combinedAllLabels.json

# 3. Optional: manually verified broker identities
mkdir -p verification && cp /path/to/broker_identities.csv verification/

# 4. Run
python 00_data_check.py        # gate: resolve any [X] before continuing
python 01_build_graph.py       # LINK + UNI at 1w; pass args for 5w / 64w
python 02_c1.py                # C1: three evidence lines, multi-seed, ablation
python 03_c2_weekly.py         # C2: weekly concentration series
python 04_c2_procrustes.py     # C2: window-wise retraining + alignment
python 05_c3_roles.py          # C3: role typology + hidden brokers
python 06_concentration.py     # concentration grid + entity resolution
python 07_governance.py        # rule layer (UNI only)
python 08_figures.py           # all figures, reads results/ only
```

`01`, `02` and `05` accept `TOKEN` and `SCALE` arguments, e.g.
`python 01_build_graph.py LINK 64w`. `05` accepts `--fresh` to ignore all
embedding caches.

Two auxiliary scripts support the manual verification loop:
`09_verify_gap.py` lists broker addresses not yet in the verification table
and diagnoses why previously verified addresses dropped out;
`10_fix_verification.py` recomputes the token-attribution column of that
table and emits a fill-in template.

Every reported number is appended to `results/manifest.csv` with its script,
token, metric name and timestamp.

---

## Three rules the code enforces

**Graph construction has exactly one entry point.** `common.build_graph()`
sums edge weights over repeated transfers before constructing the graph. A
simple directed graph overwrites rather than accumulates a repeated edge's
attribute, and repeat interaction accounts for 90.9% (LINK) and 91.7% (UNI)
of records in the core subgraphs, so an unaggregated construction discards
roughly 85% of transferred volume. Weighted degree and PageRank computed the
two ways correlate at only 0.80–0.83.

**Columns are matched by name, never by position.** `common.find_col()`
raises with the actual column list rather than falling back to `iloc`. The
two balance files have different column counts, and positional indexing
silently returns `total_received` instead of `balance` on one of them.

**Labels are read from an archived snapshot, never from the network.**
`common.load_labels()` fails if the local file is missing. The upstream
compilation is updated continuously — 29,945 entries on 5 August 2026,
29,772 on 9 August — and the `is_core` anchor set derives from it, so a live
query makes supervised results non-reproducible without any error.

---

## Reproducibility

Embeddings are trained on GPU, where the scatter operations underlying
neighbourhood aggregation have no fixed reduction order; seeding does not
make them bit-reproducible. All GNN results are therefore reported as means
over independent initialisations, and broker sets as rank ensembles over ten
of them. Embedding caches carry a fingerprint of the feature matrix and edge
list they were derived from; if `01_build_graph.py` is re-run with different
settings, downstream scripts detect the mismatch and retrain rather than
silently reusing a stale embedding.

Archive `results/emb_*_s*.npy` together with their `.fp` files alongside any
release, so that the reported figures can be recomputed exactly.

---

## Paper

`paper/main.tex` compiles with pdfLaTeX against `refs.bib` and the twelve
PDFs in `paper/figures/`. The figures are generated by `08_figures.py` and
should be copied over after any re-run.

The class line is currently `[final,5p,times]` (two-column, for arXiv). For
journal submission switch to `[review,12pt]`, comment out
`\emergencystretch` and `\tolerance`, and uncomment `\linenumbers`. The
single-column layout also removes the two-column float-placement constraints
that require the `dblfloatfix` settings in the preamble.

---

## Data availability

Transfer and balance data are extracted from
`bigquery-public-data.crypto_ethereum.token_transfers` using the queries in
`extract_data.sql`. The raw CSVs total roughly 700 MB and are not tracked
here; the queries reproduce them exactly, and the row counts are checked by
`00_data_check.py` against the values reported in the paper (1,233,497 LINK
transfers and 433,856 UNI transfers).
