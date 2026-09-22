# The Same Ledger, Different Verdicts

**How Measurement Specification Determines On-Chain Concentration**

Jintao Liu\*, Zhimo Ji\* and Xuzhe Lin · \*School of Computing and Information Systems, The University of Melbourne · School of Mathematics and Statistics, The University of Sydney

\* Equal contribution

[![arXiv](https://img.shields.io/badge/arXiv-2609.24176-b31b1b.svg)](https://arxiv.org/abs/2609.24176)

---

Whether a public blockchain is "decentralized" is routinely settled by citing a
concentration statistic. This repository contains the data pipeline, results
and manuscript for a paper showing that, on two ERC-20 ledgers observed over an
identical 90-day window, that verdict is a property of the measurement rather
than of the ledger.

**Four discretionary choices move the balance HHI of a single ledger by a factor
of 21** (UNI: 109 to 2,336), every specification defensible and none dictated
by the data. Over the same range the Gini coefficient moves by less than 0.003.
We also document a case in which one implementation detail — whether repeated
transfers between an address pair are summed or overwritten — overturned a
finding that had been written into an earlier draft of the paper.

| | LINK (oracle) | UNI (governance) |
|---|---|---|
| Transfers / active addresses | 1,233,497 / 642,126 | 433,856 / 38,667 |
| Balance Gini | 0.990 | 0.998 |
| Balance HHI, across specifications | 115 – 562 | 109 – 2,336 |
| Weekly flow HHI (address / entity-resolved) | 421 / 654 | 386 / 667 |
| Hidden brokers (rank ensemble, 10 seeds) | 30 | 18 |
| — shared across both ledgers | 12 | 12 |
| — of which resolve to a named protocol | 6 | 6 |

Observation window: 2026-03-26 to 2026-06-24 UTC. Concentration verdicts are
reported against both the 2010 Horizontal Merger Guidelines and the 2023
Merger Guidelines that replaced them.

---

## Repository layout

```
.
├── paper/
│   ├── main.tex                  manuscript (arXiv version)
│   ├── refs.bib
│   └── figures/                  the eight figures used in main.tex
└── gnn-decentralization/
    ├── config.py                 every path, threshold and hyperparameter
    ├── common.py                 loading, metrics, graph construction, labels
    ├── 00_data_check.py … 08_figures.py      main pipeline
    ├── 09_verify_gap.py, 10_fix_verification.py   verification bookkeeping
    ├── 11_random_control.py      matched control experiment
    ├── 12_reverify_all.py        full re-verification of broker identities
    ├── extract_data.sql          BigQuery extraction
    ├── labels/                   archived Etherscan label snapshot
    ├── verification/             manual on-chain verification records
    ├── results/                  derived artefacts (see below)
    └── results_v16_legacy/       results of the superseded construction
```

## What is archived, and where

The manuscript commits to archiving several artefacts. This table maps each
commitment to its location.

| Manuscript commitment | Location |
|---|---|
| Label snapshot, retrieved 26 Aug 2026, MD5 `16792aac5afd` | `labels/etherscan_labels_2026-08-26.json` |
| Embeddings underlying every reported ensemble, with fingerprints | `results/emb_*_1w_s*.npy` and `.fp` |
| Node features | `results/graph_*_1w.npz`, `graph_*_5w.npz` |
| Identified broker sets and cluster assignments | `results/hidden_brokers_*.csv`, `c3_typology_*.json` |
| Full per-address verification record | `verification/broker_identities.csv`, `reverify_worklist.csv` |
| Matched control experiment (blind list, key, result) | `verification/control_*.csv`, `results/random_control.json` |
| Phishing airdrop: 38 contracts, 13 funders, Etherscan classification | `verification/eventstudy_w7_phishing_addrs.json` |
| Edge-weight correction, before and after | `results/edgeweight_compare_*.npz`, `results_v16_legacy/` |
| 6.4 × 10⁵-node graph and embedding (≈140 MB) | Zenodo: DOI to be added |

Raw transfer and balance data (≈700 MB) are not tracked. `extract_data.sql`
reproduces them exactly from `bigquery-public-data.crypto_ethereum.token_transfers`,
and `00_data_check.py` verifies the row counts against the figures above.

## Reproducing the results

```bash
cd gnn-decentralization
pip install -r ../requirements.txt

# place the four CSVs from extract_data.sql in data/, then:
python 00_data_check.py          # gate: resolve any [X] before continuing
python 01_build_graph.py         # LINK and UNI at 10^4; pass args for 5w / 64w
python 02_c1.py                  # instrument evaluation, multi-seed
python 03_c2_weekly.py           # weekly concentration series
python 04_c2_procrustes.py       # window-wise retraining and alignment
python 05_c3_roles.py            # role typology and hidden brokers
python 06_concentration.py       # specification grid and entity resolution
python 07_governance.py          # rule layer (UNI only)
python 08_figures.py             # every figure, from results/ only
```

To reproduce the figures and tables without retraining, the archived
`results/` is sufficient: `05_c3_roles.py` reuses cached embeddings whose
fingerprint matches the current graph, and `08_figures.py` reads only
`results/`.

## Four things that determine whether the numbers reproduce

**Edge weights must be summed.** Standard graph libraries overwrite a repeated
edge's attribute rather than accumulating it, without raising an error. Repeat
interaction accounts for over 90% of records in the core subgraphs, so the
overwrite construction discards roughly 85% of transferred volume.
`common.build_graph()` is the single entry point and sums explicitly. This is
the detail that overturned the earlier draft's finding (manuscript §5.5).

**Labels must come from the archived snapshot.** The upstream compilation is
updated continuously and silently. `common.load_labels()` refuses to query it
live.

**GNN training on GPU is not bit-reproducible.** Scatter operations have no
fixed reduction order, so seeding alone does not fix the result. Every GNN
figure is a mean over independent initialisations, and the broker set is a
rank ensemble over ten. Embedding caches carry a fingerprint of the feature
matrix and edge list; a mismatch triggers retraining rather than silent reuse.

**Manual verification must use the token-transfer view.** A routing contract's
Etherscan transaction count records only calls made to it directly — the most
central shared broker shows six transactions against several thousand token
transfers. Etherscan also caps the displayed transfer count at 10,000, which the
verification records note as a lower bound where it binds.

## A negative result we report

A degree-matched control drawn from the same eligibility pool as the hidden
brokers resolves to named routing protocols at 35%, against 37% for the brokers
(Fisher one-sided *p* = 0.57). The discriminative work is done by the criterion
— in particular by counterparty count — and not by the graph neural network used
to operationalise "structurally central". `11_random_control.py` implements the
design, including blinding and a calibration check on the verifier.

## Citation

```bibtex
@article{liu2026sameledger,
  title   = {The Same Ledger, Different Verdicts: How Measurement Specification
             Determines On-Chain Concentration},
  author  = {Liu, Jintao and Ji, Zhimo and Lin, Xuzhe},
  journal = {arXiv preprint arXiv:2609.24176},
  year    = {2026}
}
```

## Licence

Code: MIT (see `LICENSE`). Manuscript: CC BY-NC-ND 4.0, as deposited on arXiv.
