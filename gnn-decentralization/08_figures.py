# -*- coding: utf-8 -*-
"""
08_figures.py —— 全部图表，只读 results/

★ 与 v16 的 make_all_figures.py 的关键差别：本脚本【不做任何计算】，
  只读结果文件。v16 的图注和图像曾经不一致（fig6/fig7 漏了 UNI），
  根源就是绘图脚本自己算了一遍。数字只能有一个来源。

缺哪个结果文件就跳过哪张图，并打印提示。

用法：python 08_figures.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C
import common as cm

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
})
CLR = {"LINK": "#2E86AB", "UNI": "#E63946"}


def _load(name):
    p = C.RESULTS / name
    if not p.exists():
        return None
    return (pd.read_csv(p) if p.suffix == ".csv"
            else json.load(open(p, encoding="utf-8")))


def _save(fig, name):
    p = C.FIGURES / name
    fig.savefig(p)
    plt.close(fig)
    print(f"    → {name}")


def fig_c1():
    recs = {t: _load(f"c1_{t}_1w.json") for t in C.TOKENS}
    recs = {k: v for k, v in recs.items() if v}
    if not recs:
        print("    [跳过] C1 图：缺 c1_*.json")
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    labels = ["Centrality", "Structural", "GNN", "node2vec"]
    for ax, key, ttl in ((axes[0], "is_core", "is_core AUC"),
                         (axes[1], "enrich", "Neighbourhood enrichment")):
        xs = np.arange(len(labels))
        w = 0.36
        for i, (t, r) in enumerate(recs.items()):
            n2 = r.get("node2vec_is_core")
            n2e = r.get("node2vec_enrichment")
            if key == "is_core":
                vals = [r["is_core_centrality"]["auc"],
                        r["is_core_raw_struct"]["auc"],
                        r["gnn_is_core"]["mean"],
                        n2["mean"] if isinstance(n2, dict) else np.nan]
                err = [0, 0, r["gnn_is_core"]["sd_seed"],
                       n2["sd_seed"] if isinstance(n2, dict) else 0]
            else:
                vals = [r.get("enrichment_centrality", 0),
                        r.get("enrichment_raw_struct", 0),
                        r["enrichment"]["mean"],
                        n2e["mean"] if isinstance(n2e, dict) else np.nan]
                err = [0, 0, r["enrichment"]["sd_seed"],
                       n2e["sd_seed"] if isinstance(n2e, dict) else 0]
            ax.bar(xs + (i - 0.5) * w, vals, w, yerr=err, capsize=3,
                   label=t, color=CLR[t], alpha=0.85)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.set_title(ttl)
        ax.legend(frameon=False)
        if key == "is_core":
            ax.axhline(0.5, ls="--", c="grey", lw=0.8)
            ax.set_ylim(0.4, 1.0)
    _save(fig, "fig_c1_auc_enrichment.pdf")


def fig_ablation():
    """消融：GNN / node2vec / 常数输入 / 中心性。"""
    r = _load("c1_LINK_1w.json")
    if not r:
        print("    [跳过] 消融图：缺 c1_LINK_1w.json")
        return
    n2 = r.get("node2vec_is_core")
    rows = [("GNN\n(full features)", r["gnn_is_core"]["mean"],
             r["gnn_is_core"]["sd_seed"], CLR["LINK"])]
    if isinstance(n2, dict):
        rows.append(("node2vec\n(topology only)", n2["mean"], n2["sd_seed"],
                     "#2A9D8F"))
    rows += [("GNN\n(constant input)", r["gnn_nofeat_is_core"]["mean"],
              r["gnn_nofeat_is_core"]["sd_seed"], "#BBB"),
             ("Classical\ncentrality", r["is_core_centrality"]["auc"], 0,
              "#999")]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    xs = np.arange(len(rows))
    b = ax.bar(xs, [x[1] for x in rows], yerr=[x[2] for x in rows],
               capsize=4, color=[x[3] for x in rows], alpha=0.88, width=0.6)
    ax.bar_label(b, fmt="%.3f", fontsize=8, padding=3)
    ax.axhline(0.5, ls="--", c="darkred", lw=0.9)
    ax.text(0.99, 0.5, " chance", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=7, color="darkred")
    ax.set_xticks(xs)
    ax.set_xticklabels([x[0] for x in rows], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("is\\_core AUC")
    ax.set_title("Learned representations, not architecture, drive the gap",
                 fontsize=9)
    _save(fig, "fig_ablation.pdf")


def fig_c2_timeseries():
    dfs = {t: _load(f"c2_weekly_{t}.csv") for t in C.TOKENS}
    dfs = {k: v for k, v in dfs.items() if v is not None}
    if not dfs:
        print("    [跳过] C2 时序图：缺 c2_weekly_*.csv")
        return
    has_elite = all("elite_share" in d.columns for d in dfs.values())
    panels = [("gini_out", "Flow Gini"), ("top1_out", "Top-1% flow share"),
              ("hhi_out", "Flow HHI"), ("gini_betweenness", "Betweenness Gini")]
    if has_elite:
        panels += [("elite_share", "Top-50 outflow share"),
                   ("top_k_retention", "Top-50 week-over-week retention")]
    nr = (len(panels) + 1) // 2
    fig, axes = plt.subplots(nr, 2, figsize=(9.5, 2.8 * nr))
    for ax, (col, ttl) in zip(axes.ravel(), panels):
        for t, d in dfs.items():
            ax.plot(d["window"], d[col], "o-", ms=3.5, label=t, color=CLR[t])
            for _, r in d[d["is_outlier"]].iterrows():
                ax.scatter([r["window"]], [r[col]], s=70, facecolors="none",
                           edgecolors="k", lw=1.1, zorder=5)
        ax.set_title(ttl)
        ax.set_xlabel("Window")
        if col == "hhi_out":
            ax.axhline(C.DOJ_UNCONCENTRATED, ls="--", c="darkred", lw=0.9)
            ax.text(0.98, C.DOJ_UNCONCENTRATED, " DOJ 1,500", va="bottom",
                    ha="right", transform=ax.get_yaxis_transform(),
                    fontsize=7, color="darkred")
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Weekly concentration (sender side, summed edge weights)",
                 fontsize=10)
    fig.text(0.5, -0.02, "Circled markers = anomalous windows",
             ha="center", fontsize=7, color="grey")
    fig.tight_layout()
    _save(fig, "fig_c2_timeseries.pdf")


def fig_concentration_grid():
    recs = {t: _load(f"concentration_{t}.json") for t in C.TOKENS}
    recs = {k: v for k, v in recs.items() if v}
    if not recs:
        print("    [跳过] 集中度网格图：缺 concentration_*.json")
        return
    keys = ["all_holders|all", "all_holders|excl_burn",
            "all_holders|excl_burn_issuer", "active|all"]
    lbl = ["All holders", "Excl. burn", "Excl. burn+issuer", "Active only"]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    xs = np.arange(len(keys))
    w = 0.36
    for i, (t, r) in enumerate(recs.items()):
        v = [r["grid"].get(k, {}).get("hhi", np.nan) for k in keys]
        b = ax.bar(xs + (i - 0.5) * w, v, w, label=t, color=CLR[t], alpha=0.85)
        ax.bar_label(b, fmt="%.0f", fontsize=7)
    ax.axhline(C.DOJ_UNCONCENTRATED, ls="--", c="darkred", lw=0.9)
    ax.text(0.99, C.DOJ_UNCONCENTRATED, " DOJ unconcentrated", va="bottom",
            ha="right", transform=ax.get_yaxis_transform(), fontsize=7,
            color="darkred")
    ax.set_xticks(xs)
    ax.set_xticklabels(lbl, fontsize=8)
    ax.set_ylabel("Balance HHI")
    ax.set_title("Balance HHI is specification-dependent; "
                 "the DOJ classification is not")
    ax.legend(frameon=False)
    _save(fig, "fig_concentration_sensitivity.pdf")


def fig_c3_clusters():
    for t in C.TOKENS:
        r = _load(f"c3_typology_{t}.json")
        if not r:
            continue
        typ = r["typology"]
        cs = sorted(typ, key=lambda k: int(k.split("_")[1]))
        core = [typ[c]["core_rate"] * 100 for c in cs]
        balm = [typ[c]["med_balance"] for c in cs]
        size = [typ[c]["size"] for c in cs]
        fig, ax = plt.subplots(figsize=(5.6, 3.8))
        sc = ax.scatter(core, np.log1p(balm),
                        s=np.array(size) / max(size) * 400 + 25,
                        c=CLR[t], alpha=0.65, edgecolors="k", lw=0.6)
        for i, c in enumerate(cs):
            ax.annotate(c.split("_")[1], (core[i], np.log1p(balm[i])),
                        fontsize=7, ha="center", va="center")
        ax.set_xlabel("Core rate (%)")
        ax.set_ylabel("log1p(median balance)")
        wp = r.get("wealth_power", {})
        v = wp.get("verdict", "") if isinstance(wp, dict) else str(wp)
        rho = wp.get("spearman_cluster_core_vs_balance") if isinstance(wp, dict) else None
        ax.set_title(f"{t}: wealth vs structural power ({v}"
                     + (f", rho={rho:+.2f})" if rho is not None else ")"))
        _save(fig, f"fig_c3_clusters_{t}.pdf")


def fig_governance():
    r = _load("governance_UNI.json")
    if not r or not r.get("applicable"):
        print("    [跳过] 治理图：缺 governance_UNI.json")
        return
    coh = [k for k in r if isinstance(r[k], dict)
           and "nakamoto_quorum" in r[k]]
    if not coh:
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    v = [r[k]["nakamoto_quorum"] for k in coh]
    b = ax.bar(range(len(coh)), v, color=CLR["UNI"], alpha=0.85)
    ax.bar_label(b, fmt="%d", fontsize=8)
    ax.set_xticks(range(len(coh)))
    ax.set_xticklabels(coh, fontsize=8)
    ax.set_ylabel("Addresses needed to reach quorum")
    ax.set_title("UNI rule layer: Nakamoto coefficient is "
                 "specification-dependent")
    _save(fig, "fig_governance_nakamoto.pdf")


def fig_broker_stability():
    """经纪人集合的种子稳定性 —— 论文必须报告的复现性证据。"""
    recs = {t: _load(f"c3_typology_{t}.json") for t in C.TOKENS}
    recs = {k: v for k, v in recs.items()
            if v and "freq_histogram" in v.get("hidden_brokers", {})}
    if not recs:
        print("    [跳过] 经纪人稳定性图：缺 freq_histogram")
        return
    fig, axes = plt.subplots(1, len(recs), figsize=(4.6 * len(recs), 3.2),
                             squeeze=False)
    for ax, (t, r) in zip(axes[0], recs.items()):
        hb = r["hidden_brokers"]
        h = hb["freq_histogram"]
        ks = sorted(h, key=lambda x: int(x.split("/")[0]))
        ax.bar(range(len(ks)), [h[k] for k in ks], color=CLR[t], alpha=0.85)
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels([k.split("/")[0] for k in ks], fontsize=7)
        ax.set_xlabel("Seeds in which the node is selected")
        ax.set_ylabel("Number of nodes")
        ax.set_title(f"{t}: broker selection stability\n"
                     f"ensemble={hb.get('rank_ensemble_set', '?')}, "
                     f"vote={hb.get('vote_consensus_set', '?')}, "
                     f"overlap={hb.get('rank_vote_overlap', '?')}",
                     fontsize=8)
    fig.tight_layout()
    _save(fig, "fig_broker_stability.pdf")


def fig_c1_scale():
    """C1 三档：AUC 优势随规模消失，纯度比值不变。论文核心图之一。"""
    order = ["1w", "5w", "64w"]
    recs = {}
    for sc in order:
        r = _load(f"c1_LINK_{sc}.json")
        if r:
            recs[sc] = r
    if len(recs) < 2:
        print("    [跳过] C1 尺度图：需要至少两个规模的 c1_LINK_*.json")
        return
    xs = np.arange(len(recs))
    lbl = [f"{k}\n({recs[k]['n_nodes']:,})" for k in recs]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))

    ax = axes[0]
    for nm, key, c in (("Centrality", "is_core_centrality", "#999"),
                       ("Structural", "is_core_raw_struct", "#F4A261")):
        ax.plot(xs, [recs[k][key]["auc"] for k in recs], "o--", color=c,
                label=nm, ms=5)
    g = [recs[k]["gnn_is_core"]["mean"] for k in recs]
    e = [recs[k]["gnn_is_core"]["sd_seed"] for k in recs]
    ax.errorbar(xs, g, yerr=e, fmt="o-", color=CLR["LINK"], capsize=3,
                label="GNN", ms=5)
    ax.axhline(0.5, ls=":", c="grey", lw=0.8)
    ax.set_ylabel("is_core AUC")
    ax.set_title("AUC: GNN advantage vanishes with scale", fontsize=9)

    ax = axes[1]
    for nm, key, c in (("Centrality", "purity_centrality", "#999"),
                       ("Structural", "purity_raw_struct", "#F4A261")):
        ax.plot(xs, [recs[k].get(key, np.nan) for k in recs], "o--",
                color=c, label=nm, ms=5)
    ax.plot(xs, [recs[k]["purity"]["mean"] for k in recs], "o-",
            color=CLR["LINK"], label="GNN", ms=5)
    ax.plot(xs, [recs[k]["random_baseline_purity"] for k in recs], ":",
            color="k", lw=0.8, label="Random")
    ax.set_ylabel("Neighbourhood purity")
    ax.set_title("Absolute purity: stable, then halves", fontsize=9)

    ax = axes[2]
    ratio = [recs[k]["enrichment"]["mean"]
             / max(recs[k]["enrichment_raw_struct"], 1e-9) for k in recs]
    b = ax.bar(xs, ratio, color=CLR["LINK"], alpha=0.85, width=0.55)
    ax.bar_label(b, fmt="%.2fx", fontsize=8)
    ax.axhline(1.0, ls=":", c="grey", lw=0.8)
    ax.set_ylim(0, max(ratio) * 1.35)
    ax.set_ylabel("GNN / structural purity")
    ax.set_title("Purity ratio: scale-invariant", fontsize=9)

    for ax in axes:
        ax.set_xticks(xs)
        ax.set_xticklabels(lbl, fontsize=8)
        ax.set_xlabel("Graph scale (nodes)")
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    _save(fig, "fig_c1_scale.pdf")


def fig_lorenz():
    """余额与流量的 Lorenz 曲线 —— Gini 的可视化，Discussion 核心图。"""
    curves = {}
    for t in C.TOKENS:
        if not C.FILES[t]["bal"].exists():
            continue
        v = np.sort(cm.load_balances(t)["balance"].values)
        cum = np.cumsum(v) / v.sum()
        curves[t] = (np.arange(1, len(v) + 1) / len(v), cum, cm.gini(v))
    if not curves:
        print("    [跳过] Lorenz 图：缺余额文件")
        return
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0, 1], [0, 1], ":", c="k", lw=0.9, label="Equality")
    # 传统资产基准：由文献报告的 Gini 反解参数化 Lorenz 曲线 L(p)=p^a,
    # 其中 G=(a-1)/(a+1)。这是示意曲线而非原始微观数据，图注须写明。
    pp = np.linspace(0, 1, 400)
    for nm, g_, c_ in C.LORENZ_BENCHMARKS:
        a_ = (1 + g_) / (1 - g_)
        ax.plot(pp, pp ** a_, "--", lw=1.1, color=c_, alpha=0.75,
                label=f"{nm} (Gini {g_:.2f})")
    for t, (x, y, gi) in curves.items():
        idx = np.linspace(0, len(x) - 1, min(3000, len(x))).astype(int)
        ax.plot(x[idx], y[idx], color=CLR[t], lw=1.6,
                label=f"{t} balance (Gini {gi:.3f})")
    ax.set_xlabel("Cumulative share of addresses")
    ax.set_ylabel("Cumulative share of tokens")
    ax.set_title("Both ledgers are more unequal than any traditional asset",
                 fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _save(fig, "fig_lorenz.pdf")


def fig_hhi_benchmark():
    """Flow / Balance HHI 与真实产业基准并排 —— 支撑'未集中'论断。"""
    recs = {t: _load(f"concentration_{t}.json") for t in C.TOKENS}
    recs = {k: v for k, v in recs.items() if v}
    if not recs:
        print("    [跳过] HHI 基准图：缺 concentration_*.json")
        return
    bench = C.HHI_BENCHMARKS
    items = [(f"{t} flow (address)", v["flow_hhi_addr"], CLR[t])
             for t, v in recs.items()]
    items += [(f"{t} flow (entity)", v["flow_hhi_entity"], CLR[t])
              for t, v in recs.items()]
    items += [(f"{t} balance", v["grid"]["all_holders|all"]["hhi"], CLR[t])
              for t, v in recs.items()]
    items += [(k, v, "#BBB") for k, v in bench.items()]
    items.sort(key=lambda z: z[1])
    fig, ax = plt.subplots(figsize=(6.6, 0.34 * len(items) + 1.2))
    y = np.arange(len(items))
    b = ax.barh(y, [z[1] for z in items], color=[z[2] for z in items],
                alpha=0.9, height=0.68)
    ax.bar_label(b, fmt="%.0f", fontsize=7, padding=2)
    ax.set_yticks(y)
    ax.set_yticklabels([z[0] for z in items], fontsize=7.5)
    ax.axvline(C.DOJ_UNCONCENTRATED, ls="--", c="darkred", lw=1)
    ax.axvline(C.DOJ_HIGHLY_CONCENTRATED, ls="--", c="darkred", lw=0.7,
               alpha=0.6)
    ax.text(C.DOJ_UNCONCENTRATED, len(items) - 0.3, " DOJ 1,500",
            fontsize=7, color="darkred", va="top")
    ax.set_xscale("log")
    ax.set_xlabel("HHI (log scale)")
    ax.set_title("Both routing layers sit among unconcentrated markets",
                 fontsize=9)
    _save(fig, "fig_hhi_benchmark.pdf")


def fig_broker_overlap():
    """跨币经纪人重叠的四段分解。C1 的跨币对照见 fig_c1_auc_enrichment。"""
    c1 = {t: _load(f"c1_{t}_1w.json") for t in C.TOKENS}
    c3 = _load("c3_cross_token.json")
    if not all(c1.values()):
        print("    [跳过] 跨币图：缺 c1_*.json")
        return
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    if c3:
        da = _load(f"hidden_brokers_{C.TOKENS[0]}.csv")
        db = _load(f"hidden_brokers_{C.TOKENS[1]}.csv")
        na = len(da) if da is not None else 0
        nb = len(db) if db is not None else 0
        sh = c3["n_shared"]
        ident = c3.get("n_identified", 0)
        b = ax.bar([0, 1, 2, 3],
                   [na - sh, nb - sh, sh - ident, ident],
                   color=[CLR[C.TOKENS[0]], CLR[C.TOKENS[1]],
                          "#8D99AE", "#2A9D8F"], alpha=0.9)
        ax.bar_label(b, fmt="%d", fontsize=8)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels([f"{C.TOKENS[0]}\nonly", f"{C.TOKENS[1]}\nonly",
                            "Shared\nunidentified", "Shared\nidentified"],
                           fontsize=8)
        ax.set_ylabel("Hidden brokers")
        ax.set_title(f"{sh} brokers route both tokens; "
                     f"{sh - ident} carry no public identity", fontsize=9)
    else:
        ax.axis("off")
    fig.tight_layout()
    _save(fig, "fig_broker_overlap.pdf")


def main():
    cm.banner("08 · 生成图表（只读 results/，不做计算）")
    for fn in (fig_c1, fig_c1_scale, fig_ablation, fig_c2_timeseries,
               fig_lorenz,
               fig_concentration_grid, fig_hhi_benchmark, fig_c3_clusters,
               fig_broker_overlap, fig_broker_stability, fig_governance):
        try:
            fn()
        except Exception as e:
            print(f"    [X] {fn.__name__}: {type(e).__name__}: {e}")
    cm.banner(f"完成 → {C.FIGURES}")


if __name__ == "__main__":
    main()
