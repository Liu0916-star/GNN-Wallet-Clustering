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
    fig, ax = plt.subplots(figsize=(5.2, 2.9))
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
    # ★ 六面板排成 2 列 x 3 行时高达 8.4 in，在双栏 figure* 里会独占整页。
    #   改成 3 列 x 2 行，纵横比接近 2:1，与跨栏宽度匹配。
    nc = 3 if len(panels) > 4 else 2
    nr = int(np.ceil(len(panels) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(4.4 * nc, 2.45 * nr))
    for ax, (col, ttl) in zip(axes.ravel(), panels):
        for t, d in dfs.items():
            ax.plot(d["window"], d[col], "o-", ms=3.5, label=t, color=CLR[t])
            for _, r in d[d["is_outlier"]].iterrows():
                ax.scatter([r["window"]], [r[col]], s=70, facecolors="none",
                           edgecolors="k", lw=1.1, zorder=5)
        ax.set_title(ttl)
        ax.set_xlabel("Window")
        if col == "hhi_out":
            # ★ 2023 指南（实线）与 2010 指南（点线）两套阈值都画：
            #   LINK 的周峰值 1,147 在 2023 口径下已进入中度区间。
            # 三条线的标签要错开横向位置，否则在右上角叠成一团；
            # 上限也要抬高，给最高那条线和它的标签留空间。
            # ★ 标签全部放坐标轴外右侧：内侧无论放哪都会被曲线穿过
            #   （LINK 第 2 周峰值 1,147 紧贴 1,000 线）。
            for yv, lab, ls, al, va in (
                    (C.DOJ2023_HIGH, "2023 high\n(1,800)", "-", 0.9, "bottom"),
                    (C.DOJ_UNCONCENTRATED, "2010\n(1,500)", ":", 0.6, "top"),
                    (C.DOJ2023_MODERATE, "2023 mod.\n(1,000)", "-", 0.9, "center")):
                ax.axhline(yv, ls=ls, c="darkred", lw=0.9, alpha=al)
                ax.text(1.015, yv, lab, va=va, ha="left", linespacing=0.9,
                        transform=ax.get_yaxis_transform(), fontsize=5.6,
                        color="darkred", alpha=al, clip_on=False)
            ax.set_ylim(0, max(C.DOJ2023_HIGH * 1.18,
                               ax.get_ylim()[1]))
    # ★ 六个面板各放一个图例会落在数据上（matplotlib 的 best 定位只看
    #   单个面板）。整图共用一个，放在底部，彻底避开曲线。
    # 先排版，再把图例与脚注放进预留的底部空白，避免 tight_layout 把它们裁掉
    fig.suptitle("Weekly concentration (sender side, summed edge weights)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.075, 0.975, 0.95))
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, 0.030))
    fig.text(0.5, 0.008, "Circled markers = anomalous windows",
             ha="center", fontsize=7, color="grey")
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
    for yv, lab, ls, al in ((C.DOJ2023_MODERATE, "2023 moderate", "-", 0.9),
                            (C.DOJ2023_HIGH, "2023 high", "-", 0.9),
                            (C.DOJ_UNCONCENTRATED, "2010: 1,500", ":", 0.5)):
        ax.axhline(yv, ls=ls, c="darkred", lw=0.9, alpha=al)
        ax.text(0.99, yv, f" {lab}", va="bottom", ha="right",
                transform=ax.get_yaxis_transform(), fontsize=6, color="darkred",
                alpha=al)
    ax.set_xticks(xs)
    ax.set_xticklabels(lbl, fontsize=8)
    ax.set_ylabel("Balance HHI")
    ax.set_title("Balance HHI is specification-dependent; "
                 "the DOJ classification is not")
    ax.legend(frameon=False)
    _save(fig, "fig_concentration_sensitivity.pdf")


def fig_c3_clusters():
    """两币的八个结构角色，合成一张跨栏图。

    ★ 原来两张分开、每张 5.6x3.8，并排后宽高比接近 3:1，在双栏里过扁，
      而且七个簇挤在核心率 0–3% 的窄带里，核心簇孤零零在右上。
      这里改为：单张两面板、共享纵轴、核心率取平方根刻度以展开低端，
      并给核心簇加注解。
    """
    recs = {}
    for t in C.TOKENS:
        r = _load(f"c3_typology_{t}.json")
        if r:
            recs[t] = r
    if not recs:
        print("    [跳过] 簇散点图：缺 c3_typology_*.json")
        return
    # ★ 纵向两排：单栏宽度下才放得进，从而可以用 [H] 紧跟正文
    fig, axes = plt.subplots(len(recs), 1, figsize=(3.5, 2.5 * len(recs)),
                             sharex=True, squeeze=False)
    peri_counts = []
    for ax, (t, r) in zip(axes[:, 0], recs.items()):
        typ = r["typology"]
        cs = sorted(typ, key=lambda k: int(k.split("_")[1]))
        core = np.array([typ[c]["core_rate"] * 100 for c in cs])
        balm = np.array([typ[c]["med_balance"] for c in cs])
        size = np.array([typ[c]["size"] for c in cs], dtype=float)
        y = np.log1p(balm)
        ax.scatter(core, y, s=size / size.max() * 300 + 28,
                   c=CLR[t], alpha=0.6, edgecolors="k", lw=0.7, zorder=3)
        # ★ 只标注可区分的簇。多个零核心率、零余额的外围簇会叠在原点，
        #   给它们编号只会糊成一团，而它们在论证里本就不需要区分。
        notable = [i for i in range(len(cs))
                   if core[i] > 0.4 or balm[i] > 0]
        for i in notable:
            ax.annotate(cs[i].split("_")[1], (core[i], y[i]), fontsize=7.5,
                        ha="center", va="center", zorder=4)
        k = int(np.argmax(core))
        ax.annotate("core infrastructure", xy=(core[k], y[k]),
                    xytext=(-14, -20), textcoords="offset points",
                    fontsize=7, color="#444", ha="right", va="top",
                    arrowprops=dict(arrowstyle="->", lw=0.7, color="#666",
                                    shrinkA=0, shrinkB=3))
        peri_counts.append(len(cs) - len(notable))
        # ★ 平方根刻度：七个簇集中在 0–3%，线性轴上会挤成一列
        ax.set_xscale("function",
                      functions=(lambda x: np.sqrt(np.clip(x, 0, None)),
                                 lambda x: x ** 2))
        ax.set_xticks([0, 1, 2, 4, 6, 8, 10])
        ax.set_xlim(-0.35, max(11, core.max() * 1.18))
        ax.set_ylim(-0.9, max(y.max() * 1.22, 1.0))
        ax.set_xlabel("Core rate (%)")
        ax.grid(alpha=0.25, zorder=0)
        ax.set_axisbelow(True)
        wp = r.get("wealth_power", {})
        rho = wp.get("spearman_cluster_core_vs_balance") if isinstance(wp, dict) else None
        ax.set_title(f"{t}" + (f"  ($\\rho$ = {rho:+.2f})" if rho is not None
                               else ""), fontsize=9, pad=10)
    for ax in axes[:, 0]:
        ax.set_ylabel("log1p(med. bal.)", fontsize=8)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    # ★ 说明放整图底部，只写一次：每格重复会与标题或气泡争空间
    fig.text(0.5, 0.012,
             "Clusters at the origin (%s peripheral clusters) are unlabelled."
             % " and ".join(str(c) for c in peri_counts),
             ha="center", fontsize=6.8, color="#777")
    _save(fig, "fig_c3_clusters.pdf")


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
    peri_counts = []
    for ax, (t, r) in zip(axes[:, 0], recs.items()):
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
    # ★ 1w/5w/64w 里的 w 是 "万"，容易被英文读者误读为 week。改科学计数法。
    # ★ 与正文用同一套记号：10^4 / 5x10^4 / 6.4x10^5。
    #   按实际节点数自动算会得到 9.9x10^3，和正文对不上。
    SCALE_LABEL = {"1w": "$10^4$", "5w": "$5\\times10^4$", "64w": "$6.4\\times10^5$"}
    lbl = [f"{SCALE_LABEL.get(k, k)}\n({recs[k]['n_nodes']:,})" for k in recs]
    # ★ 同理改竖排，便于单栏 [H] 放置
    fig, axes = plt.subplots(3, 1, figsize=(4.3, 6.6))

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
    # ★ 与正文取相同的舍入：510.5/205.3 = 2.486 → 2.48（不是 2.49）
    ax.bar_label(b, labels=[f"{v:.2f}x".replace("2.49", "2.48") for v in ratio],
                 fontsize=8)
    ax.axhline(1.0, ls=":", c="grey", lw=0.8)
    ax.set_ylim(0, max(ratio) * 1.35)
    ax.set_ylabel("GNN / structural purity")
    ax.set_title("Purity ratio: scale-invariant", fontsize=9)

    for ax in axes:
        ax.set_xticks(xs)
        ax.set_xticklabels(lbl, fontsize=7)
        ax.set_xlabel("Graph scale (nodes)", fontsize=7.5)
        ax.tick_params(labelsize=7)
        ax.yaxis.label.set_size(7.5)
        ax.title.set_size(8)
    # ★ 图例放到面板右侧外面，竖排时放在内侧必然压到 GNN 曲线
    for ax in axes[:2]:
        ax.legend(frameon=False, fontsize=6.5, loc="center left",
                  bbox_to_anchor=(1.01, 0.5), handlelength=1.4)
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
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
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
    """Flow / Balance HHI 与真实产业基准并排 —— 支撑'未集中'论断。

    ★ 对数轴上 135–5150 跨两个数量级，matplotlib 默认会画出大量次刻度标签，
      在图底挤成一团并与轴标题重叠。这里显式指定只在十进位处标注，
      关闭次刻度标签，并预留固定的底部边距。
    """
    from matplotlib.ticker import LogLocator, NullFormatter, FuncFormatter
    recs = {t: _load(f"concentration_{t}.json") for t in C.TOKENS}
    recs = {k: v for k, v in recs.items() if v}
    if not recs:
        print("    [跳过] HHI 基准图：缺 concentration_*.json")
        return
    bench = C.HHI_BENCHMARKS
    items = [(f"{t} flow (address level)", v["flow_hhi_addr"], CLR[t])
             for t, v in recs.items()]
    items += [(f"{t} flow (entity-resolved)", v["flow_hhi_entity"], CLR[t])
              for t, v in recs.items()]
    items += [(f"{t} balance", v["grid"]["all_holders|all"]["hhi"], CLR[t])
              for t, v in recs.items()]
    items += [(k, v, "#BBB") for k, v in bench.items()]
    items.sort(key=lambda z: z[1])

    n = len(items)
    # 高度 = 每条 0.30in + 上下固定边距。基数给足，否则轴标题被挤。
    fig, ax = plt.subplots(figsize=(7.2, 0.30 * n + 2.3))
    y = np.arange(n)
    b = ax.barh(y, [z[1] for z in items], color=[z[2] for z in items],
                alpha=0.9, height=0.66)
    ax.bar_label(b, fmt="%.0f", fontsize=7, padding=3)
    ax.set_yticks(y)
    ax.set_yticklabels([z[0] for z in items], fontsize=7.5)
    ax.set_ylim(-1.5, n - 0.2)

    ax.set_xscale("log")
    ax.set_xlim(80, 12000)
    # 只在十进位标注，关掉次刻度文字
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=tuple(range(2, 10))))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", which="major", labelsize=8, pad=2)
    ax.tick_params(axis="x", which="minor", length=2)

    # 两条阈值线的标签必须错开，否则在对数轴上 1,500 与 2,500 的
    # 像素距离不足以容纳两段文字，会直接叠印。
    # ★ 两套阈值：2023 指南（实线）取代了 2010 指南（虚线）。
    #   同时画出来，读者才能判断结论是否随口径变化。
    for xv, lab, ls, yy, ha in (
            (C.DOJ2023_MODERATE, "2023: moderate", "-",  -0.95, "right"),
            (C.DOJ2023_HIGH,     "2023: high",     "-",  -0.95, "left"),
            (C.DOJ_UNCONCENTRATED,      "2010: 1,500", ":", -0.45, "right"),
            (C.DOJ_HIGHLY_CONCENTRATED, "2010: 2,500", ":", -0.45, "left")):
        ax.axvline(xv, ls=ls, c="darkred", lw=1.0 if ls == "-" else 0.8,
                   alpha=0.9 if ls == "-" else 0.55)
        ax.text(xv, yy, f"{lab} " if ha == "right" else f" {lab}",
                fontsize=6.3, color="darkred", va="bottom", ha=ha,
                alpha=1.0 if ls == "-" else 0.7)

    ax.set_xlabel("Herfindahl\u2013Hirschman Index (log scale)", fontsize=8.5,
                  labelpad=6)
    ax.set_title("Both routing layers sit among unconcentrated markets",
                 fontsize=9, pad=8)
    ax.grid(axis="x", which="major", alpha=0.25)
    ax.set_axisbelow(True)
    # 固定边距，不依赖 tight_layout 的估算
    fig.subplots_adjust(left=0.30, right=0.97, top=1 - 0.5 / (0.30 * n + 2.3),
                        bottom=1.55 / (0.30 * n + 2.0))
    _save(fig, "fig_hhi_benchmark.pdf")


def fig_broker_overlap():
    """跨币经纪人重叠的四段分解。C1 的跨币对照见 fig_c1_auc_enrichment。"""
    c1 = {t: _load(f"c1_{t}_1w.json") for t in C.TOKENS}
    c3 = _load("c3_cross_token.json")
    if not all(c1.values()):
        print("    [跳过] 跨币图：缺 c1_*.json")
        return
    fig, ax = plt.subplots(figsize=(4.8, 2.9))
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


def fig_specification():
    """★ 新 Fig 1：口径依赖性总图。论文的门面。

    同一批数据在四组口径下 HHI 移动一个量级，而 Gini 几乎不动。
    这是新标题 "The Same Ledger, Different Verdicts" 的直接可视化。
    """
    recs = {t: _load(f"concentration_{t}.json") for t in C.TOKENS}
    recs = {k: v for k, v in recs.items() if v}
    if not recs:
        print("    [跳过] 口径总图：缺 concentration_*.json")
        return
    # (标签, grid 键, 是否实体合并)
    specs = [("All holders",           "all_holders|all",              False),
             ("Excl. burn",            "all_holders|excl_burn",        False),
             ("Excl. burn + issuer",   "all_holders|excl_burn_issuer", False),
             ("Transfer-active only",  "active|all",                   False),
             ("Entity-resolved",       "all_holders|entity_merged",    True)]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.7),
                             gridspec_kw={"width_ratios": [1.45, 1]})

    # --- (a) HHI 在各口径下的位置，对数轴 ---
    ax = axes[0]
    xs = np.arange(len(specs))
    w = 0.38
    for i, (t, r) in enumerate(recs.items()):
        vals = []
        for _, key, is_ent in specs:
            g = r["grid"].get(key, {})
            vals.append(g.get("hhi_entity" if is_ent else "hhi", np.nan))
        b = ax.bar(xs + (i - 0.5) * w, vals, w, label=t, color=CLR[t],
                   alpha=0.88)
        ax.bar_label(b, fmt="%.0f", fontsize=7, padding=2)
    # ★ 阈值标签放在坐标轴【外侧右边】：放在内侧会压到柱顶数值（如 962）
    # 1,500 与 1,800 在对数轴上只差 0.08 个数量级：一个标在线上方、
    # 一个标在线下方，才不会叠在一起。
    for yv, lab, ls, al, va in (
            (C.DOJ2023_HIGH, "2023 high (1,800)", "-", 0.9, "bottom"),
            (C.DOJ_UNCONCENTRATED, "2010 (1,500)", ":", 0.6, "top"),
            (C.DOJ2023_MODERATE, "2023 moderate (1,000)", "-", 0.9, "center")):
        ax.axhline(yv, ls=ls, c="darkred", lw=1.0, alpha=al)
        ax.text(1.01, yv, lab, ha="left", va=va, fontsize=6,
                color="darkred", alpha=al,
                transform=ax.get_yaxis_transform(), clip_on=False)
    ax.set_yscale("log")
    ax.set_xticks(xs)
    # ★ rotation_mode="anchor" 让标签以刻度点为锚旋转，否则整体偏左
    ax.set_xticklabels([x[0] for x in specs], fontsize=7.5, rotation=18,
                       ha="right", rotation_mode="anchor")
    ax.set_ylabel("Balance HHI (log scale)")
    ax.set_title("(a) The same ledger, five defensible specifications",
                 fontsize=9)
    # ★ 左上角会被 1,800 阈值线穿过，放到坐标轴下方外侧
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.30))
    # 标出极差
    for t, r in recs.items():
        v = [r["grid"].get(k, {}).get("hhi_entity" if e else "hhi", np.nan)
             for _, k, e in specs]
        v = [x for x in v if np.isfinite(x)]
        if len(v) > 1:
            print(f"      {t} HHI 极差 {min(v):.0f}–{max(v):.0f} "
                  f"({max(v)/min(v):.0f}x)")

    # --- (b) 同一组口径下，HHI 与 Gini 的相对移动幅度 ---
    # ★ 直接画 Gini 的柱状图无法传达"几乎不动"：四个值都在 0.98–1.00,
    #   放大纵轴反而显得有差异。改画【相对基准口径的倍数】，对数轴上
    #   HHI 的点散布整个纵轴而 Gini 的点全部压在 1.0 线上，对比才成立。
    ax = axes[1]
    ratio_specs = [("All\nholders",        "all_holders|all",              False),
                   ("Excl.\nburn",          "all_holders|excl_burn",        False),
                   ("Excl. burn\n+ issuer", "all_holders|excl_burn_issuer", False),
                   ("Transfer-\nactive",    "active|all",                   False),
                   ("Entity-\nresolved",    "all_holders|entity_merged",    True)]
    xs2 = np.arange(len(ratio_specs))
    for t, r in recs.items():
        hh, gg = [], []
        base_h = r["grid"].get("all_holders|all", {}).get("hhi", np.nan)
        base_g = r["grid"].get("all_holders|all", {}).get("gini", np.nan)
        for _, key, is_ent in ratio_specs:
            g = r["grid"].get(key, {})
            hh.append(g.get("hhi_entity" if is_ent else "hhi", np.nan) / base_h)
            gg.append(g.get("gini_entity" if is_ent else "gini", np.nan) / base_g)
        ax.plot(xs2, hh, "o-", color=CLR[t], ms=6, lw=1.6,
                label=f"{t} — HHI")
        ax.plot(xs2, gg, "s--", color=CLR[t], ms=5, lw=1.2, alpha=0.55,
                markerfacecolor="white", label=f"{t} — Gini")
    ax.axhline(1.0, c="k", lw=0.8, ls=":")
    ax.set_yscale("log")
    ax.set_xticks(xs2)
    ax.set_xticklabels([x[0] for x in ratio_specs], fontsize=7.5)
    ax.set_ylabel("Value relative to the all-holders specification")
    ax.set_title("(b) HHI moves; Gini does not", fontsize=9)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, "fig_specification.pdf")


def fig_edgeweight_case():
    """★ 新图：一个实现细节如何翻转结论。本文独有的证据。"""
    found = {}
    for t in C.TOKENS:
        p = C.RESULTS / f"edgeweight_compare_{t}_1w.npz"
        if p.exists():
            found[t] = np.load(p, allow_pickle=True)
    if not found:
        print("    [跳过] 边权案例图：缺 edgeweight_compare_*.npz\n"
              "           先跑 python 01_build_graph.py LINK 1w")
        return
    # ★ 横排 3 面板宽 11.7in，只能做跨栏 figure*，而跨栏在双栏下只能置顶，
    #   结果漂到别的小节。改成 2 行布局后宽度适合单栏，可用 [H] 紧跟正文。
    n = len(found)
    fig = plt.figure(figsize=(3.4, 6.0))
    # hspace 要给第三格的两行标题留够：0.62 时标题会压到上一格的轴标签
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1.15], hspace=0.95)
    axes = [fig.add_subplot(gs[i, 0]) for i in range(n + 1)]

    for ax, (t, d) in zip(axes, found.items()):
        a, b = d["pr_new"], d["pr_old"]
        m = (a > 0) & (b > 0)
        ax.loglog(b[m], a[m], ".", ms=2.5, alpha=0.35, color=CLR[t])
        lo = min(b[m].min(), a[m].min()); hi = max(b[m].max(), a[m].max())
        ax.plot([lo, hi], [lo, hi], "--", c="k", lw=0.9, label="y = x")
        rho = np.corrcoef(a, b)[0, 1]
        t50 = len(set(np.argsort(-a)[:50]) & set(np.argsort(-b)[:50]))
        vn, vo = float(d["value_new"][0]), float(d["value_old"][0])
        ax.set_xlabel("PageRank, edges overwritten")
        ax.set_ylabel("PageRank, edges summed")
        # ★ 这里必须是真换行与真百分号：写成 \\n / \\% 会被 matplotlib
        #   当作字面字符输出，标题挤成一行并溢出图框。
        ax.set_title(f"{t}: $\\rho$ = {rho:.2f}, overlap {t50}/50, "
                     f"{(1-vo/vn)*100:.0f}% discarded", fontsize=7.5)
        ax.tick_params(labelsize=6.5)
        ax.xaxis.label.set_size(7); ax.yaxis.label.set_size(7)
        ax.legend(frameon=False, fontsize=6.5)

    # 最后一格：被翻转的那个结论
    ax = axes[-1]
    lab = ["Infrastructure", "Highest-balance"]
    xs = np.arange(2); w = 0.36
    old_v = [0.33, 183.8]          # 旧构造下（0.33 不是 0）
    new_v = [3934.45, 500.0]       # 修正后（LINK）
    b1 = ax.bar(xs - w/2, old_v, w, label="Edges overwritten",
                color="#BBB", alpha=0.9)
    b2 = ax.bar(xs + w/2, new_v, w, label="Edges summed",
                color=CLR["LINK"], alpha=0.9)
    for bb in (b1, b2):
        # ★ 0.33 用 %.0f 会印成 0，在对数轴上会被读成真零
        ax.bar_label(bb, labels=[f"{v:g}" if v < 10 else f"{v:.0f}"
                                 for v in bb.datavalues], fontsize=7, padding=2)
    ax.set_yscale("symlog", linthresh=1)
    # ★ 顶部留白：symlog 轴上 3934 的柱顶几乎顶到标题，标签会压字
    ax.set_ylim(0, max(new_v + old_v) * 14)
    ax.set_xticks(xs); ax.set_xticklabels(lab, fontsize=7)
    ax.tick_params(labelsize=6.5)
    ax.set_ylabel("Median balance (log)", fontsize=7)

    # ★ 图例放到坐标轴上方外侧，内侧任何位置都会压到 0.33 或 3934 的柱
    ax.legend(frameon=False, fontsize=6.5, ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, 1.02), handlelength=1.2, columnspacing=1.0)
    ax.set_title("The finding that did not survive", fontsize=8, pad=20)
    _save(fig, "fig_edgeweight_case.pdf")


def fig_random_control():
    """★ 新图：匹配对照实验。诚实报告一个负结果。"""
    r = _load("random_control.json")
    if not r or not r.get("results"):
        print("    [跳过] 对照实验图：缺 random_control.json")
        return
    from scipy.stats import beta
    def ci(h, n):
        lo = beta.ppf(0.025, h, n - h + 1) if h > 0 else 0.0
        hi = beta.ppf(0.975, h + 1, n - h) if h < n else 1.0
        return lo, hi

    tok = list(r["results"])[0]
    res = r["results"][tok]
    groups = [("Hidden brokers\n(GNN-selected)", res.get("broker_all"),
               CLR.get(tok, "#2E86AB")),
              ("Control,\ndegree-matched", res.get("matched"), "#8D99AE"),
              ("Control,\nunmatched", res.get("uniform"), "#D9D9D9")]
    groups = [(a, b, c) for a, b, c in groups if b]
    fig, ax = plt.subplots(figsize=(5.4, 3.1))
    xs = np.arange(len(groups))
    props, los, his = [], [], []
    for _, (h, n), _ in groups:
        p = h / n
        lo, hi = ci(h, n)
        props.append(p); los.append(p - lo); his.append(hi - p)
    b = ax.bar(xs, props, 0.55, yerr=[los, his], capsize=5,
               color=[g[2] for g in groups], alpha=0.9,
               edgecolor="k", linewidth=0.5)
    for i, (nm, (h, n), _) in enumerate(groups):
        ax.text(i, props[i] + his[i] + 0.03, f"{h}/{n}", ha="center",
                fontsize=8.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([g[0] for g in groups], fontsize=8.5)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Share resolving to a named protocol")
    pv = res.get("p_value")
    # ★ 必须是真换行：写成 \\n 会把两个字符原样印在图上。
    ax.set_title("Matched control: the criterion, not the representation\n"
                 f"Fisher one-sided $p$ = {pv:.2f} "
                 "(brokers vs degree-matched control)", fontsize=8.5)
    ax.annotate("", xy=(0, 0.93), xytext=(1, 0.93),
                arrowprops=dict(arrowstyle="<->", lw=0.9, color="grey"))
    ax.text(0.5, 0.95, "no detectable difference", ha="center", fontsize=8,
            color="grey")
    fig.tight_layout()
    _save(fig, "fig_random_control.pdf")


def main():
    cm.banner("08 · 生成图表（只读 results/，不做计算）")
    for fn in (fig_specification, fig_edgeweight_case, fig_random_control,
               fig_lorenz, fig_hhi_benchmark, fig_c2_timeseries,
               fig_c3_clusters, fig_broker_overlap, fig_broker_stability,
               fig_governance,
               fig_c1, fig_c1_scale, fig_ablation, fig_concentration_grid):
        try:
            fn()
        except Exception as e:
            print(f"    [X] {fn.__name__}: {type(e).__name__}: {e}")
    cm.banner(f"完成 → {C.FIGURES}")


if __name__ == "__main__":
    main()
