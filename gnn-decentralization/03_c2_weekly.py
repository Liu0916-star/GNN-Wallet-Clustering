# -*- coding: utf-8 -*-
"""
03_c2_weekly.py —— C2 轻量版：周度集中度序列

替代 v16 的 cell 18 + Hhi_calculation。三处修正：

  1. 边权求和（common.build_graph），不再让重复边互相覆盖
  2. ★ 口径同源：Gini / top1% / HHI 全部按【周度 · 发送方 · 求和边权】计算。
     v16 的三个数来自两个脚本、三种组合：
        Gini 0.976    cell 18          入流 · 12窗(剔W7) · 覆盖边权
        top1% 82.3%   cell 18          入流 · 12窗(剔W7) · 覆盖边权 · 含零值分母
        Flow HHI 421  Hhi_calculation  出流 · 13窗(全留) · 求和边权
     现在统一，并把「接收方」「全窗汇总」「剔除异常窗」「v16 top1% 实现」
     全部作为敏感性行并列输出。
  3. 落盘 results/c2_weekly_{TOKEN}.csv，08_figures.py 只读它

用法：python 03_c2_weekly.py
"""
import numpy as np
import pandas as pd

import config as C
import common as cm

SCRIPT = "03_c2_weekly"


def run(token):
    cm.banner(f"03 · C2 周度序列 {token}")
    tx = cm.load_transfers(token, drop_burn=True)
    outliers = cm.find_outlier_windows(tx)
    print(f"  转账 {len(tx):,} | 周窗 {tx['week'].nunique()} | "
          f"异常窗 {outliers if outliers else '无'}")

    rows = []
    elite_sets = {}          # 周 → 该周出流 top-K 的地址集合
    for w, g in tx.groupby("week", sort=True):
        G = cm.build_graph(g, top_n=None, largest_wcc=False, verbose=False)
        outf = np.array(list(dict(G.out_degree(weight="w")).values()))
        inf = np.array(list(dict(G.in_degree(weight="w")).values()))
        import networkx as nx
        nb = G.number_of_nodes()
        if nb > 100_000:
            print(f"    周{int(w) + 1}: {nb:,} 节点，betweenness 采样较慢"
                  f"（预计 5–20 分钟），请勿中断 ...", flush=True)
        bet = nx.betweenness_centrality(
            G, k=min(C.BETWEENNESS_K["c2_window"], nb), seed=C.SEED)
        bv = np.array(list(bet.values()))
        # ★ 采样退化检测：k=200 在 55 万节点上几乎所有点的介数为 0，
        #   Gini 会退化成 0，那不是"结构完全均等"，是这一格无效。
        zero_frac = float((bv == 0).mean())
        g_bet = cm.gini(bv) if zero_frac < 0.99 else np.nan
        if zero_frac >= 0.99:
            print(f"    [!] 周{int(w) + 1} betweenness {zero_frac * 100:.1f}% 为零"
                  f"（k={min(C.BETWEENNESS_K['c2_window'], nb)} 采样不足）"
                  f"→ 该格记为 NaN，附录需说明")
        # ---- 精英集：该周出流 top-K 地址 ----
        # v16 报告 "top50 周际留存 53%/48%" 与 "精英集流量份额均值 0.748"，
        # 但那两个数出自已废弃的脚本。这里在统一口径下重算。
        ow = g.groupby("from")["amt"].sum().sort_values(ascending=False)
        topk = list(ow.head(C.ELITE_TOP_K).index)
        elite_sets[int(w)] = set(topk)
        elite_share = float(ow.head(C.ELITE_TOP_K).sum() / max(ow.sum(), 1e-12))

        rows.append({
            "window": int(w) + 1,
            "elite_share": elite_share,
            "start": (tx["ts"].min().normalize()
                      + pd.Timedelta(days=int(w) * C.WINDOW_DAYS)).date(),
            "txns": len(g), "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(), "volume": float(g["amt"].sum()),
            "gini_out": cm.gini(outf), "gini_in": cm.gini(inf),
            "top1_out": cm.top_share(outf, frac=0.01),
            "top1_in": cm.top_share(inf, frac=0.01),
            "top1_in_v16": cm.top_share_v16(inf),      # 仅供新旧对比
            "hhi_out": cm.hhi(outf), "hhi_in": cm.hhi(inf),
            "gini_betweenness": g_bet, "betweenness_zero_frac": round(zero_frac, 4),
            "is_outlier": int(w) in outliers,
        })
        r = rows[-1]
        print(f"    周{r['window']:2d} {r['start']}  转账{r['txns']:>7,} "
              f"节点{r['nodes']:>7,} | Gini={r['gini_out']:.4f} "
              f"top1%={r['top1_out'] * 100:5.1f}% HHI={r['hhi_out']:7.1f}")

    df = pd.DataFrame(rows)
    # ---- 周际留存：相邻两周 top-K 的交集比例 ----
    # ★ 留存是【一对】窗口的属性。异常窗会污染两个值：它自己（W7 vs W6）
    #   和它的下一周（W8 vs W7）。若只按 is_outlier 剔除单行，W8 那个被污染的
    #   值仍会留在 clean 里。这里额外标记 retention_contaminated。
    ws_ = sorted(elite_sets)
    ret = [np.nan]
    contam = [True]                      # 第一窗无前序，本就无留存值
    for i in range(1, len(ws_)):
        a_, b_ = elite_sets[ws_[i - 1]], elite_sets[ws_[i]]
        ret.append(len(a_ & b_) / max(len(b_), 1))
        contam.append(ws_[i - 1] in outliers or ws_[i] in outliers)
    df["top_k_retention"] = ret
    df["retention_contaminated"] = contam
    clean = df[~df["is_outlier"]]
    ret_clean = df.loc[~df["retention_contaminated"], "top_k_retention"]
    cm.save_csv(df, f"c2_weekly_{token}.csv")

    # ---------- 主口径 ----------
    cm.banner(f"{token} 主口径（周度 · 发送方 · 求和边权 · "
              f"{'剔除' if C.EXCLUDE_OUTLIER_WINDOWS else '保留'}异常窗）", "-")
    use = clean if C.EXCLUDE_OUTLIER_WINDOWS else df
    main = {"gini_flow": use["gini_out"].mean(),
            "top1pct_flow": use["top1_out"].mean(),
            "flow_hhi": use["hhi_out"].mean()}
    print(f"    流量 Gini   {main['gini_flow']:.4f}")
    print(f"    top1% 份额  {main['top1pct_flow'] * 100:.1f}%")
    print(f"    Flow HHI    {main['flow_hhi']:.1f}   "
          f"{'不集中' if main['flow_hhi'] < C.DOJ_UNCONCENTRATED else '≥DOJ线'}")
    n_bg = int(use["gini_betweenness"].notna().sum())
    print(f"    结构 Gini   {use['gini_betweenness'].mean():.4f}"
          f"   ← {n_bg}/{len(use)} 窗"
          f"{'（异常窗采样退化记 NaN，分母与上面三行不同）' if n_bg < len(use) else ''}")
    print(f"    精英集(top{C.ELITE_TOP_K})流量份额  "
          f"{use['elite_share'].mean():.4f}")
    n_contam = int(df["retention_contaminated"].sum()) - 1   # 减去首窗
    print(f"    精英集周际留存            "
          f"{df['top_k_retention'].mean():.4f}"
          f"（剔除受异常窗污染的 {n_contam} 对后 {ret_clean.mean():.4f}）")
    if n_contam:
        print(f"      ★ 异常窗污染两个留存值：W{outliers[0] + 1} 自身与其下一周，"
              f"仅剔单行不够")
    main["elite_share"] = use["elite_share"].mean()
    main["top_k_retention"] = float(ret_clean.mean())

    # ---------- 敏感性 ----------
    cm.banner("敏感性（每一行只改一个选择）", "-")
    print(f"    {'选择':<26}{'Gini':>10}{'top1%':>9}{'HHI':>10}")

    def line(nm, d, gk, tk, hk):
        print(f"    {nm:<26}{d[gk].mean():>10.4f}"
              f"{d[tk].mean() * 100:>8.1f}%{d[hk].mean():>10.1f}")

    line("主口径（出流·全窗）", df, "gini_out", "top1_out", "hhi_out")
    line("方向 → 接收方入流", df, "gini_in", "top1_in", "hhi_in")
    line("窗集 → 剔除异常窗", clean, "gini_out", "top1_out", "hhi_out")
    full_out = tx.groupby("from")["amt"].sum().values
    full_in = tx.groupby("to")["amt"].sum().values
    print(f"    {'时间 → 全窗汇总一次':<26}{cm.gini(full_out):>10.4f}"
          f"{cm.top_share(full_out, frac=0.01) * 100:>8.1f}%"
          f"{cm.hhi(full_out):>10.1f}")
    print(f"    {'  （同上·接收方）':<26}{cm.gini(full_in):>10.4f}"
          f"{cm.top_share(full_in, frac=0.01) * 100:>8.1f}%"
          f"{cm.hhi(full_in):>10.1f}")
    ratio = df["hhi_out"].mean() / max(cm.hhi(full_out), 1e-9)
    print(f"    ★ 周度 / 全窗 = {ratio:.2f}x —— 论文用周度，"
          f"方法论必须交代为什么")

    # ---------- 与 v16 对比 ----------
    cm.banner("与 v16 对比（说明修正的影响）", "-")
    print(f"    {'指标':<30}{'v16':>10}{'本次':>10}")
    print(f"    {'流量 Gini（入流·剔异常窗）':<30}"
          f"{'0.976' if token == 'LINK' else '0.961':>10}"
          f"{clean['gini_in'].mean():>10.4f}")
    print(f"    {'top1%（v16 实现·入流·剔异常）':<30}"
          f"{'82.3%' if token == 'LINK' else '72.1%':>10}"
          f"{clean['top1_in_v16'].mean() * 100:>9.1f}%")
    print(f"    {'top1%（本仓库实现）':<30}{'—':>10}"
          f"{clean['top1_in'].mean() * 100:>9.1f}%")
    print(f"    {'Flow HHI（出流·全窗）':<30}"
          f"{'421' if token == 'LINK' else '382':>10}"
          f"{df['hhi_out'].mean():>10.1f}")
    print(f"    {f'精英集(top{C.ELITE_TOP_K})流量份额':<30}"
          f"{'0.748' if token == 'LINK' else '—':>10}"
          f"{clean['elite_share'].mean():>10.4f}")
    print(f"    {f'top{C.ELITE_TOP_K} 周际留存（去污染）':<30}"
          f"{'0.53' if token == 'LINK' else '0.48':>10}"
          f"{ret_clean.mean():>10.4f}")
    print(f"    ★ v16 的这两个数出自已废弃脚本且用的是错误边权，"
          f"应以本表为准")

    for k, v in main.items():
        cm.log_metric(SCRIPT, token, k, round(float(v), 4),
                      f"周度·发送方·求和边权 (elite K={C.ELITE_TOP_K})")
    return df


def main():
    for t in C.TOKENS:
        run(t)
    cm.banner("完成。下一步：python 06_concentration.py")


if __name__ == "__main__":
    main()
