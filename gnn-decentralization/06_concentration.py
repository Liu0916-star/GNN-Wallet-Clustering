# -*- coding: utf-8 -*-
"""
06_concentration.py —— 所有权层集中度 + 实体合并检验

产出附录 A 的敏感性表。四个可自由裁量的选择，每个单独变动：
    人群     全部持币地址（主口径，D2=A）/ 活跃地址集
    地址类型 全部 / 排除销毁 / 排除销毁+发行方储备
    实体解析 一地址=一参与者 / 同实体多地址合并
    指标     Gini / top1% / top10 / HHI

三个已知的关键事实，脚本会自动验算并打印：
  · UNI 排除销毁后 HHI 反而【上升】（872→949）。不是 bug 是归一化效应：
    销毁地址占 10.76%，贡献 10.76²=115.8 点；移除后分母缩小，
    剩余份额同比放大 1/0.8924，平方后 ×1.2556。(872−115.8)×1.2556=949.4
  · 剔除发行方储备后两币收敛（LINK 115 / UNI 109），
    说明账面 HHI 的巨大差距主要来自协议自持而非私人市场集中
  · 实体合并对 HHI 影响巨大（LINK 135→530，因 7 个等额储备地址被合并），
    但 Gini 与 top-k 几乎不动 —— HHI 对"谁算一个参与者"极度敏感

★ 实体合并只能做到部分归并（未标注地址保持独立），
  所以结果是【下界】，不能声称是"最不利上界"。

用法：python 06_concentration.py
"""
import numpy as np
import pandas as pd

import config as C
import common as cm

SCRIPT = "06_concentration"


def metrics(v):
    return {"n": len(v), "gini": cm.gini(v),
            "top1": cm.top_share(v, frac=0.01),
            "top10": cm.top_share(v, k=10), "hhi": cm.hhi(v)}


def _row(name, m):
    print(f"    {name:<28}{m['n']:>10,}{m['gini']:>9.4f}"
          f"{m['top1'] * 100:>7.1f}%{m['top10'] * 100:>7.1f}%{m['hhi']:>10.1f}")


def run(token, emap):
    cm.banner(f"06 · 集中度 {token}")
    tx = cm.load_transfers(token, drop_burn=False)
    bal = cm.load_balances(token)
    active = cm.active_addresses(tx)
    iss = C.ISSUER_RESERVES.get(token, frozenset())
    tot = bal["balance"].sum()

    burned = bal.loc[bal["address"].isin(C.BURN_ADDRS), "balance"].sum()
    reserve = bal.loc[bal["address"].isin(iss), "balance"].sum()
    print(f"  持币 {len(bal):,} | 活跃 {len(active):,}")
    print(f"  销毁 {burned:,.0f}（{burned / tot * 100:.2f}%） | "
          f"发行方储备 {reserve:,.0f}（{reserve / tot * 100:.2f}%）· {len(iss)} 址")

    pops = {"all_holders": bal,
            "active": bal[bal["address"].isin(active)]}
    excls = {"all": lambda d: d,
             "excl_burn": lambda d: d[~d["address"].isin(C.BURN_ADDRS)],
             "excl_burn_issuer": lambda d: d[~d["address"].isin(
                 set(C.BURN_ADDRS) | set(iss))]}

    print(f"\n  [2×3 网格]  ★ 主表 = {C.MAIN_POPULATION} × 全部持有人")
    print(f"    {'人群 / 排除':<28}{'n':>10}{'Gini':>9}{'top1%':>8}"
          f"{'top10':>8}{'HHI':>10}")
    grid = {}
    for pn, pdf in pops.items():
        for en, fn in excls.items():
            m = metrics(fn(pdf)["balance"].values)
            grid[f"{pn}|{en}"] = m
            _row(f"{pn} / {en}", m)

    # ---------- 归一化效应验算 ----------
    if burned > 0:
        base = grid["all_holders|all"]["hhi"]
        after = grid["all_holders|excl_burn"]["hhi"]
        sh = burned / tot
        pred = (base - (sh * 100) ** 2) / (1 - sh) ** 2
        print(f"\n  [验算] 排除销毁 {base:.1f} → {after:.1f}")
        print(f"    销毁份额 {sh * 100:.2f}%，贡献 {(sh * 100) ** 2:.1f} 点；"
              f"移除后分母缩小，剩余份额平方放大 {1 / (1 - sh) ** 2:.4f}")
        print(f"    预测 ({base:.1f}−{(sh * 100) ** 2:.1f})×"
              f"{1 / (1 - sh) ** 2:.4f} = {pred:.1f}  实测 {after:.1f}  "
              f"{'[OK] 归一化效应，不是 bug' if abs(pred - after) < 1 else '[X]'}")

    # ---------- 实体合并 ----------
    print(f"\n  [实体合并]（未识别地址保持独立 → 结果是下界，不是上界）")
    be = bal.assign(k=cm.entity_key(bal["address"], emap))
    for pn, pdf in pops.items():
        sub = be[be["address"].isin(pdf["address"])]
        a = cm.hhi(sub["balance"].values)
        b = cm.hhi(sub.groupby("k")["balance"].sum().values)
        g_a, g_b = cm.gini(sub["balance"].values), cm.gini(
            sub.groupby("k")["balance"].sum().values)
        print(f"    {pn:<16} HHI {a:>8.1f} → {b:>8.1f}（{b / a:.2f}x） | "
              f"Gini {g_a:.4f} → {g_b:.4f}")
        grid[f"{pn}|entity_merged"] = {"hhi_addr": a, "hhi_entity": b,
                                  "gini_addr": g_a, "gini_entity": g_b}

    # ---------- flow 侧 ----------
    print(f"\n  [flow 实体合并]")
    txn = cm.load_transfers(token, drop_burn=True)
    ha = cm.weekly_metric(txn, "from", cm.hhi)
    te = txn.assign(k=cm.entity_key(txn["from"], emap))
    he = pd.Series({int(w): cm.hhi(g.groupby("k")["amt"].sum().values)
                    for w, g in te.groupby("week")})
    print(f"    Flow HHI 地址级 {ha.mean():.1f} → 实体级 {he.mean():.1f}"
          f"（{he.mean() / ha.mean():.2f}x）")
    print(f"    实体级最大周值 {he.max():.1f}  "
          f"{'仍在' if he.max() < C.DOJ_UNCONCENTRATED else '[!] 越过'} "
          f"DOJ {C.DOJ_UNCONCENTRATED:,} 线内")
    print(f"    实体级逐周: " + " ".join(f"{x:.0f}" for x in he.sort_index()))

    # 敏感性：把发行方储备的出流也剔掉（它不是市场行为）
    if iss:
        t2 = txn[~txn["from"].isin(iss)]
        h2 = pd.Series({int(w): cm.hhi(
            g.assign(k=cm.entity_key(g["from"], emap))
             .groupby("k")["amt"].sum().values)
            for w, g in t2.groupby("week")})
        print(f"    再剔除发行方储备出流: {h2.mean():.1f}"
              f"（{'仍在' if h2.max() < C.DOJ_UNCONCENTRATED else '[!] 越过'}"
              f" DOJ 线内，最大 {h2.max():.1f}）")

    wk = txn.groupby("week")["amt"].sum()
    g = txn.groupby(["week", "from"])["amt"].sum().reset_index()
    g["sh"] = g["amt"] / g["week"].map(wk)
    osh = (g.groupby("from")["sh"].sum() / txn["week"].nunique()
           ).sort_values(ascending=False)
    cov = osh[osh.index.isin(emap)].sum()
    print(f"    映射覆盖周均出流的 {cov * 100:.1f}%"
          f"{'  [!] 偏低，合并检验意义有限' if cov < 0.30 else ''}")

    print(f"\n  [top-15 出流地址]")
    for i, (a, sh_) in enumerate(osh.head(15).items(), 1):
        print(f"    {i:>3} {a}  {sh_ * 100:>6.2f}%  {emap.get(a, '???')}")

    # ★ 合并后的实体份额排行 —— 论文要回答"最大的那家占多少"
    ent_sh = (osh.groupby(cm.entity_key(pd.Series(osh.index, index=osh.index),
                                        emap)).sum().sort_values(ascending=False))
    named = ent_sh[[not str(k).startswith("0x") for k in ent_sh.index]]
    print(f"\n  [合并后实体份额 top-10]（未识别地址保持独立，故为下界）")
    for i, (e, sh_) in enumerate(named.head(10).items(), 1):
        n_addr = sum(1 for a in osh.index if emap.get(a) == e)
        print(f"    {i:>3} {str(e):<28}{sh_ * 100:>6.2f}%  "
              f"（{n_addr} 个地址）")
    top_ent = float(named.iloc[0]) if len(named) else float("nan")
    print(f"    → 最大单一实体占周均出流 {top_ent * 100:.2f}%"
          f"，单独贡献 HHI {(top_ent * 100) ** 2:.0f} 点")

    cm.save_json({"token": token, "grid": grid,
                  "burned_pct": round(burned / tot * 100, 2),
                  "reserve_pct": round(reserve / tot * 100, 2),
                  "flow_hhi_addr": round(float(ha.mean()), 1),
                  "flow_hhi_entity": round(float(he.mean()), 1),
                  "entity_coverage_outflow": round(float(cov), 4),
                  "flow_hhi_entity_max": round(float(he.max()), 1),
                  "top_entity_outflow_share": round(top_ent, 4),
                  "top_entities": {str(k): round(float(v), 4)
                                   for k, v in named.head(10).items()},
                  "caveat": "实体合并仅覆盖可识别地址，结果为下界"},
                 f"concentration_{token}.json")
    cm.log_metric(SCRIPT, token, "balance_hhi_main",
                  round(grid["all_holders|all"]["hhi"], 1),
                  "主口径=全部持币地址")
    cm.log_metric(SCRIPT, token, "flow_hhi_entity",
                  round(float(he.mean()), 1))
    return grid


def main():
    lm = cm.load_labels()
    grids = {}
    for t in C.TOKENS:
        grids[t] = run(t, cm.build_entity_map(t, lm))

    cm.banner("两币对照（附录 A 主表）")
    print(f"  {'口径':<30}{'LINK':>10}{'UNI':>10}")
    for key in ("all_holders|all", "all_holders|excl_burn",
                "all_holders|excl_burn_issuer", "active|all"):
        a = grids.get("LINK", {}).get(key, {}).get("hhi", float("nan"))
        b = grids.get("UNI", {}).get(key, {}).get("hhi", float("nan"))
        print(f"  {key:<30}{a:>10.1f}{b:>10.1f}")
    ka = "all_holders|excl_burn_issuer"
    if "LINK" not in grids or "UNI" not in grids:
        return
    la, ua = grids["LINK"][ka]["hhi"], grids["UNI"][ka]["hhi"]
    print(f"\n  ★ 剔除发行方储备后两币收敛：{la:.1f} vs {ua:.1f}"
          f"（差 {abs(la - ua) / max(la, ua) * 100:.0f}%）")
    print(f"    → 账面 HHI 的巨大差距主要来自协议自持，不是私人市场集中")
    cm.banner("完成。下一步：python 07_governance.py")


if __name__ == "__main__":
    main()
