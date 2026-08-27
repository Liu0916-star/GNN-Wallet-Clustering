# -*- coding: utf-8 -*-
"""
07_governance.py —— 规则层：把 C2 从两层扩到三层

    所有权层  谁拥有        Gini + 余额 HHI       （06_concentration.py）
    路由层    谁在做生意    flow HHI + 经纪人     （03/05）
    规则层    谁能改规则    Nakamoto 系数         ← 本脚本

为什么规则层用 Nakamoto 而不是 HHI：治理是【阈值博弈】——够 40M 就能达到
法定人数，分布形状不重要。Nakamoto 直接回答"最少几个地址能改规则"。
它是 2017 年就确立的指标，不用造词，也不用论证原创性。
因此不需要给这个概念命名，functional decentralization 的术语死结自动解开。

三个产出：
  1. Nakamoto 系数（三种口径，与 HHI 用同一套敏感性方法论）
  2. 三层错位列联表（提案资格地址 × 隐藏经纪人，实测零重叠）
  3. 可撤销份额上下界 —— 路由流量中有多大比例受该协议治理支配

★ 必须写进论文的三条限定：
  · 持币 ≠ 投票权，需委托激活。这里测的是 potential voting power 的【上界】。
    Uniswap 基金会自承大额委托人参与率常低于 50%，实际决策权只会更集中，
    所以这个上界是保守的。
  · 托管地址（交易所热钱包）不代表持有人意志，单列一档。
  · 不声称这是"检测器"。n=2 无法验证分类能力，只作单协议的三层描述。

用法：python 07_governance.py [TOKEN]
"""
import sys

import numpy as np
import pandas as pd

import config as C
import common as cm

SCRIPT = "07_governance"


def is_custodial(entity: str) -> bool:
    e = str(entity).lower()
    return any(h in e for h in C.GOV_CUSTODIAL_HINTS)


def run(token):
    cm.banner(f"07 · 规则层 {token}")
    if token != "UNI":
        print(f"  {token} 没有代币加权治理（预言机节点由 Chainlink Labs 准入），")
        print(f"  规则层不适用。这本身是对照结论：路由层竞争充分的两个协议里，")
        print(f"  只有一个存在可撤销的规则层。写进正文即可，无需跑数。")
        cm.save_json({"token": token, "applicable": False,
                      "reason": "no token-weighted governance"},
                     f"governance_{token}.json")
        return

    bal = cm.load_balances(token)
    tx = cm.load_transfers(token, drop_burn=True)
    emap = cm.build_entity_map(token, verbose=False)
    tot = bal["balance"].sum()

    # ---------- 1. Nakamoto 系数 ----------
    cm.banner("1. Nakamoto 系数（达到法定人数所需的最少地址数）", "-")
    print(f"  提案门槛 {C.GOV_PROPOSAL_THRESHOLD:,} UNI "
          f"({C.GOV_PROPOSAL_THRESHOLD / tot * 100:.2f}%) | "
          f"法定人数 {C.GOV_QUORUM:,} ({C.GOV_QUORUM / tot * 100:.2f}%)")

    cohorts = {}
    cohorts["All holders"] = bal
    cohorts["Excl. non-voting"] = bal[~bal["address"].isin(C.GOV_NON_VOTING)]
    ent = bal.assign(k=cm.entity_key(bal["address"], emap))
    cust = ent["k"].map(is_custodial).fillna(False)
    cohorts["Excl. custodial"] = ent[~ent["address"].isin(C.GOV_NON_VOTING)
                                    & ~cust].drop(columns="k")

    # ★ 只保留语义明确的列。
    #   Nakamoto 系数 = "最少几个地址【合计】即可达到法定人数"。
    #   v16 的表里还有一列"达提案门槛"，用的是同一个函数但阈值换成 2.5M —— 
    #   它恒等于 1（最大持有人一个就超过 2.5M），与右边"≥门槛地址数"
    #   （有几个地址【各自】达到门槛）语义完全不同，极易误读，已删除。
    print(f"\n  {'口径':<20}{'n':>10}"
          f"{'Nakamoto(quorum)':>18}{'≥门槛地址数':>13}{'其合计份额':>11}")
    rec = {}
    for nm, d in cohorts.items():
        v = d["balance"].values
        nk_q = cm.nakamoto_absolute(v, C.GOV_QUORUM)
        elig = d[d["balance"] >= C.GOV_PROPOSAL_THRESHOLD]
        share = elig["balance"].sum() / tot
        rec[nm] = {"n": len(d), "nakamoto_quorum": nk_q,
                   "nakamoto_quorum_reachable": bool(nk_q > 0),
                   "n_eligible_each_above_threshold": len(elig),
                   "eligible_share": round(float(share), 4)}
        nk_txt = "不可达" if nk_q < 0 else f"{nk_q}"
        print(f"  {nm:<20}{len(d):>10,}{nk_txt:>18}"
              f"{len(elig):>13,}{share * 100:>10.1f}%")
    print(f"  Nakamoto(quorum) = 最少几个地址【合计】达到 "
          f"{C.GOV_QUORUM:,} UNI；")
    print(f"  ≥门槛地址数 = 有几个地址【各自】持有 ≥"
          f"{C.GOV_PROPOSAL_THRESHOLD:,} UNI。两者语义不同，勿混。")

    print(f"\n  ★ 与 HHI 同构：Nakamoto 系数同样高度依赖口径。")
    print(f"    v16 曾称「单个地址即可满足法定人数」—— 那两个地址经核验是")
    print(f"    UNI Timelock（治理合约本身）与销毁地址，都不可能投票，该说法已撤销。")

    # ---------- 2. 三层错位 ----------
    cm.banner("2. 三层错位（规则层 × 路由层）", "-")
    elig = set(bal[bal["balance"] >= C.GOV_PROPOSAL_THRESHOLD]["address"])
    elig_v = elig - set(C.GOV_NON_VOTING)
    movers = set(tx["from"]) | set(tx["to"])
    inactive = len(elig_v - movers)
    print(f"    提案资格地址（可投票） {len(elig_v):>6}")
    print(f"    其中 90 天完全不转账   {inactive:>6}"
          f"（{inactive / max(len(elig_v), 1) * 100:.0f}%）")

    # ---- 2a. 主检验：规则层 × 路由层（按出流量定义，无余额约束）----
    # ★ 不能用"隐藏经纪人"做这个检验：经纪人的定义里含 balance ≤ 中位数，
    #   而提案资格要求 balance ≥ 2.5M UNI —— 两者交集在【定义上】就必然为空。
    #   v16 的"零重叠"是循环论证，不是实证发现。
    #   这里改用【周均出流 top-N】定义路由层，它对余额没有任何约束，
    #   因此重叠与否是真正的经验问题。
    wk_ = tx.groupby("week")["amt"].sum()
    g_ = tx.groupby(["week", "from"])["amt"].sum().reset_index()
    g_["sh"] = g_["amt"] / g_["week"].map(wk_)
    outflow = (g_.groupby("from")["sh"].sum()
               / tx["week"].nunique()).sort_values(ascending=False)
    bmap = dict(zip(bal["address"], bal["balance"]))
    print(f"\n    [主检验] 规则层 × 路由层（路由层 = 周均出流 top-N，"
          f"对余额无约束）")
    print(f"    {'N':>6}{'重叠':>8}{'占比':>9}{'该组合计出流':>14}")
    strat = {}
    for N in (10, 25, 50, 100):
        top = set(outflow.head(N).index)
        ov_ = top & elig_v
        strat[f"top{N}_outflow"] = {"overlap": len(ov_),
                                    "share": round(float(
                                        outflow.head(N).sum()), 4)}
        print(f"    {N:>6}{len(ov_):>8}{len(ov_) / N:>8.0%}"
              f"{outflow.head(N).sum() * 100:>13.1f}%")
    top50 = list(outflow.head(50).index)
    n_over = sum(1 for a in top50
                 if bmap.get(a, 0.0) >= C.GOV_PROPOSAL_THRESHOLD)
    med_bal_top50 = float(np.median([bmap.get(a, 0.0) for a in top50]))
    print(f"    路由层 top-50 的中位余额 {med_bal_top50:,.0f} UNI"
          f"（提案门槛 {C.GOV_PROPOSAL_THRESHOLD:,}）")
    print(f"    其中达到提案门槛的 {n_over}/50")
    if n_over == 0:
        print(f"    → 运营网络的主力全部不具备提案资格：规则层与路由层错位成立，")
        print(f"      且此结论【不依赖】任何余额筛选，非循环论证。")

    # ---- 2b. 参考：与隐藏经纪人的重叠（循环，仅存档不作证据）----
    bp = C.RESULTS / f"hidden_brokers_{token}.csv"
    if bp.exists():
        brokers = set(pd.read_csv(bp)["address"].str.lower())
        ov = elig_v & brokers
        print(f"\n    [参考·不可作证据] 提案资格 × 隐藏经纪人 = {len(ov)}")
        print(f"    经纪人定义含 balance ≤ 中位数，与提案门槛 "
              f"≥{C.GOV_PROPOSAL_THRESHOLD:,} 互斥，")
        print(f"    交集为 0 是定义强制的。v16 用它论证三层错位属循环论证，须删。")
        strat["brokers_overlap_circular"] = len(ov)
        strat["n_brokers"] = len(brokers)

    strat.update({"n_eligible_voting": len(elig_v),
                  "eligible_inactive": inactive,
                  "routing_top50_median_balance": med_bal_top50,
                  "routing_top50_above_threshold": n_over})
    rec["stratification"] = strat

    # ---------- 3. 可撤销份额 ----------
    cm.banner("3. 可撤销份额（路由流量中受该协议治理支配的比例）", "-")
    osh = outflow            # 复用上一节算好的周均出流份额

    # 三档，粒度由严到宽：
    #   L1 治理合约直接控制（Timelock / Treasury）—— 无争议
    #   L2 协议自有合约（Labs 部署的路由器等）—— 多数不可升级，
    #      治理只能改池子参数，不能改路由器逻辑，故只作中间档
    #   L3 未识别地址 —— 不确定区间
    gov_direct = set(C.ISSUER_RESERVES.get(token, ()))
    l1 = float(osh[osh.index.isin(gov_direct)].sum())
    l2 = float(osh[[str(emap.get(a, "")).lower().startswith("uniswap")
                    and a not in gov_direct for a in osh.index]].sum())
    unknown = float(osh[~osh.index.isin(emap)].sum())
    print(f"    L1 治理合约直接控制（Timelock/Treasury）  {l1 * 100:>6.2f}%")
    print(f"    L2 协议自有合约（Labs 部署，多数不可升级） {l2 * 100:>6.2f}%")
    print(f"    L3 未识别地址（不确定区间）               {unknown * 100:>6.2f}%")
    lower = l1
    print(f"    → 下界 = L1 = {lower * 100:.2f}%  |  "
          f"上界 ≤ L1+L2+L3 = {(l1 + l2 + unknown) * 100:.2f}%")
    print(f"\n    ★ 报区间而非点估计。这堵住一个必然的反驳：")
    print(f"      即使 1inch / 0x 这些聚合器不归 Uniswap 管，它们路由的目的地")
    print(f"      大量是 Uniswap 池 —— 费用开关一开，经过池子的流量都受影响。")
    print(f"      上界应理解为「最终结算于治理可调参数池」的流量。")
    rec["revocable_share"] = {"L1_governance_direct": round(l1, 4),
                              "L2_protocol_owned": round(l2, 4),
                              "L3_unidentified": round(unknown, 4),
                              "lower": round(l1, 4),
                              "upper_bound": round(l1 + l2 + unknown, 4)}

    rec["caveats"] = [
        "持币≠投票权，需委托激活；此处为 potential voting power 的上界",
        "Uniswap 基金会自承大额委托人参与率常低于 50%，实际决策权更集中",
        "托管地址不代表持有人意志，已单列一档",
        "n=2，不作为检测器主张，仅为单协议的三层描述",
    ]
    cm.save_json({"token": token, "applicable": True,
                  "proposal_threshold": C.GOV_PROPOSAL_THRESHOLD,
                  "quorum": C.GOV_QUORUM, **rec},
                 f"governance_{token}.json")
    for nm, d in rec.items():
        if isinstance(d, dict) and "nakamoto_quorum" in d:
            cm.log_metric(SCRIPT, token, f"nakamoto_quorum[{nm}]",
                          d["nakamoto_quorum"])


def main():
    a = sys.argv[1:]
    for t in (a if a else C.TOKENS):
        run(t)
    cm.banner("完成。下一步：python 08_figures.py")


if __name__ == "__main__":
    main()
