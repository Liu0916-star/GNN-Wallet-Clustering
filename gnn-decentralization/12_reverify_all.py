# -*- coding: utf-8 -*-
"""
12_reverify_all.py —— 全量重新核验清单

背景
----
首轮人工核验时，ERC-20 转账记录只翻了最新一页，未穷尽全部历史。凡是
【依据行为模式】做出的判定（对手方是谁、交易频率、是否套利），其证据基础
都不完整；凡是【依据 Name Tag 或已验证源码】做出的判定则不受影响，因为
那些信息在页面顶部与 Contract 标签页，与转账列表无关。

本脚本读取现有核验表，按判定依据分层，输出一份带优先级与"该看哪里"提示
的重查清单。既然决定全量重查，脚本不做抽样，列出全部地址；分层的作用是
让每个地址的复核有明确重点，而不是从头再来一遍。

★ 一致性要求
    随机对照组（11_random_control.py）的 38 个地址是用完整方法核验的。
    若经纪人一侧仍停留在浅层方法，两臂的测量强度不同，主检验会因差异性
    误分类而失效——且偏差方向对我们有利，这在审稿中比不做更糟。因此必须
    先完成本脚本的重查，再跑 11 的 score 模式。

用法
----
    python 12_reverify_all.py                # 生成清单
    python 12_reverify_all.py --diff         # 重查完成后，比对前后差异

输出
----
    verification/reverify_worklist.csv       重查工作表（人工填写 new_*）
    verification/broker_identities_v2.csv    --diff 模式生成的新版核验表
    results/reverify_diff.json               前后差异统计
"""
import sys

import numpy as np
import pandas as pd

import config as C
import common as cm

SCRIPT = "12_reverify_all"
VDIR = C.ROOT / "verification"
SRC = VDIR / "broker_identities.csv"
WORK = VDIR / "reverify_worklist.csv"
OUT = VDIR / "broker_identities_v2.csv"

# 判定依据分层：决定重查时该看页面的哪一部分
BEHAVIOURAL = ("unlabelled", "unlabeled", "mev", "unknown", "suspected",
               "compromised", "nan")


def classify(identity, evidence):
    """按【当初的判定依据】分层，而非按标签本身。

    返回 (层级, 优先级, 重查时该看哪里)
    """
    ident = str(identity).strip().lower()
    ev = str(evidence).strip().lower()
    blank = (not ident) or ident == "nan"

    # 证据里提到标签/源码/部署者的，依据与转账列表无关
    tag_based = any(k in ev for k in
                    ("name tag", "public name", "etherscan tag", "#"))
    src_based = any(k in ev for k in
                    ("verified source", "source code", "exact match",
                     "contract name", "deployer"))

    if blank or ident == "unknown":
        return ("A. 未定性", 1,
                "顶部 Name Tag → Contract 标签页 → 完整 ERC-20 转账")
    if ident.startswith(BEHAVIOURAL):
        return ("B. 行为模式判定", 2,
                "完整 ERC-20 转账（对手方、频率、路径）→ 顶部 Name Tag 复核")
    if tag_based or src_based:
        return ("C. 标签/源码判定", 3,
                "复核 Name Tag 与 Contract Creator（转账页不影响此判定）")
    return ("D. 具名但依据不详", 2,
            "确认依据：Name Tag？已验证源码？还是行为推断？")


def main():
    if "--diff" in sys.argv:
        return do_diff()

    cm.banner("12 · 全量重新核验清单")
    if not SRC.exists():
        print(f"  [X] 未找到 {SRC}")
        return
    vb = pd.read_csv(SRC)
    vb.columns = [c.strip().lstrip("\ufeff") for c in vb.columns]
    acol = next(c for c in vb.columns if "addr" in c.lower())
    icol = next((c for c in vb.columns if c.lower() in
                 ("verified_identity", "identity", "name", "label")), None)
    ecol = next((c for c in vb.columns if "evidence" in c.lower()), None)
    vb[acol] = vb[acol].astype(str).str.strip().str.lower()
    print(f"  核验表 {len(vb)} 条")

    # 当前经纪人集合，用于标注哪些地址仍在使用
    live = {}
    for t in C.TOKENS:
        p = C.RESULTS / f"hidden_brokers_{t}.csv"
        if p.exists():
            d = pd.read_csv(p)
            d["address"] = d["address"].astype(str).str.strip().str.lower()
            for _, r in d.iterrows():
                live.setdefault(r["address"], {})[t] = {
                    "counterparties": r.get("counterparties", np.nan),
                    "seed_frequency": r.get("seed_frequency", np.nan)}

    rows = []
    for _, r in vb.iterrows():
        a = r[acol]
        ident = r[icol] if icol else ""
        ev = r[ecol] if ecol else ""
        tier, prio, where = classify(ident, ev)
        inv = live.get(a, {})
        rows.append({
            "tier": tier, "priority": prio, "address": a,
            "in_tokens": "+".join(sorted(inv)) or "NEITHER",
            "counterparties": max(
                [v["counterparties"] for v in inv.values()], default=np.nan),
            "seed_frequency": max(
                [v["seed_frequency"] for v in inv.values()], default=np.nan),
            "old_identity": str(ident).strip(),
            "old_evidence": str(ev).strip()[:120],
            "where_to_look": where,
            "new_identity": "",       # ← 人工填
            "new_evidence": "",       # ← 人工填
            "erc20_transfer_count": "",   # ← 人工填：完整转账笔数
        })

    df = pd.DataFrame(rows).sort_values(
        ["priority", "in_tokens", "counterparties"],
        ascending=[True, False, False])
    df.to_csv(WORK, index=False)

    cm.banner("分层结果", "-")
    for tier, g in df.groupby("tier", sort=True):
        n_live = int((g["in_tokens"] != "NEITHER").sum())
        print(f"\n  {tier}  共 {len(g)} 条（其中 {n_live} 条仍在当前经纪人集合）")
        print(f"    重查重点: {g['where_to_look'].iloc[0]}")
        if tier.startswith("C"):
            print(f"    ★ 此层的判定依据与转账列表无关，复核只需确认"
                  f"标签与部署者仍然成立")
        for r in g.head(6).itertuples():
            cp = f"{r.counterparties:.0f}" if pd.notna(r.counterparties) else "—"
            print(f"      {r.address}  {r.in_tokens:<9}cp={cp:<5}"
                  f"{r.old_identity[:34]}")
        if len(g) > 6:
            print(f"      … 其余 {len(g) - 6} 条见 {WORK.name}")

    cm.banner("填写说明", "-")
    print(f"  → {WORK.name}  {len(df)} 条")
    print(f"""
  三列需要填写：
    new_identity           判定结果，四选一：
                             · 具体协议名（有 Name Tag 或已验证源码+部署者）
                             · Unlabelled routing contract
                             · MEV Bot
                             · Unknown
    new_evidence           依据，一句话。务必写明是"标签/源码"还是"行为推断"
    erc20_transfer_count   完整 ERC-20 转账笔数（不是 Transactions 页的数）。
                           Etherscan 对超过一万笔的地址只显示
                           "More than 10,000"，此时如实填 >10000 即可；
                           这一列用于自证核验深度，不参与任何计算。

  ★ 只有【具体协议名】计入论文的"具名协议"比例，后三者等价。
    因此 Unknown → Unlabelled routing contract 这类修正不改变论文数字，
    只提升 evidence 质量；真正会改变数字的是 Unknown → 具体协议名。

  ★ 判准从严，与随机对照组保持一致：看着像路由器但查不出归属的，
    一律 Unlabelled routing contract。宁可低估。

  填完执行：python 12_reverify_all.py --diff""")
    cm.log_metric(SCRIPT, "BOTH", "n_reverify", len(df))


def do_diff():
    cm.banner("12 · 重查前后差异")
    if not WORK.exists():
        print(f"  [X] 未找到 {WORK}")
        return
    d = pd.read_csv(WORK)
    filled = d["new_identity"].astype(str).str.strip().replace("nan", "")
    n_blank = int((filled == "").sum())
    if n_blank:
        print(f"  [!] {n_blank}/{len(d)} 条尚未填写，差异统计不完整")

    GEN = ("unlabelled", "unlabeled", "unknown", "suspected", "mev",
           "fake_phishing", "compromised", "nan")

    def named(v):
        t = str(v).strip().lower()
        return bool(t) and t != "nan" and not t.startswith(GEN)

    d["old_named"] = d["old_identity"].map(named)
    d["new_named"] = filled.map(named)
    chg = d[(filled != "") & (d["old_identity"].astype(str).str.strip()
                              != filled)]

    print(f"\n  填写 {len(d) - n_blank}/{len(d)} 条，其中 {len(chg)} 条判定改变")
    up = d[(~d["old_named"]) & (d["new_named"])]
    down = d[(d["old_named"]) & (~d["new_named"]) & (filled != "")]
    print(f"    未识别 → 具名协议: {len(up)}  ← 这会改变论文数字")
    print(f"    具名协议 → 未识别: {len(down)} ← 这也会")
    for r in pd.concat([up, down]).itertuples():
        print(f"      {r.address}  {str(r.old_identity)[:26]:<28}"
              f"→ {str(r.new_identity)[:30]}")

    # 转账笔数：报告分布与截断比例，并提示该列含非数值项
    cnt = d["erc20_transfer_count"].astype(str).str.strip()
    trunc = cnt.str.startswith(">").sum()
    num = pd.to_numeric(cnt, errors="coerce")
    n_num = int(num.notna().sum())
    if n_num or trunc:
        print(f"\n  [核验深度] ERC-20 转账笔数")
        print(f"    可读数值 {n_num} 条，中位 "
              f"{num.median():.0f}" if n_num else "", end="")
        print(f" | 截断为 >10000 的 {trunc} 条")
        print(f"    ★ 该列含非数值项（Etherscan 对万笔以上只显示上限），"
              f"仅作核验深度的凭据，不可直接做数值统计。")

    live = d[d["in_tokens"] != "NEITHER"]
    for t in C.TOKENS:
        sub = live[live["in_tokens"].str.contains(t, na=False)]
        if not len(sub):
            continue
        o = int(sub["old_named"].sum())
        n = int(sub["new_named"].sum())
        print(f"\n  {t} 经纪人具名比例: {o}/{len(sub)} = {o/len(sub):.0%}"
              f"  →  {n}/{len(sub)} = {n/len(sub):.0%}")

    # 生成新版核验表
    out = d[["in_tokens", "address", "new_identity", "new_evidence",
             "erc20_transfer_count"]].copy()
    out.columns = ["token", "address", "verified_identity", "evidence",
                   "erc20_transfers"]
    out = out[out["verified_identity"].astype(str).str.strip().ne("")]
    out.to_csv(OUT, index=False)
    print(f"\n  → {OUT.name}（{len(out)} 条）")
    print(f"    确认无误后覆盖 broker_identities.csv，再依次重跑：")
    print(f"      python 05_c3_roles.py")
    print(f"      python 11_random_control.py score")

    cm.save_json({"n_total": len(d), "n_filled": int(len(d) - n_blank),
                  "n_transfer_count_truncated": int(trunc),
                  "transfer_count_note": "Etherscan caps the displayed "
                                         "ERC-20 transfer count at 10,000; "
                                         "entries recorded as >10000 are "
                                         "censored from above",
                  "n_changed": len(chg),
                  "unnamed_to_named": len(up),
                  "named_to_unnamed": len(down),
                  "changed": [{"address": r.address,
                               "old": str(r.old_identity),
                               "new": str(r.new_identity)}
                              for r in chg.itertuples()]},
                 "reverify_diff.json")
    cm.banner("完成")


if __name__ == "__main__":
    main()
