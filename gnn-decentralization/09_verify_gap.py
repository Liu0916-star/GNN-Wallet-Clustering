# -*- coding: utf-8 -*-
"""
09_verify_gap.py —— 核实缺口盘点

经纪人集合在重构后发生了实质变化（LINK 36→30、UNI 29→18、跨币 15→12），
需要在定稿前搞清两件事：

  A. 【待核实】新集合里哪些地址不在 verification/broker_identities.csv 中
     → 这些要上 Etherscan 逐个查，补进核验表

  B. 【已掉出】旧核验表里哪些地址不再入选，以及是被哪一道过滤剔掉的
     → 如果一个已知路由器是被"对手方≥5"或"介数>0"剔掉的，
       说明下限设得过严，需要重新斟酌 config 里的阈值

输出 results/verify_gap.csv，可直接照着它去 Etherscan 逐条查。

用法：python 09_verify_gap.py
"""
import numpy as np
import pandas as pd

import config as C
import common as cm

SCRIPT = "09_verify_gap"


def ensemble_score(token, scale="1w"):
    """用 05 缓存的多种子嵌入重算排名集成分。不训练，只做逻辑回归。

    ★ 没有这个分数就无法判断一个地址是"被下限剔掉"还是"本来就没进 top1%"。
      前者说明阈值过严需要调，后者说明它本就不是核心节点 —— 结论完全相反。
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import rankdata
    d = np.load(C.RESULTS / f"graph_{token}_{scale}.npz", allow_pickle=True)
    y = cm.is_core_mask(d["cat"]).astype(int)
    scores = []
    for sd in C.BROKER_CONSENSUS_SEEDS:
        f = C.RESULTS / f"emb_{token}_{scale}_s{sd}.npy"
        if not f.exists():
            continue
        e = StandardScaler().fit_transform(np.load(f))
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(e, y)
        scores.append(clf.predict_proba(e)[:, 1])
    if not scores:
        return None, 0
    n = len(scores[0])
    r = np.mean([rankdata(x) / n for x in scores], axis=0)
    return r, len(scores)


def load_verified():
    vp = C.ROOT / "verification" / "broker_identities.csv"
    if not vp.exists():
        print(f"  [!] 未找到 {vp}")
        return {}
    vb = pd.read_csv(vp)
    vb.columns = [c.strip().lstrip("\ufeff") for c in vb.columns]
    acol = next(c for c in vb.columns if "addr" in c.lower())
    icol = next((c for c in vb.columns
                 if c.lower() in ("verified_identity", "identity", "name",
                                  "label")), None)
    return {str(a).strip().lower(): (str(vb[icol].iloc[i]) if icol else "?")
            for i, a in enumerate(vb[acol])}


GENERIC = ("unlabelled", "unlabeled", "unknown", "suspected", "mev",
           "fake_phishing", "compromised")


def main():
    cm.banner("09 · 核实缺口盘点")
    ident = load_verified()
    print(f"  核验表 {len(ident)} 条")

    sets, feats = {}, {}
    for t in C.TOKENS:
        bp = C.RESULTS / f"hidden_brokers_{t}.csv"
        gp = C.RESULTS / f"graph_{t}_1w.npz"
        if not (bp.exists() and gp.exists()):
            print(f"  [X] 缺 {bp.name} 或 {gp.name}")
            return
        df = pd.read_csv(bp)
        df["address"] = df["address"].astype(str).str.lower()
        sets[t] = df
        d = np.load(gp, allow_pickle=True)
        nodes = [str(x).lower() for x in d["nodes"]]
        F = C.FEATURE_NAMES
        sc, n_sc = ensemble_score(t)
        if sc is None:
            print(f"  [!] {t} 缺 emb_*_s*.npy，无法区分"
                  f"「被下限剔除」与「未进 top1%」，先跑 05_c3_roles.py")
            sc = np.full(len(nodes), np.nan)
        else:
            print(f"  {t}: 用 {n_sc} 份缓存嵌入重算集成分")
        bal_v = d["feat"][:, F.index("balance")]
        feats[t] = pd.DataFrame({
            "address": nodes,
            "balance": bal_v,
            "counterparties": d["feat"][:, F.index("counterparties")],
            "betweenness": d["feat"][:, F.index("betweenness")],
            "cat": d["cat"],
            "score_pct": sc,
            "in_top1": sc >= np.nanpercentile(sc, C.BROKER_PROB_PCT)
                       if not np.all(np.isnan(sc)) else False,
            "bal_ok": bal_v <= np.median(bal_v),
        }).set_index("address")

    union = set(sets[C.TOKENS[0]]["address"]) | set(sets[C.TOKENS[1]]["address"])
    both = set(sets[C.TOKENS[0]]["address"]) & set(sets[C.TOKENS[1]]["address"])
    print(f"  新集合并集 {len(union)} 个（{C.TOKENS[0]} "
          f"{len(sets[C.TOKENS[0]])} + {C.TOKENS[1]} "
          f"{len(sets[C.TOKENS[1]])} − 交集 {len(both)}）")

    # ---------- A. 待核实 ----------
    todo = sorted(union - set(ident))
    cm.banner(f"A. 待核实：{len(todo)} 个新地址不在核验表中", "-")
    rows = []
    for a in todo:
        where = "+".join(t for t in C.TOKENS
                         if a in set(sets[t]["address"]))
        # ★ 跨币地址要取两币的最大值，只看第一个代币会低估其重要性
        vals = {"seed_frequency": [], "counterparties": [], "betweenness": []}
        for t in C.TOKENS:
            sub = sets[t][sets[t]["address"] == a]
            if not len(sub):
                continue
            row = sub.iloc[0]
            for k in vals:
                if k in row.index and pd.notna(row[k]):
                    vals[k].append(float(row[k]))
        rows.append({"address": a, "in_tokens": where,
                     "seed_frequency": max(vals["seed_frequency"], default=np.nan),
                     "counterparties": max(vals["counterparties"], default=np.nan),
                     "betweenness": max(vals["betweenness"], default=np.nan),
                     "status": "TODO_VERIFY"})
    if rows:
        td = pd.DataFrame(rows)
        td["n_tokens"] = td["in_tokens"].str.count(r"\+") + 1
        td = td.sort_values(["n_tokens", "counterparties"],
                            ascending=[False, False])
        print(f"    {'address':<44}{'代币':<10}{'出现率':>7}{'对手方':>8}{'介数':>11}")
        for r in td.itertuples():
            fq = f"{r.seed_frequency:.0%}" if pd.notna(r.seed_frequency) else "—"
            cpv = f"{r.counterparties:.0f}" if pd.notna(r.counterparties) else "—"
            bt = f"{r.betweenness:.6f}" if pd.notna(r.betweenness) else "—"
            print(f"    {r.address:<44}{r.in_tokens:<10}"
                  f"{fq:>7}{cpv:>8}{bt:>11}")
        print(f"\n    ★ 优先查跨币的与对手方多的 —— 它们对结论影响最大")
    else:
        td = pd.DataFrame()
        print("    无（新集合全部已核验）")

    # ---------- B. 已掉出 ----------
    dropped = sorted(set(ident) - union)
    cm.banner(f"B. 已掉出：{len(dropped)} 个旧核验地址不再入选", "-")
    rows2 = []
    n_by_threshold = 0
    for a in dropped:
        det, hit_thr = [], False
        for t in C.TOKENS:
            if a not in feats[t].index:
                det.append(f"{t}:不在图中")
                continue
            f = feats[t].loc[a]
            if isinstance(f, pd.DataFrame):
                f = f.iloc[0]
            # ★ 先看它是否进了 top1% —— 这决定后面的原因怎么解读
            if not bool(f["in_top1"]):
                det.append(f"{t}:未进 top1%"
                           f"(分位{f['score_pct']:.3f})")
                continue
            r = []
            if f["cat"] != "Unlabeled":
                r.append(f"已有标签({f['cat']})")
            if not bool(f["bal_ok"]):
                r.append("余额>中位")
            if f["counterparties"] < C.BROKER_MIN_COUNTERPARTIES:
                r.append(f"★对手方{int(f['counterparties'])}<"
                         f"{C.BROKER_MIN_COUNTERPARTIES}")
                hit_thr = True
            if C.BROKER_REQUIRE_POSITIVE_BETWEENNESS and f["betweenness"] <= 0:
                r.append("★介数=0")
                hit_thr = True
            det.append(f"{t}:在top1%但被剔[{'/'.join(r) if r else '?'}]")
        n_by_threshold += int(hit_thr)
        rows2.append({"address": a, "identity": ident[a],
                      "reason": " | ".join(det),
                      "dropped_by_threshold": hit_thr, "status": "DROPPED"})
    if rows2:
        dd = pd.DataFrame(rows2)
        hard = dd[~dd["identity"].str.lower().str.startswith(GENERIC)]
        print(f"    其中 {len(hard)} 个是有具体身份的已知聚合器：")
        for r in hard.itertuples():
            print(f"    {r.address}")
            print(f"      {r.identity[:60]}")
            print(f"      {r.reason}")
        if len(dd) > len(hard):
            print(f"\n    另有 {len(dd) - len(hard)} 个是 Unlabelled/MEV 等"
                  f"通用条目，掉出不影响论证")
        hard_thr = int(hard["dropped_by_threshold"].sum()) if len(hard) else 0
        print(f"\n    掉出原因分解（★ 标记 = 进了 top1% 但被下限剔除）")
        print(f"      有具体身份且【被下限剔除】: {hard_thr}")
        print(f"      有具体身份但【本就没进 top1%】: {len(hard) - hard_thr}")
        if hard_thr:
            print(f"\n    [!] {hard_thr} 个已知聚合器进了 top1% 却被下限剔掉。")
            print(f"        这说明 config 的阈值可能过严，须重新斟酌：")
            print(f"        BROKER_MIN_COUNTERPARTIES="
                  f"{C.BROKER_MIN_COUNTERPARTIES}, "
                  f"BROKER_REQUIRE_POSITIVE_BETWEENNESS="
                  f"{C.BROKER_REQUIRE_POSITIVE_BETWEENNESS}")
        else:
            print(f"\n    [OK] 没有已知聚合器是被下限误杀的 —— 阈值设置合理，")
            print(f"         掉出者都是本就未进 top1% 的节点。")
    else:
        dd = pd.DataFrame()
        print("    无（旧核验地址全部仍在集合中）")

    out = pd.concat([td, dd], ignore_index=True) if len(td) or len(dd) \
        else pd.DataFrame()
    if len(out):
        cm.save_csv(out, "verify_gap.csv")
    cm.log_metric(SCRIPT, "BOTH", "n_todo_verify", len(todo))
    cm.log_metric(SCRIPT, "BOTH", "n_dropped", len(dropped))

    cm.banner("下一步：照 A 段清单上 Etherscan 逐个查（看 ERC-20 Token "
              "Transfers 标签页，不是 Transactions），补进 "
              "verification/broker_identities.csv 后重跑 05")


if __name__ == "__main__":
    main()
