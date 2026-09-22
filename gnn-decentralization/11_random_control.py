# -*- coding: utf-8 -*-
"""
11_random_control.py —— 随机对照组：排除隐藏经纪人核验的循环偏倚

问题
----
论文第 5.5 节报告"经纪人中 33%（LINK）/ 39%（UNI）可核验为具名路由协议"。
但这 36 个地址【全部是 GNN 选出来的】，我们只查了它选中的那些。审稿人必然
会问：你怎么知道这个比例高于随便挑的？承认偏倚不等于排除偏倚。

设计
----
从【同一资格池】（无标签 & 对手方 ≥5 & 介数 >0 & 余额 ≤中位）中抽取未被
GNN 选中的地址作对照，人工核验后比较具名路由协议的占比。

★ 两个必须避开的陷阱：

  1. 朴素均匀抽样会使检验失去意义。经纪人按分数选出，而分数与对手方数
     强相关，均匀抽样得到的对照组度数远低于经纪人组，于是"经纪人更像
     路由器"退化为"度数高的节点是路由器"——与 GNN 无关的同义反复。
     因此主检验采用【按对手方数配对】的对照组：为每个经纪人匹配一个
     对手方数最接近的非经纪人。两组度数分布对齐，差异只能归因于嵌入。
     朴素随机组一并报告，作为对照的对照。

  2. 核验时若知道谁是经纪人，判断会被无意识影响。因此输出一份打乱顺序、
     不含分组标签的盲测清单，分组信息另存于 key 文件，核验完成前不要打开。

用法
----
    python 11_random_control.py sample          # 抽样并生成盲测清单
    #  → 人工在 Etherscan 逐个核验，填写 control_blind.csv 的两列
    python 11_random_control.py score           # 揭盲、统计、Fisher 检验

输出
----
    verification/control_blind.csv   盲测清单（人工填写）
    verification/control_key.csv     分组答案（核验前勿看）
    results/random_control.json      检验结果
"""
import sys

import numpy as np
import pandas as pd

import config as C
import common as cm

SCRIPT = "11_random_control"
VDIR = C.ROOT / "verification"
BLIND = VDIR / "control_blind.csv"
KEY = VDIR / "control_key.csv"

# ---------------------------------------------------------------- 抽样规模
# 核验是人工的，清单大小直接决定这个实验做不做得完。按功效倒推：
# Fisher 单侧检验，效应量 35% vs 5% 时每组 n=20 即可达 p<0.05（n=15 不够）；
# 若对照命中率高到 15%，则需每组 n=40，那已超出一晚能完成的量。
# 因此预先固定 n=20 并如实报告结果，不做"看了结果再加样本"的事后扩容。
CONTROL_TOKENS = ("LINK",)     # 主账本。两个都做则核验量翻倍
N_MATCHED = 20                 # 度数配对对照 —— 主检验
N_UNIFORM = 10                 # 朴素随机对照 —— 说明度数混杂，非主检验
N_BROKER_ANCHOR = 8            # 混入的经纪人锚点 —— 校准盲测判准是否漂移
CONTROL_SEED = 20260827        # ★ 写死，与主管线的 SEEDS 无关，便于复现

# ★ 不能包含空串：str.startswith(("",...)) 恒为 True，会把所有身份判为未识别。
#   空值另行单独判断。
GENERIC = ("unlabelled", "unlabeled", "unknown", "suspected", "mev",
           "fake_phishing", "compromised", "nan")


def ensemble_score(token, scale="1w"):
    """用 05 缓存的多种子嵌入重算排名集成分。与 09_verify_gap 同一实现。"""
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
        return None, d, 0
    n = len(scores[0])
    r = np.mean([rankdata(x) / n for x in scores], axis=0)
    return r, d, len(scores)


def eligibility(d):
    """资格池：与 config 中经纪人定义完全一致，唯独不含分数条件。"""
    F = C.FEATURE_NAMES
    bal = d["feat"][:, F.index("balance")]
    cp = d["feat"][:, F.index("counterparties")]
    bet = d["feat"][:, F.index("betweenness")]
    cat = d["cat"]
    m = (cat == "Unlabeled") & (bal <= np.median(bal))
    if C.BROKER_MIN_COUNTERPARTIES:
        m &= cp >= C.BROKER_MIN_COUNTERPARTIES
    if C.BROKER_REQUIRE_POSITIVE_BETWEENNESS:
        m &= bet > 0
    return m, bal, cp, bet


def matched_sample(broker_cp, pool_idx, pool_cp, n, rng):
    """为每个经纪人匹配对手方数最接近的非经纪人（不放回）。

    ★ 这是主检验的对照组。度数对齐后，两组差异不能再用"度数不同"解释。
    """
    avail = list(pool_idx)
    avail_cp = {i: pool_cp[k] for k, i in enumerate(pool_idx)}
    targets = rng.permutation(broker_cp)[:n]
    picked = []
    for t in targets:
        if not avail:
            break
        j = min(avail, key=lambda i: abs(avail_cp[i] - t))
        picked.append(j)
        avail.remove(j)
    return np.array(picked, dtype=int)


def do_sample():
    cm.banner("11 · 随机对照组抽样")
    rng = np.random.default_rng(CONTROL_SEED)
    rows = []
    for token in CONTROL_TOKENS:
        gp = C.RESULTS / f"graph_{token}_1w.npz"
        bp = C.RESULTS / f"hidden_brokers_{token}.csv"
        if not (gp.exists() and bp.exists()):
            print(f"  [X] 缺 {gp.name} 或 {bp.name}，先跑 01/05")
            return
        sc, d, n_emb = ensemble_score(token)
        if sc is None:
            print(f"  [X] {token} 缺 emb_*_s*.npy，先跑 05_c3_roles.py")
            return
        nodes = [str(x).lower() for x in d["nodes"]]
        elig, bal, cp, bet = eligibility(d)
        brokers = set(pd.read_csv(bp)["address"].astype(str).str.lower())
        is_broker = np.array([a in brokers for a in nodes])

        pool = np.flatnonzero(elig & ~is_broker)
        bidx = np.flatnonzero(is_broker)
        print(f"\n  {token}: 用 {n_emb} 份嵌入重算集成分")
        print(f"    资格池 {int(elig.sum()):,} | 其中经纪人 {len(bidx)} | "
              f"可抽对照 {len(pool):,}")
        print(f"    经纪人对手方 中位 {np.median(cp[bidx]):.0f} "
              f"[{cp[bidx].min():.0f}, {cp[bidx].max():.0f}]")
        print(f"    资格池对手方 中位 {np.median(cp[pool]):.0f} "
              f"[{cp[pool].min():.0f}, {cp[pool].max():.0f}]")
        if np.median(cp[pool]) < np.median(cp[bidx]) / 2:
            print(f"    ★ 两者度数分布差异大 —— 正是需要配对抽样的原因")

        m_idx = matched_sample(cp[bidx], pool, cp[pool],
                               min(N_MATCHED, len(pool)), rng)
        remain = np.setdiff1d(pool, m_idx)
        u_idx = rng.choice(remain, size=min(N_UNIFORM, len(remain)),
                           replace=False)
        # ★ 经纪人只抽一部分混入盲测清单。它们的身份已在主核验中确定，
        #   这里的作用是校准：若盲测下的经纪人命中率与已知值接近，
        #   说明判准没有因为"知道整份清单都是对照"而收紧。
        a_idx = rng.choice(bidx, size=min(N_BROKER_ANCHOR, len(bidx)),
                           replace=False)

        print(f"    配对对照 {len(m_idx)} 个，对手方中位 "
              f"{np.median(cp[m_idx]):.0f}  ← 与经纪人对齐")
        print(f"    随机对照 {len(u_idx)} 个，对手方中位 "
              f"{np.median(cp[u_idx]):.0f}")
        print(f"    经纪人锚点 {len(a_idx)} 个（校准用）")

        for grp, idx in (("broker_anchor", a_idx), ("matched", m_idx),
                         ("uniform", u_idx)):
            for i in idx:
                rows.append({"token": token, "group": grp,
                             "address": nodes[i],
                             "counterparties": int(cp[i]),
                             "betweenness": round(float(bet[i]), 6),
                             "score_pct": round(float(sc[i]), 4)})

    key = pd.DataFrame(rows).drop_duplicates(subset=["token", "address"])
    key.to_csv(KEY, index=False)

    blind = key.sample(frac=1.0, random_state=CONTROL_SEED)[
        ["address", "counterparties"]].copy()
    blind["verified_identity"] = ""
    blind["evidence"] = ""
    blind.to_csv(BLIND, index=False)

    cm.banner("抽样完成", "-")
    print(f"  → {BLIND.name}  {len(blind)} 个地址（已打乱，不含分组）")
    print(f"  → {KEY.name}    ★ 核验完成前请勿打开")
    print(f"\n  核验方式：Etherscan 地址页看 Name Tag → ERC-20 Token Transfers")
    print(f"  标签页 → Contract 标签页。填 verified_identity 一列：")
    print(f"    · 能确认归属某协议 → 写协议名（如 '1inch: Router V6'）")
    print(f"    · 无标签但对手方为 DEX 池/聚合器 → 'Unlabelled routing contract'")
    print(f"    · 高频套利模式 → 'MEV Bot'")
    print(f"    · 其他 → 'Unknown'")
    print(f"\n  填完执行：python 11_random_control.py score")
    cm.log_metric(SCRIPT, "BOTH", "n_blind", len(blind))


def do_score():
    cm.banner("11 · 揭盲与统计检验")
    if not (BLIND.exists() and KEY.exists()):
        print(f"  [X] 缺 {BLIND.name} 或 {KEY.name}，先跑 sample 模式")
        return
    b = pd.read_csv(BLIND)
    k = pd.read_csv(KEY)
    for df in (b, k):
        df["address"] = df["address"].astype(str).str.strip().str.lower()
    n_blank = b["verified_identity"].isna().sum() + \
        (b["verified_identity"].astype(str).str.strip() == "").sum()
    if n_blank:
        print(f"  [!] {n_blank} 行身份列为空，这些行将计为未识别。")
        print(f"      若尚未核验完，请先补齐再跑，否则会低估两组的比例。")
    m = k.merge(b[["address", "verified_identity", "evidence"]],
                on="address", how="left")

    def is_named(v):
        t = str(v).strip().lower()
        return bool(t) and not t.startswith(GENERIC)

    m["named"] = m["verified_identity"].map(is_named)
    globals()["is_named"] = is_named

    from scipy.stats import fisher_exact
    out = {}
    for token in CONTROL_TOKENS:
        sub = m[m["token"] == token]
        cm.banner(f"{token}", "-")
        print(f"    {'组':<16}{'n':>5}{'具名协议':>10}{'占比':>9}"
              f"{'中位对手方':>12}")
        stat = {}
        for g in ("broker_anchor", "matched", "uniform"):
            d = sub[sub["group"] == g]
            if not len(d):
                continue
            hit = int(d["named"].sum())
            stat[g] = (hit, len(d))
            print(f"    {g:<16}{len(d):>5}{hit:>10}{hit / len(d):>8.0%}"
                  f"{d['counterparties'].median():>12.0f}")

        # ---- 主检验用【全部已核验经纪人】，而非 8 个盲测锚点 ----
        vp = VDIR / "broker_identities.csv"
        bfull = None
        if vp.exists():
            vb = pd.read_csv(vp)
            vb.columns = [c.strip().lstrip("\ufeff") for c in vb.columns]
            ac = next(c for c in vb.columns if "addr" in c.lower())
            ic = next((c for c in vb.columns if c.lower() in
                       ("verified_identity", "identity", "name", "label")),
                      None)
            bp = C.RESULTS / f"hidden_brokers_{token}.csv"
            bset = set(pd.read_csv(bp)["address"].astype(str).str.lower())
            vb[ac] = vb[ac].astype(str).str.strip().str.lower()
            vsub = vb[vb[ac].isin(bset)]
            nh = int(vsub[ic].map(is_named).sum()) if ic else 0
            bfull = (nh, len(bset))
            print(f"    {'broker (all)':<16}{len(bset):>5}{nh:>10}"
                  f"{nh / max(len(bset), 1):>8.0%}{'':>12}")
            if "broker_anchor" in stat:
                ah, an = stat["broker_anchor"]
                d_ = abs(ah / max(an, 1) - nh / max(len(bset), 1))
                print(f"\n    [校准] 盲测锚点命中率 {ah / max(an,1):.0%} vs "
                      f"已知全集 {nh / max(len(bset),1):.0%}，差 {d_:.0%}")
                print(f"      {'判准未漂移 ✓' if d_ <= 0.15 else '[!] 差异较大，盲测下判准可能收紧，解读结果时需谨慎'}")

        if bfull and "matched" in stat:
            bh, bn = bfull
            mh, mn = stat["matched"]
            tab = [[bh, bn - bh], [mh, mn - mh]]
            odds, p = fisher_exact(tab, alternative="greater")
            print(f"\n    [主检验] 经纪人 vs 度数配对对照")
            print(f"      Fisher 精确检验（单侧）: OR = "
                  f"{odds if np.isfinite(odds) else float('inf'):.2f}, "
                  f"p = {p:.4f}")
            verdict = ("循环偏倚被排除：在度数相同的地址中，GNN 选出的"
                       "具名路由协议显著更多"
                       if p < 0.05 else
                       "未达显著：在度数相同的地址中，GNN 的筛选没有可检出的增量")
            print(f"      → {verdict}")
            if p >= 0.05 and min(bn, mn) < 30:
                print(f"      [i] 样本偏小（n={min(bn, mn)}），"
                      f"不显著可能是功效不足而非无效应")
            out[token] = {"broker_all": bfull,
                          "broker_anchor_blind": stat.get("broker_anchor"),
                          "matched": stat.get("matched"),
                          "uniform": stat.get("uniform"),
                          "odds_ratio": None if not np.isfinite(odds)
                          else round(float(odds), 3),
                          "p_value": round(float(p), 5),
                          "verdict": verdict}

    cm.save_json({"control_seed": CONTROL_SEED,
                  "n_matched": N_MATCHED, "n_uniform": N_UNIFORM,
                  "n_broker_anchor": N_BROKER_ANCHOR,
                  "tokens": list(CONTROL_TOKENS),
                  "design": "counterparty-matched control from the same "
                            "eligibility pool; blind manual verification",
                  "results": out}, "random_control.json")
    cm.banner("完成。把 results/random_control.json 的数字写进 C3")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "sample"
    if mode == "sample":
        do_sample()
    elif mode == "score":
        do_score()
    else:
        print("用法: python 11_random_control.py [sample|score]")


if __name__ == "__main__":
    main()
