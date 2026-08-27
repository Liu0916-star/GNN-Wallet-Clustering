# -*- coding: utf-8 -*-
"""
05_c3_roles.py —— C3：角色类型学 + 隐藏经纪人

提炼自 v16 的 cell 41（经纪人）+ cell 42（簇命名）。cell 40 是被覆盖的死代码，
已丢弃 —— 判据：hidden_brokers.csv 的列名是 core_score，只有 cell 41 用这个
变量名，cell 40 用的是 struct_score，两者是不同的定义。

相对 v16 的五处修正：

  1. 多种子集成（config.BROKER_CONSENSUS_SEEDS）
     单次运行不可复现：MPS 上 GraphSAGE 的 scatter 聚合是非确定性的，
     同一 seed 两次运行的经纪人集合相差 13–15%，跨币交集相差 25%。
     主口径 = 排名集成（各种子百分位排名平均后切一次 top1%）；
     投票共识（≥50% 种子）作为高精度子集与稳定性诊断并列报告。

  2. 退化经纪人过滤（config.BROKER_MIN_COUNTERPARTIES）
     v16 的定义会选进 counterparties=1、betweenness=0 的节点，
     其中两对是 vanity 孪生地址。论文称经纪人"卡在关键路径上"，
     而 betweenness=0 直接反驳该描述。

  3. 中心性对照改为等大小 top-K
     GNN 侧经集成收紧后若仍与固定百分位的 betweenness 集合比较，
     两边大小不同，"GNN 多找出 N 个"没有意义。现在同池同大小，
     只报重合度。

  4. 簇命名拆成 name + flags；财富–权力关系报连续量（Spearman）而非二元判定
     v16 的 cell 42 允许多重命名，UNI 的簇 3 被拼成
     "核心基础设施簇(...) / 巨鲸/囤币簇(...)"，无法写进表格。

  5. node2vec 全流程对照（同一集成方式），并给出 top-1% 标签构成，
     解释 C1（node2vec 富集更高）与 C3（GNN 找出更多经纪人）的表面矛盾。

★ core_score 是样本内拟合（无 CV），只作描述性筛选，不主张预测效力。
  预测证据见 C1 的交叉验证 AUC。

★ 嵌入缓存带指纹（特征+边表的 md5）。若 01_build_graph 改参数重跑，
  指纹变化会强制重训，不会静默使用旧嵌入。

用法：
    python 05_c3_roles.py            # LINK + UNI
    python 05_c3_roles.py LINK       # 单个代币
    python 05_c3_roles.py --fresh    # 忽略所有嵌入缓存，全部重训
"""
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import config as C
import common as cm

SCRIPT = "05_c3_roles"


def _fingerprint(d) -> str:
    """图数据的指纹：特征矩阵 + 边表。用于校验嵌入缓存是否仍然匹配。"""
    import hashlib
    h = hashlib.md5()
    h.update(np.ascontiguousarray(d["feat"]).tobytes())
    h.update(np.ascontiguousarray(d["edge_index"]).tobytes())
    return h.hexdigest()[:12]


def _cache_ok(cache_path, fp) -> bool:
    if "--fresh" in sys.argv:
        return False
    """缓存必须带上与当前图一致的指纹，否则视为失效。

    ★ 这是硬性要求：若 01_build_graph 改了参数重跑，而 05 静默读了旧嵌入，
      结果会对不上却不报错 —— 这类 bug 最难查。
    """
    meta = cache_path.with_suffix(".fp")
    return (cache_path.exists() and meta.exists()
            and meta.read_text().strip() == fp)


def _write_cache(cache_path, arr, fp):
    np.save(cache_path, arr)
    cache_path.with_suffix(".fp").write_text(fp)


def _train_gnn(d, seed):
    """训练一份 GraphSAGE 嵌入。与 02_c1.train_emb 同构，超参全部来自 config。"""
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import SAGEConv
    from torch_geometric.utils import negative_sampling
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    x, _ = cm.prepare_gnn_input(d["feat"])
    ei = torch.tensor(d["edge_index"], dtype=torch.long).to(dev)

    class S(nn.Module):
        def __init__(s, i, h, o):
            super().__init__()
            s.a, s.b = SAGEConv(i, h), SAGEConv(h, o)

        def forward(s, x, e):
            return s.b(F.relu(s.a(x, e)), e)

    torch.manual_seed(seed)
    xt = torch.tensor(x, dtype=torch.float).to(dev)
    m = S(xt.size(1), C.HIDDEN_DIM, C.EMB_DIM).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=C.LR)
    m.train()
    for _ in range(C.EPOCHS):
        opt.zero_grad()
        z = m(xt, ei)
        pos = (z[ei[0]] * z[ei[1]]).sum(1)
        ne = negative_sampling(ei, z.size(0), num_neg_samples=ei.size(1))
        neg = (z[ne[0]] * z[ne[1]]).sum(1)
        loss = (F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos))
                + F.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg)))
        loss.backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        return m(xt, ei).cpu().numpy()


def _multi_seed_embs(token, scale, d):
    """按 config.BROKER_CONSENSUS_SEEDS 逐个训练/复用嵌入。带指纹校验。"""
    fp = _fingerprint(d)
    out = []
    for sd in C.BROKER_CONSENSUS_SEEDS:
        cache = C.RESULTS / f"emb_{token}_{scale}_s{sd}.npy"
        if _cache_ok(cache, fp):
            out.append(np.load(cache))
            continue
        print(f"    训练种子 {sd} ...", flush=True)
        e = _train_gnn(d, sd)
        _write_cache(cache, e, fp)
        out.append(e)
    return out


def _brokers_from_emb(embs, bal, cp, bet, cat):
    """给定（已标准化的）嵌入，按论文定义筛出隐藏经纪人。

    定义：is_core 分类器概率 top1% & 余额 ≤ 中位数 & 无标签
    再加下限：对手方 ≥ BROKER_MIN_COUNTERPARTIES 且 介数 > 0

    ★ core_score 是样本内拟合（无 CV），只作描述性筛选，不主张预测效力。
      预测证据见 C1 的交叉验证 AUC。
    返回 (base_mask, strict_mask, score)
    """
    y = cm.is_core_mask(cat).astype(int)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(embs, y)
    score = clf.predict_proba(embs)[:, 1]
    return _apply_broker_rules(score, bal, cp, bet, cat) + (score,)


def _apply_broker_rules(score, bal, cp, bet, cat):
    """把经纪人定义作用在任意分数向量上。返回 (base, strict)。"""
    thr = np.percentile(score, C.BROKER_PROB_PCT)
    base = (score >= thr) & (bal <= np.median(bal)) & (cat == "Unlabeled")
    strict = base.copy()
    if C.BROKER_MIN_COUNTERPARTIES:
        strict &= cp >= C.BROKER_MIN_COUNTERPARTIES
    if C.BROKER_REQUIRE_POSITIVE_BETWEENNESS:
        strict &= bet > 0
    return base, strict


def _rank_ensemble(scores_all):
    """把各种子的分数转成百分位排名后平均 —— 尺度无关的标准集成。"""
    from scipy.stats import rankdata
    n = len(scores_all[0])
    return np.mean([rankdata(s) / n for s in scores_all], axis=0)


def _node2vec_emb(token, scale, d, n_nodes, seed=None):
    """node2vec 嵌入（纯无权拓扑），带缓存与指纹校验。装了才跑。"""
    seed = C.SEED if seed is None else seed
    cache = C.RESULTS / f"emb_n2v_{token}_{scale}_s{seed}.npy"
    fp = _fingerprint(d)
    if _cache_ok(cache, fp):
        print(f"  [缓存] 复用 {cache.name}（指纹 {fp} 匹配）")
        return np.load(cache)
    try:
        from node2vec import Node2Vec
        import networkx as nx
    except ImportError:
        return None
    print(f"    [node2vec] 训练种子 {seed} ...", flush=True)
    Gx = nx.DiGraph()
    Gx.add_edges_from(d["edge_index"].T.tolist())
    np.random.seed(seed)
    n2v = Node2Vec(Gx, dimensions=C.EMB_DIM, walk_length=80, num_walks=10,
                   workers=1, quiet=True, seed=seed)
    mdl = n2v.fit(window=10, min_count=1, seed=seed, workers=1)
    e = np.array([mdl.wv[str(i)] if str(i) in mdl.wv else np.zeros(C.EMB_DIM)
                  for i in range(n_nodes)])
    _write_cache(cache, e, fp)
    return e


def run(token, scale="1w"):
    cm.banner(f"05 · C3 角色类型学 {token} / {scale}")
    gp = C.RESULTS / f"graph_{token}_{scale}.npz"
    if not gp.exists():
        print(f"  [X] 缺 {gp.name}，先跑 01_build_graph.py")
        return
    d = np.load(gp, allow_pickle=True)
    print(f"  [多种子] {len(C.BROKER_CONSENSUS_SEEDS)} 个种子"
          f"（缓存命中则跳过训练）")
    embs_all = _multi_seed_embs(token, scale, d)
    emb = embs_all[0]                       # 主种子，用于聚类与展示
    feat, nodes, cat = d["feat"], list(d["nodes"]), d["cat"]
    F_ = C.FEATURE_NAMES
    bal = feat[:, F_.index("balance")]
    cp = feat[:, F_.index("counterparties")]
    bet = feat[:, F_.index("betweenness")]
    deg = feat[:, F_.index("in_deg")] + feat[:, F_.index("out_deg")]
    y_core = cm.is_core_mask(cat).astype(int)
    print(f"  节点 {len(nodes):,} | 核心锚点 {y_core.sum()}")

    # ---------- 聚类 ----------
    embs = StandardScaler().fit_transform(emb)
    km = KMeans(n_clusters=C.N_CLUSTERS, random_state=C.SEED, n_init=10)
    lab = km.fit_predict(embs)

    typ = {}
    core_rates = {c: y_core[lab == c].mean() for c in range(C.N_CLUSTERS)}
    med_bals = {c: float(np.median(bal[lab == c])) for c in range(C.N_CLUSTERS)}
    top_core = max(core_rates, key=core_rates.get)
    top_bal = max(med_bals, key=med_bals.get)

    print(f"\n  {'簇':>3}{'规模':>7}{'核心率':>9}{'中位余额':>12}"
          f"{'中位对手方':>10}{'中位介数':>11}  标记")
    for c in range(C.N_CLUSTERS):
        m = lab == c
        flags = []
        if c == top_core:
            flags.append("core_infrastructure")
        if c == top_bal:
            flags.append("whale_hoarding")
        cats, cnt = np.unique(cat[m & (cat != "Unlabeled")], return_counts=True)
        typ[f"cluster_{c}"] = {
            "size": int(m.sum()),
            "core_rate": round(float(core_rates[c]), 4),
            "core_n": int(y_core[m].sum()),
            "med_balance": round(med_bals[c], 2),
            "med_counterparties": float(np.median(cp[m])),
            "med_betweenness": float(np.median(bet[m])),
            "med_deg": float(np.median(deg[m])),
            "flags": flags,          # ★ 不再拼成一个复合字符串
            "top_labels": dict(zip(cats.tolist(), cnt.tolist())),
        }
        print(f"  {c:>3}{int(m.sum()):>7,}{core_rates[c] * 100:>8.2f}%"
              f"{med_bals[c]:>12.2f}{np.median(cp[m]):>10.0f}"
              f"{np.median(bet[m]):>11.5f}  {','.join(flags) or '-'}")

    # ---------- 财富 vs 结构权力：报连续量，不做二元判定 ----------
    # v16 用"核心率最高簇 == 余额最高簇"作二元判定，太脆：只要两个簇的
    # 中位余额接近，结论就会翻转。这里同时给簇级相关与节点级相关。
    from scipy.stats import spearmanr
    cr = np.array([core_rates[c] for c in range(C.N_CLUSTERS)])
    mb = np.array([med_bals[c] for c in range(C.N_CLUSTERS)])
    rho_c = float(spearmanr(cr, mb).statistic) if C.N_CLUSTERS > 2 else np.nan
    zero_frac = float((bal <= 0).mean())
    nz = bal > 0                      # ★ 只在持币节点上算，否则大量并列 0 稀释相关
    rho_n = float(spearmanr(bal, cp).statistic)
    rho_nb = float(spearmanr(bal, bet).statistic)
    rho_n_nz = float(spearmanr(bal[nz], cp[nz]).statistic) if nz.sum() > 10 else np.nan
    rho_nb_nz = float(spearmanr(bal[nz], bet[nz]).statistic) if nz.sum() > 10 else np.nan
    print(f"\n  [财富 vs 结构权力]")
    print(f"    零余额节点占 {zero_frac * 100:.1f}%"
          f"（{int((~nz).sum()):,}/{len(bal):,}）—— 全样本的 Spearman 会被并列值稀释")
    print(f"    簇级 Spearman(核心率, 中位余额) = {rho_c:+.3f}"
          f"（{C.N_CLUSTERS} 个簇）")
    print(f"    {'':<4}{'全部节点':>12}{'仅持币节点':>14}")
    print(f"    余额 vs 对手方 {rho_n:>+11.3f}{rho_n_nz:>+14.3f}")
    print(f"    余额 vs 介数   {rho_nb:>+11.3f}{rho_nb_nz:>+14.3f}"
          f"   (n={int(nz.sum()):,})")
    if max(abs(rho_n_nz), abs(rho_nb_nz)) < 0.15:
        print(f"    ★ 两项均接近 0 → 节点层面财富与结构权力基本【正交】。")
        print(f"      这正是不平等悖论的机制：Gini 0.99 的财富分布与未集中的")
        print(f"      结构权力可以共存，因为二者本就不相关。")
    if top_core == top_bal:
        print(f"    核心率最高簇与余额最高簇同为簇 {top_core} → 倾向【融合】")
    else:
        print(f"    核心簇 {top_core} 与巨鲸簇 {top_bal} 不同 → 倾向【分离】"
              f"（巨鲸簇核心率 {core_rates[top_bal] * 100:.1f}%）")
    print(f"    ★ v16 报告 LINK【分离】、UNI【融合】。若本次两币同向，"
          f"该跨币对比须从正文删除，")
    print(f"      并说明差异源于边权修正（旧口径下核心簇中位余额仅 0.33）。")
    wp = {"spearman_cluster_core_vs_balance": round(rho_c, 4),
          "spearman_node_balance_vs_counterparties": round(rho_n, 4),
          "spearman_node_balance_vs_betweenness": round(rho_nb, 4),
          "spearman_nonzero_balance_vs_counterparties": round(rho_n_nz, 4),
          "spearman_nonzero_balance_vs_betweenness": round(rho_nb_nz, 4),
          "zero_balance_frac": round(zero_frac, 4),
          "n_nonzero": int(nz.sum()),
          "top_core_cluster": int(top_core), "top_balance_cluster": int(top_bal),
          "verdict": "fused" if top_core == top_bal else "separated"}

    # ---------- 隐藏经纪人（多种子共识）----------
    # ★ 单次运行不可复现：MPS 上 GraphSAGE 的 scatter 聚合非确定性，
    #   同一 seed 两次运行的经纪人集合相差 13–15%，跨币交集相差 25%。
    #   因此报告【共识集合】：出现在 ≥50% 种子中的节点。
    n_seed = len(embs_all)
    votes = np.zeros(len(nodes), dtype=int)
    votes_base = np.zeros(len(nodes), dtype=int)
    scores_all = []
    for e in embs_all:
        es = StandardScaler().fit_transform(e)
        b_, s_, sc_ = _brokers_from_emb(es, bal, cp, bet, cat)
        votes += s_.astype(int)
        votes_base += b_.astype(int)
        scores_all.append(sc_)
    score_sd = np.std(scores_all, axis=0)
    freq = votes / n_seed

    # ---- 主口径：排名集成（先平均百分位排名，再切一次 top1%）----
    # 投票口径先在每个种子上切 top1% 再计票，硬阈值边界处的节点会反复进出；
    # 排名集成对底层信号求平均，边界抖动小得多，是标准的 ensemble 做法。
    rank_score = _rank_ensemble(scores_all)
    base_r, strict_r = _apply_broker_rules(rank_score, bal, cp, bet, cat)
    vote_strict = freq >= C.BROKER_CONSENSUS_MIN_FRAC

    if C.BROKER_ENSEMBLE == "rank":
        base, strict, score = base_r, strict_r, rank_score
    else:
        base = (votes_base / n_seed) >= C.BROKER_CONSENSUS_MIN_FRAC
        strict, score = vote_strict, np.mean(scores_all, axis=0)
    core_set = freq >= C.BROKER_CORE_SET_FRAC
    dropped = int(base.sum() - (base & strict).sum())

    print(f"\n  [隐藏经纪人 · {n_seed} 种子集成]")
    print(f"    主口径 = {C.BROKER_ENSEMBLE}"
          f"（{'排名集成' if C.BROKER_ENSEMBLE == 'rank' else '投票共识'}）")
    print(f"    {'口径':<30}{'节点数':>8}")
    print(f"    {'★ 排名集成（top1% 切一次）':<30}"
          f"{int(strict_r.sum()):>8}")
    print(f"    {'投票 ≥50% 种子':<30}{int(vote_strict.sum()):>8}")
    print(f"    {'两者交集':<30}"
          f"{int((strict_r & vote_strict).sum()):>8}")
    print(f"    {'单种子平均':<30}{votes.sum() / n_seed:>8.1f}")
    print(f"\n    [稳定性诊断] 投票频次分布"
          f"（曾入选过 {int((votes > 0).sum())} 个节点）")
    print(f"    {'门槛':<26}{'节点数':>8}")
    for thr, nm in ((1.0, f"全部 {n_seed}/{n_seed} 种子"),
                    (C.BROKER_CORE_SET_FRAC,
                     f"≥{C.BROKER_CORE_SET_FRAC:.0%} 种子（高置信）"),
                    (C.BROKER_CONSENSUS_MIN_FRAC,
                     f"≥{C.BROKER_CONSENSUS_MIN_FRAC:.0%} 种子"),
                    (0.1, "≥10% 种子（含偶发）")):
        print(f"    {nm:<26}{int((freq >= thr).sum()):>8}")
    hist = {k: int((votes == k).sum()) for k in range(1, n_seed + 1)}
    print(f"    直方图: " +
          " ".join(f"{k}:{v}" for k, v in hist.items() if v))
    churn = int(((votes >= 1) & (votes <= 2)).sum())
    print(f"    仅出现 1–2 次的 {churn} 个"
          f"（占曾入选的 {churn / max(int((votes > 0).sum()), 1):.0%}）")
    print(f"    ★ 硬性 top-1% 切点使边界节点反复进出，这是定义的固有性质。")
    print(f"      论文须同时报告集成集合与本频次分布，不报单次运行结果。")
    print(f"    ★ 复现性：集成结果对给定的 {n_seed} 份嵌入是确定的，但嵌入本身")
    print(f"      由 GPU 训练（MPS 的 scatter 聚合非确定性），第三方重训会略有差异。")
    print(f"      归档时须连同 results/emb_*_s*.npy 与 .fp 指纹一起提交。")

    # 退化节点在共识口径下的剔除情况
    deg_mask = base & ~strict
    if deg_mask.sum():
        print(f"    被下限剔除的退化节点 {int(deg_mask.sum())} 个"
              f"（对手方<{C.BROKER_MIN_COUNTERPARTIES} 或 介数=0）:")
        for i in np.flatnonzero(deg_mask)[:6]:
            print(f"      {nodes[i]}  对手方={int(cp[i])} 介数={bet[i]:.6f} "
                  f"出现率={freq[i]:.0%}")

    bidx = np.flatnonzero(strict)
    brokers = pd.DataFrame({
        "rank": range(1, len(bidx) + 1),
        "address": [nodes[i] for i in bidx],
        "ensemble_score": np.round(score[bidx], 4),
        "seed_frequency": np.round(freq[bidx], 2),
        "score_sd_across_seeds": np.round(score_sd[bidx], 4),
        "in_vote_consensus": vote_strict[bidx],
        "high_confidence": core_set[bidx],
        "balance": bal[bidx], "counterparties": cp[bidx].astype(int),
        "betweenness": np.round(bet[bidx], 6), "cluster": lab[bidx],
    }).sort_values(["ensemble_score", "seed_frequency"],
                   ascending=[False, False])
    brokers["rank"] = range(1, len(brokers) + 1)

    # ---- betweenness 对照（等大小 top-K，口径对齐）----
    # ★ GNN 侧经过集成收紧，若 betweenness 仍取固定百分位，两边集合大小不同，
    #   "GNN 多找出 N 个" 就没有意义。这里取与 GNN 等大小的 top-K。
    elig = (bal <= np.median(bal)) & (cat == "Unlabeled")
    if C.BROKER_MIN_COUNTERPARTIES:
        elig &= cp >= C.BROKER_MIN_COUNTERPARTIES
    if C.BROKER_REQUIRE_POSITIVE_BETWEENNESS:
        elig &= bet > 0
    K = int(strict.sum())
    ei_ = np.flatnonzero(elig)
    topk = ei_[np.argsort(-bet[ei_])[:K]] if len(ei_) else np.array([], int)
    bet_def = np.zeros(len(nodes), bool)
    bet_def[topk] = True
    overlap = int((strict & bet_def).sum())
    print(f"\n    [与中心性基线对照 · 等大小 K={K}]")
    print(f"    GNN {K} | betweenness top-{K}（同一资格池） | 重叠 {overlap}")
    print(f"    → 重合度 {overlap / max(K, 1):.0%}"
          f"（等大小口径下双方独有数必然相等，各 {K - overlap} 个，"
          f"故不可再写'GNN 多找出 N 个'）")
    if C.BROKER_SIZE_MATCHED_BASELINE:
        bthr = np.percentile(bet, C.BROKER_PROB_PCT)
        bet_pct = elig & (bet >= bthr)
        print(f"    [参考] 固定百分位口径下 betweenness 集合为 "
              f"{int(bet_pct.sum())} 个，与 GNN 不等大，不用于主表")

    # ---------- node2vec 对照：GNN 到底带来了什么 ----------
    # C1 显示 node2vec 在分类与富集上不输 GNN。若它也能找出同一批经纪人，
    # 则 GNN 在 C3 里并非必需，论文的方法论述必须弱化。
    n2e = _node2vec_emb(token, scale, d, len(nodes))
    n2v_stats = None
    if n2e is not None:
        # node2vec 侧同样做共识，否则与 GNN 的比较不对等
        # node2vec 侧用同一套集成方式，否则两边口径不同，比较无效
        sc2 = []
        for sd in C.BROKER_CONSENSUS_SEEDS[:C.N2V_CONSENSUS_N]:
            e2 = _node2vec_emb(token, scale, d, len(nodes), seed=sd)
            if e2 is None:
                break
            _, _, s2 = _brokers_from_emb(StandardScaler().fit_transform(e2),
                                         bal, cp, bet, cat)
            sc2.append(s2)
        if not sc2:
            sc2 = [np.zeros(len(nodes))]
        r2 = _rank_ensemble(sc2)
        _, s_n2v = _apply_broker_rules(r2, bal, cp, bet, cat)
        print(f"    （node2vec 用同一集成方式：{len(sc2)} 个种子排名平均）")
        gset = set(np.flatnonzero(strict))
        nset = set(np.flatnonzero(s_n2v))
        inter = gset & nset
        jac = len(inter) / max(len(gset | nset), 1)
        print(f"\n  [GNN vs node2vec 经纪人集合]")
        print(f"    GNN {len(gset)} 个 | node2vec {len(nset)} 个 | "
              f"交集 {len(inter)} | Jaccard {jac:.3f}")

        def prof(idx):
            idx = list(idx)
            if not idx:
                return (0, 0, 0)
            return (float(np.median(cp[idx])), float(np.median(bet[idx])),
                    float(np.median(deg[idx])))

        print(f"    {'集合':<22}{'中位对手方':>10}{'中位介数':>12}{'中位度':>9}")
        for nm, ix in (("GNN 独有", gset - nset), ("两者共有", inter),
                       ("node2vec 独有", nset - gset)):
            a, b_, c_ = prof(ix)
            print(f"    {nm:<22}{a:>10.0f}{b_:>12.6f}{c_:>9.0f}  (n={len(ix)})")

        # ---- 与已核验身份对照 ★ 存在循环偏倚，见下方提示 ----
        vp = C.ROOT / "verification" / "broker_identities.csv"
        if vp.exists():
            vb = pd.read_csv(vp)
            acol_ = [c for c in vb.columns if "addr" in c.lower()][0]
            icol_ = next((c for c in vb.columns
                          if c.lower() in ("verified_identity", "identity",
                                           "name", "label")), None)
            # ★ 只有【填了身份】的行才算已核验，否则命中率会虚高到 100%
            vset = set()
            for i_, a_ in enumerate(vb[acol_]):
                v_ = vb[icol_].iloc[i_] if icol_ else "verified"
                if pd.isna(v_) or not str(v_).strip() \
                        or str(v_).strip().lower() == "nan":
                    continue
                vset.add(str(a_).strip().lower())
            # ★ 核验完成后，我们集合里的地址【全部】都在核验表中，
            #   "命中率"必然是 100%，那是构造出来的，不能作为质量指标。
            #   有意义的是【具体身份占比】：有多少被识别为某个具体协议，
            #   而不是 Unlabelled / MEV Bot 这类通用条目。
            GEN = ("unlabelled", "unlabeled", "unknown", "suspected", "mev",
                   "fake_phishing", "compromised")
            # run() 里没有 ident（那是 cross_token 的局部变量），
            # 这里直接从核验表重建"具体身份"集合。
            spec = set()
            for _i2, _a2 in enumerate(vb[acol_]):
                _v2 = vb[icol_].iloc[_i2] if icol_ else ""
                if pd.isna(_v2) or not str(_v2).strip():
                    continue
                if str(_v2).strip().lower().startswith(GEN):
                    continue
                spec.add(str(_a2).strip().lower())
            ident = vset          # 供下方 print 统计条数用

            def _hit(mask):
                ix = np.flatnonzero(mask)
                h = sum(1 for i in ix if nodes[i] in vset)
                sp = sum(1 for i in ix if nodes[i] in spec)
                n_ = len(ix)
                return h, sp, n_

            print(f"\n    [核验构成] 三层集合并排")
            print(f"    {'集合':<26}{'在表内':>9}{'具体身份':>10}{'占比':>8}")
            layers = [("GNN 排名集成（主）", strict_r),
                      ("GNN 投票 ≥50%", vote_strict),
                      ("GNN 两者交集（最保守）", strict_r & vote_strict),
                      ("node2vec 排名集成", s_n2v)]
            hits = {}
            for nm, mk in layers:
                h, sp, n_ = _hit(mk)
                r_ = sp / n_ if n_ else float("nan")
                hits[nm] = {"in_table": h, "specific_identity": sp, "n": n_,
                            "specific_share": round(r_, 3)}
                print(f"    {nm:<26}{f'{h}/{n_}':>9}{sp:>10}{r_:>7.0%}")
            print(f"    ★ 「在表内」在核验完成后必为满分（我们逐个查过），"
                  f"不作为质量指标；")
            print(f"      有意义的是「具体身份」—— 被识别为某个协议而非 "
                  f"Unlabelled/MEV 的比例。")
            print(f"    [!] 核验表共 {len(ident)} 条，其中 {len(spec)} 条有具体身份。"
                  f"这些地址是当初用 GNN 定义筛出后才去核验的，")
            print(f"      存在循环偏倚，node2vec 一侧的数字只可作参考。")
            n2v_stats = {"verification_by_layer": hits,
                         "n_verified_total": len(ident),
                         "n_specific_identity": len(spec)}

        # ---- top1% 的标签构成：解释 C1 与 C3 为何看似矛盾 ----
        # C1 里 node2vec 的邻域富集更高，C3 里它找出的经纪人却少得多。
        # 机制：node2vec 把【已有标签】的核心设施排在最前，
        #      GNN 把【无标签】的高吞吐节点排上来。
        # 一个擅长 recall known，一个擅长 discover unknown —— 同一机制的两面。
        def _top1_profile(sc):
            thr = np.percentile(sc, C.BROKER_PROB_PCT)
            m = sc >= thr
            n_ = int(m.sum())
            lab_frac = float((cat[m] != "Unlabeled").mean()) if n_ else np.nan
            core_frac = float(cm.is_core_mask(cat[m]).mean()) if n_ else np.nan
            return n_, lab_frac, core_frac

        g_n, g_lab, g_core = _top1_profile(score)
        n_n, n_lab, n_core = _top1_profile(r2)
        print(f"\n    [top-1% 的标签构成] 解释 C1 与 C3 的表面矛盾")
        print(f"    {'方法':<20}{'top1%':>8}{'有标签占比':>12}{'其中核心锚点':>14}")
        print(f"    {'GNN':<20}{g_n:>8}{g_lab:>11.0%}{g_core:>13.0%}")
        print(f"    {'node2vec':<20}{n_n:>8}{n_lab:>11.0%}{n_core:>13.0%}")
        if n_lab > g_lab:
            print(f"    → node2vec 的 top1% 里有标签者更多"
                  f"（{n_lab:.0%} vs {g_lab:.0%}）：它擅长把【已知】设施排在最前，")
            print(f"      因此 C1 的邻域富集更高，但被'无标签'过滤后剩下的经纪人更少。")
            print(f"      GNN 则把【未知】的高吞吐节点排上来 —— C1 与 C3 并不矛盾。")
        n2v_stats = (n2v_stats or {}) | {
            "n_gnn": len(gset), "n_node2vec": len(nset),
            "overlap": len(inter), "jaccard": round(jac, 4),
            "top1_labeled_frac_gnn": round(g_lab, 4),
            "top1_labeled_frac_node2vec": round(n_lab, 4),
            "top1_core_frac_gnn": round(g_core, 4),
            "top1_core_frac_node2vec": round(n_core, 4)}
        if jac > 0.7:
            print(f"    → 高度重叠：GNN 在 C3 中并非必需，"
                  f"论文应弱化「只有 GNN 能发现」的表述")
        else:
            print(f"    → 差异明显：两种嵌入捕捉到不同的结构角色，"
                  f"需在正文说明各自侧重")

    cm.save_csv(brokers, f"hidden_brokers_{token}.csv")
    cm.save_json({"token": token, "scale": scale, "n_nodes": len(nodes),
                  "n_clusters": C.N_CLUSTERS, "typology": typ,
                  "wealth_power": wp,
                  "hidden_brokers": {
                      "consensus_seeds": list(C.BROKER_CONSENSUS_SEEDS),
                      "consensus_min_frac": C.BROKER_CONSENSUS_MIN_FRAC,
                      "per_seed_mean": round(float(votes.sum() / n_seed), 1),
                      "consensus_set": int(strict.sum()),
                      "high_confidence_set": int(core_set.sum()),
                      "unanimous_set": int((freq >= 1.0).sum()),
                      "freq_histogram": {f"{k}/{n_seed}":
                                         int((votes == k).sum())
                                         for k in range(1, n_seed + 1)},
                      "gnn_def_v16": int(base.sum()),
                      "gnn_def_filtered": int(strict.sum()),
                      "n_degenerate_dropped": dropped,
                      "betweenness_def_topk": int(bet_def.sum()),
                      "overlap_size_matched": overlap,
                      "ensemble": C.BROKER_ENSEMBLE,
                      "rank_ensemble_set": int(strict_r.sum()),
                      "vote_consensus_set": int(vote_strict.sum()),
                      "rank_vote_overlap": int((strict_r & vote_strict).sum()),
                      "churn_1_2_seeds": churn,
                      "definition": f"[{C.BROKER_ENSEMBLE} ensemble over "
                                    f"{len(C.BROKER_CONSENSUS_SEEDS)} seeds] "
                                    f"top{100 - C.BROKER_PROB_PCT}% & "
                                    f"balance<=median & unlabeled & "
                                    f"counterparties>="
                                    f"{C.BROKER_MIN_COUNTERPARTIES} & "
                                    f"betweenness>0",
                      "caveat": "core_score 为样本内拟合，仅作描述性筛选，"
                                "不主张预测效力（预测证据见 C1 的 AUC）"},
                  "node2vec_contrast": n2v_stats},
                 f"c3_typology_{token}.json")
    cm.log_metric(SCRIPT, token, "n_brokers", int(strict.sum()),
                  f"v16={int(base.sum())}, dropped={dropped}")
    return brokers


def cross_token():
    """跨币重叠 —— 论文最强的 C3 结论之一。"""
    ps = {t: C.RESULTS / f"hidden_brokers_{t}.csv" for t in C.TOKENS}
    if not all(p.exists() for p in ps.values()):
        return
    cm.banner("跨币重叠")
    dfs = {t: pd.read_csv(p) for t, p in ps.items()}
    sets = {t: set(d["address"].str.lower()) for t, d in dfs.items()}
    a, b = C.TOKENS
    inter = sets[a] & sets[b]
    print(f"  {a} {len(sets[a])} 个 | {b} {len(sets[b])} 个 | 交集 {len(inter)}")
    for t, o in ((a, b), (b, a)):
        top10 = set(dfs[t].nlargest(10, "counterparties")["address"].str.lower())
        print(f"  {t} 对手方最高的 10 个里，{len(top10 & sets[o])} 个也在 {o}")

    # ★ 这里要查的是【人工核验出的身份】，不是标签快照。
    #   经纪人的定义里已含 cat=="Unlabeled"，拿标签库去查必然全部为 0，
    #   那是同义反复。v16 第 779 行的争议点是"有没有已知聚合器"，
    #   而已知聚合器恰恰是靠 Etherscan 人工核验认出来的。
    vp = C.ROOT / "verification" / "broker_identities.csv"
    ident, n_id = {}, 0
    if vp.exists():
        vb = pd.read_csv(vp)
        vb.columns = [c.strip().lstrip("\ufeff") for c in vb.columns]
        acol = next(c for c in vb.columns if "addr" in c.lower())
        icol = next((c for c in vb.columns
                     if c.lower() in ("verified_identity", "identity",
                                      "name", "label")), None)
        # ★ 空值或 nan 不算已核验：模板合并进来但两列没填时，
        #   若不过滤会让命中率虚假地变成 100%。
        ident = {}
        # ★ 变量名不能用 a / b / v：外层 a, b = C.TOKENS，遮蔽会让后面的
        #   sets[a] 变成 sets[<地址>] 并抛 KeyError。
        for _i, _addr in enumerate(vb[acol]):
            _v = vb[icol].iloc[_i] if icol else "verified"
            if pd.isna(_v) or not str(_v).strip() \
                    or str(_v).strip().lower() == "nan":
                continue
            ident[str(_addr).strip().lower()] = str(_v).strip()
        n_blank = len(vb) - len(ident)
        if n_blank:
            print(f"\n  [!] 核验表 {len(vb)} 行中 {n_blank} 行的身份列为空，"
                  f"已排除；请填完 to_verify_template.csv 再合并")
        n_id = len(ident)
        print(f"\n  [人工核验表] {n_id} 条")
    else:
        print(f"\n  [!] 未找到 verification/broker_identities.csv，"
              f"跳过身份对照")

    GENERIC = ("unlabelled", "unlabeled", "unknown", "suspected", "mev",
               "fake_phishing", "compromised")
    named = {x: ident[x] for x in inter
             if x in ident and not str(ident[x]).lower().startswith(GENERIC)}
    unnamed = sorted(inter - set(named))
    print(f"\n  ★ 跨币经纪人 {len(inter)} 个中，{len(named)} 个已核验出具体身份、"
          f"{len(unnamed)} 个仍无身份")
    if ident:
        print(f"    → v16 第 779 行标题「共享经纪人都没有标签」应改为"
              f"「{len(unnamed)} 个共享经纪人没有可核验的身份」")
        print(f"      其余 {len(named)} 个是已核验的已知聚合器 —— "
              f"两段由此互补而非互斥。")
        print(f"      具体是哪几个见下方【论文点名聚合器的实际归属】，"
              f"正文清单须据此核对")
    for x, nm in list(named.items())[:15]:
        print(f"      {x}  {nm}")

    # ★ 论文第 765 行点名六个聚合器"同时中介两个代币"。加了经纪人下限之后，
    #   有的可能落选或只剩单币。逐一定位，避免正文与结果文件对不上。
    # "0x protocol" 而非 "0x"：后者会命中任何含地址的字符串
    CLAIMED = {"1inch": "1inch", "uniswap x": "Uniswap X",
               "universal router": "Uniswap V4 Universal Router",
               "0x protocol": "0x Protocol", "0x proto": "0x Protocol",
               "bitget": "Bitget", "mayan": "Mayan", "metamask": "MetaMask",
               "li.fi": "LI.FI", "lifi": "LI.FI", "velora": "Velora",
               "augustus": "Velora", "okx": "OKX", "rizzolver": "Uniswap X"}
    loc = {}
    if ident:
        print(f"\n  [论文点名聚合器的实际归属]")
        print(f"    {'聚合器':<34}{'状态':<12}地址")
        loc = {}
        for addr, nm in ident.items():
            low = str(nm).lower()
            key = next((v for k, v in CLAIMED.items() if k in low), None)
            if not key:
                continue
            in_a, in_b = addr in sets[a], addr in sets[b]
            st = ("跨币" if in_a and in_b else f"仅{a}" if in_a
                  else f"仅{b}" if in_b else "两边都未入选")
            loc.setdefault(key, []).append((st, addr, nm))
        for key in sorted(loc):
            for st, addr, nm in loc[key]:
                print(f"    {nm[:32]:<34}{st:<12}{addr}")
        dropped_claims = [k for k, v in loc.items()
                          if all(x[0] == "两边都未入选" for x in v)]
        if dropped_claims:
            print(f"    [!] {dropped_claims} 在本次口径下未进入任何经纪人集合，")
            print(f"        正文第 765 行的点名清单需据此修订")

    cm.save_json({"tokens": list(C.TOKENS), "n_shared": len(inter),
                  "shared": sorted(inter),
                  "claimed_aggregator_status": {
                      k: [{"status": x[0], "address": x[1], "identity": x[2]}
                          for x in v] for k, v in
                      loc.items()},
                  "shared_identified": named,
                  "n_identified": len(named),
                  "n_unidentified": len(unnamed),
                  "verification_table_size": n_id},
                 "c3_cross_token.json")


def main():
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    if "--fresh" in sys.argv:
        print("[--fresh] 忽略所有嵌入缓存，全部重新训练")
    for t in (a if a else C.TOKENS):
        run(t)
    cross_token()
    cm.banner("完成。下一步：python 06_concentration.py")


if __name__ == "__main__":
    main()
