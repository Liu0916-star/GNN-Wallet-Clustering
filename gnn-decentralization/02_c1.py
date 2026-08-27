# -*- coding: utf-8 -*-
"""
02_c1.py —— C1：GNN 嵌入是经典中心性的超集

三条证据 + 多种子稳健性 + 消融，一次跑完：
  证据1   预测余额 top5%（内部目标）
  证据1'  预测 is_core（外部标签，防泄漏）      ← 正文主表
  证据2   邻域纯度富集倍数
  多种子  5 个种子的 mean ± sd                  ← Neurocomputing 基本必然要求
  消融A   去节点特征（保留图结构）              ← v16 只做过 LINK
  消融B   node2vec 基线（可选，装了才跑）

依赖 01_build_graph.py 的 results/graph_{TOKEN}_{SCALE}.npz。

★ 两个必须守住的方法论细节：
  1. 防泄漏顺序：先剔 balance，再对剩余 7 维 log1p+z-score（common 已封装）
  2. CV 必须 shuffle：正样本聚集，不打乱会让 AUC 系统性 < 0.5

用法：python 02_c1.py [TOKEN] [SCALE]
"""
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import negative_sampling

import config as C
import common as cm

SCRIPT = "02_c1"
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
CV = StratifiedKFold(n_splits=C.CV_FOLDS, shuffle=C.CV_SHUFFLE,
                     random_state=C.SEED)


class SAGE(nn.Module):
    def __init__(s, i, h, o):
        super().__init__()
        s.c1, s.c2 = SAGEConv(i, h), SAGEConv(h, o)

    def forward(s, x, ei):
        return s.c2(F.relu(s.c1(x, ei)), ei)


def _loss(z, ei):
    pos = (z[ei[0]] * z[ei[1]]).sum(1)
    ne = negative_sampling(ei, z.size(0), num_neg_samples=ei.size(1))
    neg = (z[ne[0]] * z[ne[1]]).sum(1)
    return (F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos))
            + F.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg)))


def train_emb(x, ei, seed):
    torch.manual_seed(seed)
    xt = torch.tensor(x, dtype=torch.float).to(DEV)
    m = SAGE(xt.size(1), C.HIDDEN_DIM, C.EMB_DIM).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=C.LR)
    m.train()
    for _ in range(C.EPOCHS):
        opt.zero_grad()
        loss = _loss(m(xt, ei), ei)
        loss.backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        return m(xt, ei).cpu().numpy()


def auc(X, y):
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    Xs = StandardScaler().fit_transform(X)
    s = cross_val_score(LogisticRegression(max_iter=1000,
                                           class_weight="balanced"),
                        Xs, y, cv=CV, scoring="roc_auc")
    return float(s.mean()), float(s.std())


def purity(emb, y, k=None):
    """邻域纯度：每个正样本的 k 近邻中正样本比例。分块算，省显存。"""
    k = k or C.PURITY_K
    z = torch.tensor(emb, dtype=torch.float, device=DEV)
    z = z / (z.norm(dim=1, keepdim=True) + 1e-9)
    yt = torch.tensor(y, dtype=torch.float, device=DEV)
    idx = np.flatnonzero(y)
    # ★ sim 矩阵是 B×N，64万档 B=512 会吃掉 ~1.3GB 显存。按 N 自适应。
    B = max(16, min(512, int(2e7 // max(z.shape[0], 1))))
    tot = 0.0
    for i in range(0, len(idx), B):
        sub = torch.tensor(idx[i:i + B], device=DEV)
        sim = z[sub] @ z.T
        sim[torch.arange(len(sub), device=DEV), sub] = -2
        nb = sim.topk(k, dim=1).indices
        tot += yt[nb].mean(1).sum().item()
    return tot / max(len(idx), 1)


def run(token, scale):
    cm.banner(f"02 · C1 {token} / {scale}")
    p = C.RESULTS / f"graph_{token}_{scale}.npz"
    if not p.exists():
        print(f"  [X] 缺 {p.name}，先跑 01_build_graph.py {token} {scale}")
        return
    d = np.load(p, allow_pickle=True)
    feat, cat = d["feat"], d["cat"]
    ei = torch.tensor(d["edge_index"], dtype=torch.long).to(DEV)

    x, names = cm.prepare_gnn_input(feat)
    y_core = cm.is_core_mask(cat).astype(int)
    bi = C.FEATURE_NAMES.index("balance")
    y_bal = (feat[:, bi] >= np.percentile(feat[:, bi], 95)).astype(int)
    ci = [names.index("betweenness"), names.index("pagerank")]
    X_cent = x[:, ci]

    print(f"  节点 {len(y_core):,} | 核心锚点 {y_core.sum()} | "
          f"GNN 输入 {len(names)} 维 | device={DEV}")
    print(f"  CV = StratifiedKFold({C.CV_FOLDS}, shuffle={C.CV_SHUFFLE}, "
          f"rs={C.SEED})")

    rec = {"token": token, "scale": scale, "n_nodes": int(len(y_core)),
           "seeds": list(C.SEEDS_BY_SCALE.get(scale, C.SEEDS)),
           "n_anchors": int(y_core.sum()), "gnn_input": names,
           "cv": f"StratifiedKFold({C.CV_FOLDS},shuffle={C.CV_SHUFFLE},"
                 f"seed={C.SEED})",
           "preprocessing": "drop balance from raw, THEN log1p+zscore on 7"}

    # ---------- 与种子无关的基线 ----------
    print("\n  [基线]（与种子无关）")
    for tag, y in (("is_core", y_core), ("balance_top5pct", y_bal)):
        a_c, s_c = auc(X_cent, y)
        a_r, s_r = auc(x, y)
        rec[f"{tag}_centrality"] = {"auc": round(a_c, 4), "sd_fold": round(s_c, 4)}
        rec[f"{tag}_raw_struct"] = {"auc": round(a_r, 4), "sd_fold": round(s_r, 4)}
        print(f"    {tag:16} centrality {a_c:.4f}  raw-struct {a_r:.4f}")

    # ---------- 多种子 GNN + 消融 ----------
    seeds = C.SEEDS_BY_SCALE.get(scale, C.SEEDS)
    print(f"\n  [多种子] {list(seeds)}"
          + (f"   ← {scale} 档单次训练很贵，按 config.SEEDS_BY_SCALE 降至 "
             f"{len(seeds)} 个" if len(seeds) < len(C.SEEDS) else ""))
    print(f"    {'seed':>6}{'GNN is_core':>13}{'GNN balance':>13}"
          f"{'去特征':>10}{'富集':>9}")
    x_nofeat = np.ones((x.shape[0], 1))
    rows = []
    rnd = y_core.mean()
    for sd in seeds:
        emb = train_emb(x, ei, sd)
        if sd == C.SEED:
            # ★ 落盘主种子嵌入，05_c3_roles.py 直接复用。
            #   否则 05 会用自己那份实现再训一遍，两处代码可能悄悄分叉。
            #   同时写指纹（特征+边表的 md5），05 据此判断缓存是否仍匹配；
            #   若 01 改参数重跑，指纹变化会强制 05 重训，不会静默用旧嵌入。
            import hashlib
            _h = hashlib.md5()
            _h.update(np.ascontiguousarray(d["feat"]).tobytes())
            _h.update(np.ascontiguousarray(d["edge_index"]).tobytes())
            _fp = _h.hexdigest()[:12]
            np.save(C.RESULTS / f"emb_{token}_{scale}.npy", emb)
            (C.RESULTS / f"emb_{token}_{scale}.fp").write_text(_fp)
        a_core, _ = auc(emb, y_core)
        a_bal, _ = auc(emb, y_bal)
        pu = purity(emb, y_core)
        enr = pu / rnd if rnd else np.nan
        a_nof, _ = auc(train_emb(x_nofeat, ei, sd), y_core)
        rows.append((sd, a_core, a_bal, a_nof, pu, enr))
        print(f"    {sd:>6}{a_core:>13.4f}{a_bal:>13.4f}"
              f"{a_nof:>10.4f}{enr:>8.1f}x")

    arr = np.array([[r[1], r[2], r[3], r[4], r[5]] for r in rows])
    keys = ["gnn_is_core", "gnn_balance", "gnn_nofeat_is_core",
            "purity", "enrichment"]
    for j, kk in enumerate(keys):
        rec[kk] = {"mean": round(float(arr[:, j].mean()), 4),
                   "sd_seed": round(float(arr[:, j].std(ddof=1)), 4),
                   "min": round(float(arr[:, j].min()), 4),
                   "max": round(float(arr[:, j].max()), 4)}

    # ---------- 中心性的富集对照 ----------
    pu_c = purity(X_cent, y_core)
    pu_r = purity(x, y_core)
    rec["enrichment_centrality"] = round(pu_c / rnd, 1) if rnd else None
    rec["enrichment_raw_struct"] = round(pu_r / rnd, 1) if rnd else None
    rec["purity_centrality"] = round(float(pu_c), 4)
    rec["purity_raw_struct"] = round(float(pu_r), 4)
    rec["random_baseline_purity"] = round(float(rnd), 5)

    # ---------- node2vec 基线（多种子）----------
    # ★ 必须和 GNN 一样跑多种子：gensim 的 seed 参数不能完全锁住随机性
    #   （还依赖 PYTHONHASHSEED 与线程调度），实测同一 seed 两次运行相差约 0.01,
    #   而 LINK 上 GNN 与 node2vec 的差距本身就在这个量级。单点值会误导。
    try:
        from node2vec import Node2Vec
        import networkx as nx
        if scale != "1w":
            raise ImportError    # 大规模档跳过 node2vec（随机游走极慢）
        print(f"\n  [node2vec 基线] 种子 {list(C.SEEDS_N2V)}")
        Gx = nx.DiGraph()
        Gx.add_edges_from(d["edge_index"].T.tolist())
        n2 = []
        for sd in C.SEEDS_N2V:
            np.random.seed(sd)
            n2v = Node2Vec(Gx, dimensions=C.EMB_DIM, walk_length=80,
                           num_walks=10, workers=1, quiet=True, seed=sd)
            mdl = n2v.fit(window=10, min_count=1, seed=sd, workers=1)
            e2 = np.array([mdl.wv[str(i)] if str(i) in mdl.wv
                           else np.zeros(C.EMB_DIM)
                           for i in range(len(y_core))])
            a2, _ = auc(e2, y_core)
            p2 = purity(e2, y_core)
            n2.append((a2, p2 / rnd if rnd else np.nan))
            print(f"    seed {sd}: AUC {a2:.4f}  富集 {n2[-1][1]:.1f}x")
        n2a = np.array(n2)
        rec["node2vec_is_core"] = {
            "mean": round(float(n2a[:, 0].mean()), 4),
            "sd_seed": round(float(n2a[:, 0].std(ddof=1)), 4),
            "min": round(float(n2a[:, 0].min()), 4),
            "max": round(float(n2a[:, 0].max()), 4)}
        rec["node2vec_enrichment"] = {
            "mean": round(float(n2a[:, 1].mean()), 2),
            "sd_seed": round(float(n2a[:, 1].std(ddof=1)), 2)}
        print(f"    node2vec AUC {n2a[:, 0].mean():.4f} "
              f"± {n2a[:, 0].std(ddof=1):.4f}  |  富集 "
              f"{n2a[:, 1].mean():.1f}x ± {n2a[:, 1].std(ddof=1):.1f}")

        # ---------- 与 GNN 的配对比较 ----------
        g = arr[:, 0]
        d_mean = g.mean() - n2a[:, 0].mean()
        pooled = np.sqrt(g.var(ddof=1) / len(g)
                         + n2a[:, 0].var(ddof=1) / len(n2a))
        print(f"\n    [GNN vs node2vec] 差 {d_mean:+.4f}  "
              f"（合并标准误 {pooled:.4f} → {d_mean / pooled:+.2f}）")
        verdict = ("GNN 显著更优" if d_mean / pooled > 2 else
                   "node2vec 显著更优" if d_mean / pooled < -2 else
                   "两者无显著差异")
        print(f"    → {verdict}")
        print(f"    ★ node2vec 只用拓扑，GNN 额外吸收节点属性。"
              f"两者同属【学习表征】阵营，")
        print(f"      真正的对照是它们 vs 手工中心性"
              f"（{rec['is_core_centrality']['auc']:.3f}）。")
        rec["gnn_vs_node2vec"] = {"diff": round(float(d_mean), 4),
                                  "pooled_se": round(float(pooled), 4),
                                  "z": round(float(d_mean / pooled), 2),
                                  "verdict": verdict}
    except ImportError:
        msg = ("大规模档跳过（随机游走极慢，1万档已有对照）" if scale != "1w"
               else "未安装，跳过（pip install node2vec）")
        print(f"\n  [node2vec] {msg}")

    # ---------- 汇总 ----------
    cm.banner(f"{token}-{scale} 汇总（论文表格直接用）", "-")
    print(f"  centrality            {rec['is_core_centrality']['auc']:.3f}")
    print(f"  raw structural        {rec['is_core_raw_struct']['auc']:.3f}")
    print(f"  GNN                   {rec['gnn_is_core']['mean']:.3f} "
          f"± {rec['gnn_is_core']['sd_seed']:.3f}  (种子间)")
    print(f"  GNN (no features)     {rec['gnn_nofeat_is_core']['mean']:.3f} "
          f"± {rec['gnn_nofeat_is_core']['sd_seed']:.3f}")
    if isinstance(rec.get("node2vec_is_core"), dict):
        print(f"  node2vec              "
              f"{rec['node2vec_is_core']['mean']:.3f} "
              f"± {rec['node2vec_is_core']['sd_seed']:.3f}  (种子间)")
    # ★ 富集 = 纯度 / 随机基线，而基线随规模缩小。只报富集会让人误以为
    #   "规模越大表现越好"，实际绝对纯度在 64 万档是下降的。两个都要报。
    print(f"\n  {'方法':<14}{'邻域纯度':>10}{'富集倍数':>11}")
    print(f"  {'随机基线':<14}{rec['random_baseline_purity']:>10.5f}"
          f"{'1.0x':>11}")
    print(f"  {'centrality':<14}{rec['purity_centrality']:>10.4f}"
          f"{rec['enrichment_centrality']:>10.1f}x")
    print(f"  {'raw structural':<14}{rec['purity_raw_struct']:>10.4f}"
          f"{rec['enrichment_raw_struct']:>10.1f}x")
    print(f"  {'GNN':<14}{rec['purity']['mean']:>10.4f}"
          f"{rec['enrichment']['mean']:>10.1f}x")
    if "node2vec_enrichment" in rec:
        n2p = rec["node2vec_enrichment"]["mean"] * rec["random_baseline_purity"]
        print(f"  {'node2vec':<14}{n2p:>10.4f}"
              f"{rec['node2vec_enrichment']['mean']:>10.1f}x")
    ratio = rec["enrichment"]["mean"] / max(rec["enrichment_raw_struct"], 1e-9)
    print(f"  → GNN/raw 纯度比 {ratio:.2f}x"
          f"（该比值不受随机基线影响，可跨规模比较）")
    print(f"\n  ★ 这里的 ± 是【种子间】标准差，与 v16 表格里的【折间】sd "
          f"不是一回事，论文需分别说明")

    cm.save_json(rec, f"c1_{token}_{scale}.json")
    for kk in keys:
        cm.log_metric(SCRIPT, token, f"{scale}_{kk}", rec[kk]["mean"],
                      f"sd_seed={rec[kk]['sd_seed']}")
    return rec


def main():
    a = sys.argv[1:]
    if a:
        run(a[0], a[1] if len(a) > 1 else "1w")
    else:
        for t in C.TOKENS:
            run(t, "1w")
    cm.banner("完成。下一步：python 03_c2_weekly.py")


if __name__ == "__main__":
    main()
