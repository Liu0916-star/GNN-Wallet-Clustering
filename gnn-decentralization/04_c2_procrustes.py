# -*- coding: utf-8 -*-
"""
04_c2_procrustes.py —— C2 重量版：逐窗重训 + Procrustes 对齐

每个周窗独立建图、独立训练嵌入，再用共同锚点做正交 Procrustes 对齐到参考窗，
从而让不同窗口的嵌入可比。输出：对齐残差、共同锚点数、持续精英集、窗间漂移。

★ 相对 v16 的改动：
  1. 边权求和（common.build_graph）
  2. 逐窗嵌入落盘为 c2_embeddings_{TOKEN}.npz。v16 训完就丢，
     导致任何复查都要重跑 10–20 分钟。

用法：python 04_c2_procrustes.py [TOKEN]
"""
import sys

import numpy as np
import pandas as pd
import torch

import config as C
import common as cm

SCRIPT = "04_c2_procrustes"
ELITE_PCT = 95          # 精英 = PageRank top 5%


def _embed_window(g, seed=None):
    """对单窗建图、算特征、训嵌入。返回 {address: vector}。"""
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import SAGEConv
    from torch_geometric.utils import negative_sampling

    seed = C.SEED if seed is None else seed
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    G = cm.build_graph(g, top_n=None, largest_wcc=True, verbose=False)
    if G.number_of_nodes() < 50:
        return None, None
    feat, nodes = cm.node_features(G, {}, C.BETWEENNESS_K["c2_window"], seed)
    x, _ = cm.prepare_gnn_input(feat)
    idx = {a: i for i, a in enumerate(nodes)}
    ei = torch.tensor([[idx[u], idx[v]] for u, v in G.edges()],
                      dtype=torch.long).t().contiguous().to(dev)

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
        emb = m(xt, ei).cpu().numpy()

    pr_i = C.FEATURE_NAMES.index("pagerank")
    return dict(zip(nodes, emb)), dict(zip(nodes, feat[:, pr_i]))


def procrustes(A, B):
    """求正交 R 使 ||A R - B|| 最小。A/B 为同序锚点矩阵。"""
    U, _, Vt = np.linalg.svd(A.T @ B)
    return U @ Vt


def run(token):
    cm.banner(f"04 · C2 Procrustes {token}")
    tx = cm.load_transfers(token, drop_burn=True)
    outliers = cm.find_outlier_windows(tx)
    print(f"  周窗 {tx['week'].nunique()} | 异常窗 {outliers if outliers else '无'}")

    emb_by_w, pr_by_w = {}, {}
    for w, g in tx.groupby("week", sort=True):
        n_nodes = len(set(g["from"]) | set(g["to"]))
        # ★ 异常窗（LINK 的 W7 有 55 万节点）单窗训练极慢且易 OOM。
        #   它本身是钓鱼空投造成的一次性冲击，不代表常态路由结构，
        #   逐窗对齐分析本就应排除。跳过并在结果中显式记录。
        if int(w) in outliers and n_nodes > C.PROCRUSTES_MAX_NODES:
            print(f"    周{int(w) + 1}: 异常窗且 {n_nodes:,} 节点 "
                  f"> {C.PROCRUSTES_MAX_NODES:,}，跳过（附录须说明）")
            continue
        if n_nodes > C.PROCRUSTES_MAX_NODES:
            print(f"    周{int(w) + 1}: [!] {n_nodes:,} 节点，训练会很慢")
        e, pr = _embed_window(g)
        if e is None:
            print(f"    周{int(w) + 1}: 节点过少，跳过")
            continue
        emb_by_w[int(w)], pr_by_w[int(w)] = e, pr
        print(f"    周{int(w) + 1}: {len(e):,} 节点，嵌入完成", flush=True)

    ws = sorted(emb_by_w)
    ref = ws[0]
    aligned = {ref: emb_by_w[ref]}
    diag = []
    for w in ws[1:]:
        common_a = sorted(set(emb_by_w[ref]) & set(emb_by_w[w]))
        if len(common_a) < 50:
            print(f"    [!] 周{w + 1} 共同锚点仅 {len(common_a)}，跳过对齐")
            continue
        A = np.stack([emb_by_w[w][a] for a in common_a])
        B = np.stack([emb_by_w[ref][a] for a in common_a])
        before = float(np.linalg.norm(A - B))
        R = procrustes(A, B)
        after = float(np.linalg.norm(A @ R - B))
        aligned[w] = {a: v @ R for a, v in emb_by_w[w].items()}
        diag.append({"window": w + 1, "anchors": len(common_a),
                     "resid_before": round(before, 2),
                     "resid_after": round(after, 2),
                     "improved": after < before})

    dd = pd.DataFrame(diag)
    if dd.empty:
        print("\n  [X] 没有任何窗口完成对齐，检查数据或阈值")
        return
    ok = int(dd["improved"].sum())
    print(f"\n  [对齐] {ok}/{len(dd)} 成功 | 最少共同锚点 "
          f"{int(dd['anchors'].min()):,}")

    # ---------- 持续精英 ----------
    elite = {}
    for w in ws:
        v = np.array(list(pr_by_w[w].values()))
        thr = np.percentile(v, ELITE_PCT)
        elite[w] = {a for a, p in pr_by_w[w].items() if p >= thr}
    persistent = set.intersection(*elite.values()) if elite else set()
    print(f"  [精英] 全窗持续 {len(persistent):,} 个"
          f"（每窗 PageRank top {100 - ELITE_PCT}%）")

    # ---------- 窗间漂移 ----------
    drift = []
    for i in range(len(ws) - 1):
        a, b = ws[i], ws[i + 1]
        if a not in aligned or b not in aligned:
            continue
        shared = persistent & set(aligned[a]) & set(aligned[b])
        if not shared:
            continue
        d = np.mean([np.linalg.norm(aligned[b][x] - aligned[a][x])
                     for x in shared])
        drift.append({"from": a + 1, "to": b + 1, "drift": round(float(d), 4),
                      "is_outlier": a in outliers or b in outliers})
    dfd = pd.DataFrame(drift)
    normal = dfd[~dfd["is_outlier"]]["drift"].mean() if len(dfd) else np.nan
    print(f"  [漂移] 正常窗间均值 {normal:.4f}")
    if len(dfd) and dfd["is_outlier"].any():
        print(f"          含异常窗 {dfd[dfd['is_outlier']]['drift'].mean():.4f}")

    # ---------- 落盘（★ v16 从没存过嵌入）----------
    np.savez_compressed(
        C.RESULTS / f"c2_embeddings_{token}.npz",
        **{f"w{w}_addr": np.array(sorted(aligned[w]), dtype=object)
           for w in aligned},
        **{f"w{w}_emb": np.stack([aligned[w][a] for a in sorted(aligned[w])])
           for w in aligned})
    print(f"    → c2_embeddings_{token}.npz（v16 从未保存，复查不必再重跑）")

    print(f"\n  ★ 复现性：逐窗嵌入由 GPU 训练（MPS scatter 非确定性），")
    print(f"    持续精英集与漂移均值会随重训略有变化。已落盘对齐后的嵌入，")
    print(f"    归档时一并提交。论文报告这些数字时须注明单次运行。")

    cm.save_json({"token": token, "n_windows": len(ws),
                  "skipped_outlier_windows": [w + 1 for w in outliers],
                  "anomaly_windows": outliers, "ref_window": ref + 1,
                  "alignment_diag": diag,
                  "alignment_ok": f"{ok}/{len(dd)}",
                  "min_anchors": int(dd["anchors"].min()) if len(dd) else None,
                  "n_persistent_elite": len(persistent),
                  "elite_drift": drift,
                  "normal_drift_mean": round(float(normal), 4)},
                 f"c2_procrustes_{token}.json")
    cm.log_metric(SCRIPT, token, "persistent_elite", len(persistent))
    cm.log_metric(SCRIPT, token, "normal_drift_mean", round(float(normal), 4))


def main():
    a = sys.argv[1:]
    for t in (a if a else C.TOKENS):
        run(t)
    cm.banner("完成。下一步：python 05_c3_roles.py")


if __name__ == "__main__":
    main()
