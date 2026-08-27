# -*- coding: utf-8 -*-
"""
01_build_graph.py —— 建图 + 节点特征 + 标签，落盘供下游复用

v16 从没保存过 LINK 1万/5万档的特征，每次都要重建。这里一次性存好：
    results/graph_{TOKEN}_{SCALE}.npz     feat / nodes / cat / edge_index
下游脚本（02_c1 / 05_c3）直接读，不再重算 betweenness。

同时输出「新旧边权对比」：PageRank 相关系数与 top-50 重合度，
这两个数直接写进论文，用来量化边权修正的影响幅度。

用法：
    python 01_build_graph.py                  # LINK+UNI 的 1w
    python 01_build_graph.py LINK 5w
    python 01_build_graph.py LINK 64w         # 慢，betweenness k=5000
"""
import sys

import numpy as np
import pandas as pd

import config as C
import common as cm

SCRIPT = "01_build_graph"


def reuse_betweenness(token, scale, nodes):
    """复用 v16 已算好的 betweenness（k=5,000）。

    ★ 只有在节点集【完全一致】时才复用；否则返回 None 触发重算。
      betweenness 只依赖拓扑，与边权无关，所以边权修正不影响这一列 ——
      这正是可以复用的原因，也是必须校验节点集的原因。
    """
    entry = C.LEGACY_FEAT.get((token, scale))
    if not entry:
        return None
    fp, np_ = entry
    if not (fp.exists() and np_.exists()):
        print(f"    [i] 未找到旧特征 {fp.name}，betweenness 将重算")
        return None
    old_feat = np.load(fp)
    old_nodes = [str(x).strip().lower()
                 for x in pd.read_csv(np_)["address"]]
    if len(old_nodes) != len(nodes):
        print(f"    [!] 旧节点数 {len(old_nodes):,} != 本次 {len(nodes):,}"
              f" → 不复用，重算 betweenness")
        return None
    if set(old_nodes) != set(nodes):
        print(f"    [!] 节点集不一致（数量相同但成员不同）→ 不复用")
        return None
    bi = C.FEATURE_NAMES.index("betweenness")
    m = dict(zip(old_nodes, old_feat[:, bi]))
    print(f"    [复用] 旧 betweenness(k=5,000)，节点集完全一致 "
          f"({len(nodes):,})，跳过数小时的采样")
    print(f"           拓扑未变，边权修正不影响该列；"
          f"只重算 in_deg_w / out_deg_w / pagerank")
    return np.array([m[a] for a in nodes])


def legacy_compare(tx, G, nodes, pr_new, verbose=True):
    """用 v16 的错误建图方式重算一遍，量化差异。拓扑相同，只有边权不同。"""
    import networkx as nx
    Gl = nx.from_pandas_edgelist(tx, "from", "to", edge_attr="amt",
                                 create_using=nx.DiGraph())
    Gl = Gl.subgraph(nodes).copy()
    nx.set_edge_attributes(
        Gl, {(u, v): d["amt"] for u, v, d in Gl.edges(data=True)}, "w")
    pr_old = cm.safe_pagerank(Gl)

    v_new = sum(dict(G.in_degree(weight="w")).values())
    v_old = sum(dict(Gl.in_degree(weight="w")).values())
    a = np.array([pr_new[n] for n in nodes])
    b = np.array([pr_old.get(n, 0.0) for n in nodes])
    rho = float(np.corrcoef(a, b)[0, 1])
    t50a = set(np.array(nodes)[np.argsort(-a)[:50]])
    t50b = set(np.array(nodes)[np.argsort(-b)[:50]])
    ov = len(t50a & t50b)
    loss = (1 - v_old / v_new) * 100 if v_new else 0.0
    if verbose:
        print(f"\n  [新旧边权对比]")
        print(f"    图内总流量 旧 {v_old:,.0f} → 新 {v_new:,.0f}"
              f"（旧口径丢失 {loss:.1f}%）")
        print(f"    PageRank 相关系数 {rho:.4f} | top-50 重合 {ov}/50")
        print(f"    ★ 这两个数写进论文，说明修正的影响幅度")
    return {"value_loss_pct": round(loss, 2), "pagerank_corr": round(rho, 4),
            "top50_overlap": ov}


def run(token: str, scale: str):
    cm.banner(f"01 · 建图 {token} / {scale}")
    top_n = C.SCALES[scale]

    tx = cm.load_transfers(token, drop_burn=True)
    bal = cm.load_balances(token)
    bmap = dict(zip(bal["address"], bal["balance"]))
    print(f"  转账 {len(tx):,} | 持币 {len(bal):,}")

    G = cm.build_graph(tx, top_n=None if scale == "64w" else top_n)
    exp_n = C.EXPECT_NODES.get((token, scale))
    if exp_n:
        cm.check("节点数", G.number_of_nodes(), exp_n)

    k = C.BETWEENNESS_K[scale]
    nodes_pre = list(G.nodes())
    reused = None if "--no-reuse" in sys.argv else \
        reuse_betweenness(token, scale, nodes_pre)
    if reused is not None:
        print(f"  计算特征（betweenness 复用）...")
        feat, nodes = cm.node_features(G, bmap, k_betweenness=1,
                                       skip_betweenness=True)
        feat[:, C.FEATURE_NAMES.index("betweenness")] = reused
    else:
        print(f"  计算特征（betweenness k={k:,}，"
              f"{'可能需要数小时' if scale == '64w' else '1–2 分钟'}）...")
        feat, nodes = cm.node_features(G, bmap, k_betweenness=k)

    pr_i = C.FEATURE_NAMES.index("pagerank")
    pr_new = dict(zip(nodes, feat[:, pr_i]))
    # 64w 档要再建一张 641k 节点的图并跑第二次 PageRank，内存与时间都翻倍。
    # 1w/5w 的对比结论已足以说明修正幅度，64w 默认跳过（需要时传 --legacy）。
    if scale == "64w" and "--legacy" not in sys.argv:
        print("\n  [新旧边权对比] 64w 档默认跳过（加 --legacy 强制执行）")
        cmp = {}
    else:
        nodeset = set(nodes)
        cmp = legacy_compare(
            tx[tx["from"].isin(nodeset) & tx["to"].isin(nodeset)],
            G, nodes, pr_new)

    lm = cm.load_labels()
    cat = cm.node_categories(nodes, lm)
    n_core = int(cm.is_core_mask(cat).sum())
    print(f"\n  [标签] 匹配 {int((cat != 'Unlabeled').sum()):,} | "
          f"核心锚点 {n_core}")
    exp_a = C.EXPECT_ANCHORS.get((token, scale))
    if exp_a:
        cm.check("is_core 锚点", n_core, exp_a)
    print(f"    分布 {pd.Series(cat).value_counts().to_dict()}")

    idx = {a: i for i, a in enumerate(nodes)}
    edges = np.array([[idx[u], idx[v]] for u, v in G.edges()], dtype=np.int64).T

    out = C.RESULTS / f"graph_{token}_{scale}.npz"
    np.savez_compressed(out, feat=feat, nodes=np.array(nodes, dtype=object),
                        cat=cat, edge_index=edges,
                        feature_names=np.array(C.FEATURE_NAMES, dtype=object),
                        meta=np.array([token, scale, str(k),
                                       str(G.number_of_nodes()),
                                       str(G.number_of_edges())], dtype=object))
    print(f"\n  → {out.name}  feat{feat.shape} edges{edges.shape}")

    for k_, v_ in cmp.items():
        cm.log_metric(SCRIPT, token, f"{scale}_{k_}", v_)
    cm.log_metric(SCRIPT, token, f"{scale}_n_nodes", G.number_of_nodes())
    cm.log_metric(SCRIPT, token, f"{scale}_n_anchors", n_core)
    return feat, nodes, cat, edges


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args:
        run(args[0], args[1] if len(args) > 1 else "1w")
    else:
        for t in C.TOKENS:
            run(t, "1w")
    cm.banner("完成。下一步：python 02_c1.py")


if __name__ == "__main__":
    main()
