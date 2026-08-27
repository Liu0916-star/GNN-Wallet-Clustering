# -*- coding: utf-8 -*-
"""
common.py —— 唯一真理来源

所有加载、指标、建图、标签逻辑都在这里。脚本不允许自己实现这些。

三条硬规则（针对已知的三个坑）：
  1. build_graph() 是建图的唯一入口，边权求和写死在里面。
     v16 有 13 处 from_pandas_edgelist，其中每一处都因为 DiGraph 只保留
     最后一条边而丢掉了 LINK 48.8% / UNI 59.6% 的转账金额。
  2. 列名一律按名字识别，禁用 iloc 取列。
     v16 栽过两次：UNI 余额 2 列 / LINK 4 列，iloc[:,1] 在 LINK 上取到
     total_received，算出 3,918,660 个地址（真值 885,011）。
  3. load_labels() 只读本地快照，联网直接报错。
     标签库在漂移，is_core 锚点会静默改变。
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config as C

# ============================================================ 列名识别
_CANDIDATES = {
    "from": ["from", "from_address", "sender", "evt_from", "src", "from_addr"],
    "to": ["to", "to_address", "receiver", "recipient", "evt_to", "dst"],
    "value": ["value", "value_link", "value_uni", "amount", "raw_value",
              "quantity", "val"],
    "time": ["block_timestamp", "evt_block_time", "timestamp", "block_time",
             "evt_block_timestamp", "time", "date"],
    "address": ["address", "addr", "holder", "wallet", "account", "owner"],
    "balance": ["balance", "balance_link", "balance_uni", "token_balance"],
}


def find_col(df: pd.DataFrame, role: str) -> str:
    """按列名识别。找不到就抛错并打印真实列名 —— 绝不静默回退到 iloc。"""
    norm = {str(c).strip().lower().lstrip("\ufeff"): c for c in df.columns}
    for cand in _CANDIDATES[role]:
        if cand in norm:
            return norm[cand]
    raise KeyError(
        f"找不到 '{role}' 对应的列。实际列名: {list(df.columns)}\n"
        f"候选: {_CANDIDATES[role]}\n"
        f"（如果这是新的列名写法，加进 common._CANDIDATES，不要改用 iloc）"
    )


def _norm_addr(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()


# ============================================================ 加载
def load_transfers(token: str, drop_burn: bool = True) -> pd.DataFrame:
    """返回 from / to / amt / ts / week。amt 已 /1e18。

    drop_burn=True 时从收发两端剔除零地址与销毁地址。
    """
    path = C.FILES[token]["tx"]
    raw = pd.read_csv(path, low_memory=False)
    c_f, c_t = find_col(raw, "from"), find_col(raw, "to")
    c_v, c_ts = find_col(raw, "value"), find_col(raw, "time")

    df = pd.DataFrame({
        "from": _norm_addr(raw[c_f]),
        "to": _norm_addr(raw[c_t]),
        "amt": pd.to_numeric(raw[c_v], errors="coerce") / 1e18,
        "ts": pd.to_datetime(raw[c_ts], utc=True, errors="coerce",
                             format="mixed"),
    }).dropna(subset=["amt", "ts"])

    if drop_burn:
        df = df[~df["from"].isin(C.BURN_ADDRS) & ~df["to"].isin(C.BURN_ADDRS)]

    t0 = pd.Timestamp(C.WINDOW_START)
    df["week"] = ((df["ts"] - t0).dt.days // C.WINDOW_DAYS).astype(int)
    return df.reset_index(drop=True)


def load_balances(token: str) -> pd.DataFrame:
    """返回 address / balance。自动识别 wei 并 /1e18。只保留正余额。"""
    raw = pd.read_csv(C.FILES[token]["bal"], low_memory=False)
    c_a, c_b = find_col(raw, "address"), find_col(raw, "balance")
    df = pd.DataFrame({
        "address": _norm_addr(raw[c_a]),
        "balance": pd.to_numeric(raw[c_b], errors="coerce"),
    }).dropna()
    df = df[df["balance"] > 0]
    if df["balance"].median() > 1e12:
        df["balance"] = df["balance"] / 1e18
    return df.reset_index(drop=True)


def active_addresses(tx: pd.DataFrame, drop_zero_value: bool = None) -> set:
    if drop_zero_value is None:
        drop_zero_value = C.DROP_ZERO_VALUE_FOR_ACTIVE
    d = tx[tx["amt"] > 0] if drop_zero_value else tx
    return set(d["from"]) | set(d["to"])


# ============================================================ 标签
def load_labels() -> dict:
    """只读本地快照。文件不存在时报错并给出抓取命令，绝不联网。"""
    snap = C.LABEL_SNAPSHOT
    if not snap.exists():
        # 自动发现 labels/ 下任何一份快照，避免"文件名与 config 对不上"这类阻塞
        found = sorted(C.LABELS.glob("etherscan_labels_*.json"))
        if len(found) == 1:
            snap = found[0]
            print(f"    [i] config 指向 {C.LABEL_SNAPSHOT.name} 不存在，"
                  f"改用 {snap.name}")
        elif len(found) > 1:
            snap = found[-1]
            print(f"    [!] labels/ 下有 {len(found)} 份快照，"
                  f"自动选用最新的 {snap.name}。为可复现性请只保留一份，"
                  f"并在 config.LABEL_SNAPSHOT 里写死。")
    if not snap.exists():
        raise FileNotFoundError(
            f"标签快照不存在: {C.LABEL_SNAPSHOT}\n"
            f"抓取一次并存进仓库（之后永远读本地）：\n"
            f"  python -c \"import urllib.request,shutil;"
            f"shutil.copyfileobj(urllib.request.urlopen('{C.LABEL_SNAPSHOT_URL}'),"
            f"open('{C.LABEL_SNAPSHOT}','wb'))\"\n"
            f"★ 标签库持续更新，is_core 锚点随之漂移。快照必须提交进仓库，"
            f"论文需注明抓取日期。"
        )
    with open(snap, encoding="utf-8") as f:
        raw = json.load(f)
    return {a.lower(): v for a, v in raw.items()}


def categorize(labels) -> str:
    labs = {str(x).lower() for x in labels}
    for cat, keys in C.LABEL_CATEGORIES.items():
        if labs & keys:
            return cat
    return "Other-labeled"


def node_categories(nodes, label_map: dict) -> np.ndarray:
    return np.array([categorize(label_map[a]["labels"]) if a in label_map
                     else "Unlabeled" for a in nodes])


def is_core_mask(cat: np.ndarray) -> np.ndarray:
    return np.isin(cat, C.CORE_CATS)


# ============================================================ 建图（唯一入口）
def build_graph(tx: pd.DataFrame, top_n: int | None = None,
                largest_wcc: bool = True, verbose: bool = True):
    """★ 建图的唯一入口。边权按 (from,to) 求和 —— 这是 v16 最大的 bug 所在。

    tx 需含 from / to / amt。top_n=None 表示不做活跃度筛选（全图）。
    返回 networkx.DiGraph，边属性名为 'w'。
    """
    import networkx as nx

    if top_n is not None:
        activity = pd.concat([tx["from"], tx["to"]]).value_counts()
        core = set(activity.head(top_n).index)
        tx = tx[tx["from"].isin(core) & tx["to"].isin(core)]
        if verbose:
            print(f"    活跃度 top{top_n:,} → 核心集 {len(core):,}，"
                  f"内部转账 {len(tx):,}")

    agg = tx.groupby(["from", "to"], as_index=False)["amt"].sum()
    agg = agg.rename(columns={"amt": "w"})
    if verbose and len(tx):
        dup = len(tx) - len(agg)
        print(f"    去重边 {len(agg):,}（{dup:,} 行重复边，"
              f"占 {dup / len(tx) * 100:.1f}%）")

    G = nx.from_pandas_edgelist(agg, "from", "to", edge_attr="w",
                                create_using=nx.DiGraph())

    if largest_wcc and G.number_of_nodes():
        G = G.subgraph(max(nx.weakly_connected_components(G), key=len)).copy()
    if verbose:
        print(f"    最终图 {G.number_of_nodes():,} 节点 / "
              f"{G.number_of_edges():,} 边")
    return G


def safe_pagerank(G, weight="w"):
    """修正边权后权重更极端，默认 max_iter=100 可能不收敛。逐级放宽并如实报告。"""
    import networkx as nx
    for mi, tol in ((100, 1e-6), (500, 1e-6), (2000, 1e-8)):
        try:
            pr = nx.pagerank(G, weight=weight, max_iter=mi, tol=tol)
            if mi > 100:
                print(f"    [!] PageRank 在 max_iter={mi} 才收敛"
                      f"（论文用默认 100），需在方法论注明")
            return pr
        except nx.PowerIterationFailedConvergence:
            continue
    raise RuntimeError("PageRank 不收敛")


def node_features(G, balance_map: dict, k_betweenness: int, seed: int = None,
                  skip_betweenness: bool = False):
    """返回 (feat, nodes)。feat 列顺序 = config.FEATURE_NAMES。

    skip_betweenness=True 时该列填 0，由调用方写入复用值（见 01 的
    reuse_betweenness）。betweenness 只依赖拓扑，与边权无关。
    """
    import networkx as nx
    seed = C.SEED if seed is None else seed
    nodes = list(G.nodes())
    ind, outd = dict(G.in_degree()), dict(G.out_degree())
    indw = dict(G.in_degree(weight="w"))
    outdw = dict(G.out_degree(weight="w"))
    pr = safe_pagerank(G)
    if skip_betweenness:
        bet = {n: 0.0 for n in nodes}
    else:
        k = min(k_betweenness, G.number_of_nodes())
        bet = nx.betweenness_centrality(G, k=k, seed=seed)
    cp = {n: len(set(G.predecessors(n)) | set(G.successors(n))) for n in nodes}

    feat = np.array([[balance_map.get(n, 0.0), ind[n], outd[n], indw[n],
                      outdw[n], cp[n], pr[n], bet[n]] for n in nodes],
                    dtype=np.float64)
    return feat, nodes


def prepare_gnn_input(feat: np.ndarray):
    """★ 防泄漏顺序：先剔除 balance，再对剩余 7 维做 log1p + z-score。

    顺序反了会让标准化基准被 balance 污染。返回 (x, kept_names)。
    """
    names = C.FEATURE_NAMES
    bi = names.index(C.LEAKY_FEATURE)
    keep = [i for i in range(len(names)) if i != bi]
    raw = feat[:, keep]
    lg = np.log1p(np.abs(raw)) * np.sign(raw)
    x = (lg - lg.mean(0)) / (lg.std(0) + 1e-8)
    return x, [names[i] for i in keep]


# ============================================================ 指标
def gini(values) -> float:
    x = np.sort(np.asarray(values, dtype=float))
    x = x[x > 0]
    n = len(x)
    if n == 0:
        return np.nan
    c = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(c) / c[-1]) / n)


def hhi(values) -> float:
    """0–10,000 量表。DOJ: <1500 不集中，1500–2500 中度，>2500 高度。"""
    v = np.asarray(values, dtype=float)
    v = v[v > 0]
    if len(v) == 0 or v.sum() <= 0:
        return np.nan
    s = v / v.sum()
    return float(np.sum((s * 100) ** 2))


def top_share(values, frac: float = None, k: int = None,
              include_zeros: bool = False) -> float:
    """★ include_zeros 决定"1%"的分母。

    v16 的 cell 18 对 in_degree() 的全部返回值（含入流为 0 的纯发送方）
    取 int(len*0.01)，所以 LINK 得到 82.3%；只对正值取 ceil 则是 79.4%。
    两者都"对"，但必须说清楚是哪一种。本仓库主口径 include_zeros=False。
    """
    x = np.asarray(values, dtype=float)
    if not include_zeros:
        x = x[x > 0]
    if len(x) == 0:
        return np.nan
    x = np.sort(x)[::-1]
    n = k if k is not None else max(1, int(np.ceil(len(x) * frac)))
    return float(x[:n].sum() / max(x.sum(), 1e-12))


def top_share_v16(values, frac: float = 0.01) -> float:
    """精确复现 v16 cell 18 的实现，只用于修正前后对比，不用于新结果。

    差别有二：不过滤零值（in_degree 会返回纯发送方的 0），且用 int 截断
    而非 ceil 进位。LINK 上这两点合起来把 79.4% 抬到了 82.3%。
    """
    sv = np.sort(np.asarray(values, dtype=float))[::-1]
    if len(sv) == 0:
        return np.nan
    return float(sv[:max(1, int(len(sv) * frac))].sum() / max(sv.sum(), 1e-9))


def nakamoto(values, threshold_frac: float = 0.5) -> int:
    """达到 threshold_frac 份额所需的最少参与者数。治理层用。"""
    x = np.sort(np.asarray(values, dtype=float))[::-1]
    x = x[x > 0]
    if len(x) == 0:
        return 0
    return int(np.searchsorted(np.cumsum(x) / x.sum(), threshold_frac) + 1)


def nakamoto_absolute(values, threshold: float) -> int:
    """达到绝对阈值（如 40M UNI 法定人数）所需的最少地址数。"""
    x = np.sort(np.asarray(values, dtype=float))[::-1]
    x = x[x > 0]
    cum = np.cumsum(x)
    if len(cum) == 0 or cum[-1] < threshold:
        return -1                       # 全部加起来也不够
    return int(np.searchsorted(cum, threshold) + 1)


def weekly_metric(tx: pd.DataFrame, side: str, fn) -> pd.Series:
    """按周分组，对 side（'from'/'to'）侧的聚合金额施加 fn。"""
    out = {}
    for w, g in tx.groupby("week", sort=True):
        out[int(w)] = fn(g.groupby(side)["amt"].sum().values)
    return pd.Series(out, name=getattr(fn, "__name__", "metric"))


def find_outlier_windows(tx: pd.DataFrame) -> list:
    per = pd.concat([
        tx[["week", "from"]].rename(columns={"from": "a"}),
        tx[["week", "to"]].rename(columns={"to": "a"}),
    ]).groupby("week")["a"].nunique()
    med = per.median()
    return sorted(per[per > C.OUTLIER_NODE_MULTIPLIER * med].index.tolist())


# ============================================================ 实体归并
_TAIL = None


def _family(name: str):
    """把 Etherscan name 归并到实体族：'Binance 14' → 'Binance'。"""
    global _TAIL
    if _TAIL is None:
        import re
        _TAIL = re.compile(
            r"\s+(v\d+(\.\d+)?|\d+|#\d+|deposit|hot\s*wallet|cold\s*wallet)\s*$",
            re.I)
    if not name or not str(name).strip():
        return None
    head = str(name).split(":")[0].split("：")[0].strip()
    prev = None
    while prev != head:
        prev = head
        head = _TAIL.sub("", head).strip()
    low = head.lower()
    if len(head) < 2 or low in C.GENERIC_LABELS:
        return None
    if any(k in low for k in C.ALERT_KEYWORDS):
        return None
    return head


def build_entity_map(token: str, label_map: dict = None,
                     verbose: bool = True) -> dict:
    """三层来源，优先级由低到高：标签库 → 发行方储备 → 人工核实。"""
    if label_map is None:
        label_map = load_labels()
    emap = {}
    for a, info in label_map.items():
        fam = _family(info.get("name", ""))
        if fam:
            emap[a] = fam
    n_lib = len(emap)
    for a in C.ISSUER_RESERVES.get(token, ()):
        emap[a] = f"{token} Issuer Reserve"
    emap.update({k.lower(): v for k, v in C.MANUAL_ENTITIES.items()})
    if verbose:
        print(f"    实体映射 {len(emap):,} 个地址 → {len(set(emap.values())):,} "
              f"个实体（标签库 {n_lib:,} / 人工 {len(C.MANUAL_ENTITIES)}）")
    return emap


def entity_key(addrs: pd.Series, emap: dict) -> pd.Series:
    """未识别的地址保持独立 —— 合并只能做到部分，结果是下界不是上界。"""
    return addrs.map(emap).fillna(addrs)


# ============================================================ 落盘 / 溯源
def file_md5(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def log_metric(script: str, token: str, metric: str, value,
               note: str = "") -> None:
    """把关键数字追加到 results/manifest.csv，论文里每个数都能一行溯源。"""
    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "script": script, "token": token, "metric": metric,
        "value": value, "note": note,
    }
    path = C.RESULTS / "manifest.csv"
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(),
                               index=False)


def save_json(obj, name: str) -> Path:
    path = C.RESULTS / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    print(f"    → {path.name}")
    return path


def save_csv(df: pd.DataFrame, name: str) -> Path:
    path = C.RESULTS / name
    df.to_csv(path, index=False)
    print(f"    → {path.name} ({len(df)} 行)")
    return path


def banner(text: str, ch: str = "=") -> None:
    print("\n" + ch * 76)
    print(f"  {text}")
    print(ch * 76)


def check(label: str, got, expect, tol=0) -> bool:
    """闸门：对不上就醒目报警，但不中断（让整批跑完再看）。"""
    ok = (abs(got - expect) <= tol) if isinstance(got, (int, float)) \
        else (got == expect)
    mark = "[OK]" if ok else "[X] "
    print(f"    {mark} {label}: {got:,} (论文 {expect:,})" if
          isinstance(got, (int, float)) else
          f"    {mark} {label}: {got} (论文 {expect})")
    return ok
