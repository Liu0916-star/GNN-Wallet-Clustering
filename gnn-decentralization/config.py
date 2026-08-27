# -*- coding: utf-8 -*-
"""
config.py —— 全仓库唯一的配置来源

任何常量、路径、超参、口径定义都写在这里。脚本里不允许出现硬编码的
魔法数字或路径。要改口径，改这个文件，然后重跑，不要去脚本里翻。
"""
from pathlib import Path

# ============================================================ 路径
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
LABELS = ROOT / "labels"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
LEGACY = ROOT / "results_v16_legacy"      # v16 旧结果，只读，用于修正前后对比

for _d in (RESULTS, FIGURES, LABELS):
    _d.mkdir(exist_ok=True)

# ============================================================ 数据文件
TOKENS = ("LINK", "UNI")

FILES = {
    "LINK": {"tx": DATA / "link_90d_transfers.csv",
             "bal": DATA / "link_balances.csv"},
    "UNI":  {"tx": DATA / "uni_90d_transfers.csv",
             "bal": DATA / "uni_balances.csv"},
}

# 论文声明值，00_data_check.py 用它做闸门
EXPECT = {
    "LINK": {"n_tx": 1_233_497, "n_active": 642_126, "n_holders": 885_011,
             "contract": "0x514910771af9ca656af840dff83e8264ecf986ca"},
    "UNI":  {"n_tx": 433_856, "n_active": 38_667, "n_holders": 386_029,
             "contract": "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984"},
}

WINDOW_START = "2026-03-26 00:00:00+00:00"
WINDOW_END = "2026-06-25 00:00:00+00:00"      # 左闭右开
WINDOW_DAYS = 7
N_WINDOWS = 13

# ============================================================ 标签快照
# ★ 只读本地文件。common.load_labels() 联网会直接报错。
# 原因：brianleect/etherscan-labels 持续更新（8/5 = 29,945 条，8/9 = 29,772 条），
# is_core 锚点随之漂移，且不会报错 —— C1 的全部 AUC 会静默改变。
LABEL_SNAPSHOT = LABELS / "etherscan_labels_2026-08-26.json"
LABEL_SNAPSHOT_URL = ("https://raw.githubusercontent.com/brianleect/"
                      "etherscan-labels/main/data/etherscan/combined/"
                      "combinedAllLabels.json")

LABEL_CATEGORIES = {
    "CEX": {"exchange", "binance", "coinbase", "kraken", "okx", "huobi",
            "bitfinex", "gate.io", "kucoin", "crypto-com", "bybit"},
    "DEX": {"uniswap", "sushiswap", "balancer", "curve", "dex", "1inch",
            "0x-protocol", "bancor", "kyber"},
    "Bridge": {"bridge", "arbitrum", "optimism", "polygon", "wormhole",
               "multichain"},
    "MEV": {"mev-bot", "mev", "flashbots"},
    "DeFi": {"defi", "lending", "aave", "compound", "maker"},
    "Malicious": {"phish-hack", "phishing", "scam", "exploit", "heist"},
}
CORE_CATS = ("CEX", "DEX", "Bridge", "DeFi")      # is_core 的正类

# ============================================================ 建图
SCALES = {"1w": 10_000, "5w": 50_000, "64w": 640_000}

EXPECT_NODES = {("LINK", "1w"): 9_876, ("UNI", "1w"): 9_765,
                ("LINK", "5w"): 49_587, ("LINK", "64w"): 641_278}
EXPECT_ANCHORS = {("LINK", "1w"): 74, ("UNI", "1w"): 57,
                  ("LINK", "5w"): 86, ("LINK", "64w"): 161}

# betweenness 采样数，按规模分档（C2 逐窗单列，论文原来漏写了这一档）
BETWEENNESS_K = {"1w": 500, "5w": 500, "64w": 5_000, "c2_window": 200}

# ★ betweenness 只依赖拓扑，与边权无关。64万档的 k=5,000 采样极贵（数小时），
#   而 v16 已经算好并存在 featF_fixed_k5000.npy 里。只要节点集完全一致，
#   就可以复用那一列，只重算受边权影响的 in_deg_w / out_deg_w / pagerank。
#   节点集不一致时自动退回重算，绝不静默复用。
LEGACY_FEAT = {("LINK", "64w"): (LEGACY / "featF_fixed_k5000.npy",
                                 LEGACY / "nodesF.csv")}

FEATURE_NAMES = ["balance", "in_deg", "out_deg", "in_deg_w", "out_deg_w",
                 "counterparties", "pagerank", "betweenness"]
# ★ 防泄漏：balance 既是输入又是目标会泄漏。必须先从原始特征剔除，
#   再对剩余 7 维做 log1p + z-score。顺序反了会让基准被 balance 污染。
LEAKY_FEATURE = "balance"

# ============================================================ 模型超参
SEED = 42
SEEDS = tuple(range(42, 52))      # GNN 多种子稳健性（10 个）
# 大规模档单次训练很贵，按规模降种子数。1万档保持 10 个。
SEEDS_BY_SCALE = {"1w": tuple(range(42, 52)), "5w": (42, 43, 44, 45, 46),
                  "64w": (42, 43, 44)}
# node2vec 也必须多种子：gensim 的 seed 不能完全锁住随机性，实测同 seed
# 两次运行相差约 0.01，而 LINK 上 GNN 与 node2vec 的差距正在这个量级。
# 它比 GNN 慢，5 个种子足够给出区间。
SEEDS_N2V = (42, 43, 44, 45, 46)
EMB_DIM = 16
HIDDEN_DIM = 32
EPOCHS = 200
LR = 0.01
CV_FOLDS = 5
CV_SHUFFLE = True                 # ★ 正样本聚集，不 shuffle 会让 AUC 系统性 < 0.5
BOOTSTRAP_B = 1_000
PURITY_K = 10
N_CLUSTERS = 8

# ============================================================ 口径（★ 集中管理）
# 人群
POP_ALL_HOLDERS = "all_holders"       # 全部正余额地址   ← 主表口径（决策 D2=A）
POP_ACTIVE = "active"                 # 窗内有转账的地址 ← 附录敏感性
MAIN_POPULATION = POP_ALL_HOLDERS

# 方向：flow 指标一律以发送方为主口径，接收方进敏感性表
MAIN_DIRECTION = "out"

# 时间聚合：周度算再平均。全窗汇总进敏感性表
# （两者差异很大：LINK 421.2 vs 229.8，1.83 倍，必须在方法论交代）
MAIN_AGGREGATION = "weekly"

# 异常窗：主口径保留，剔除后单列一行
# （LINK 的 W7 剔除后 Flow HHI 从 421.2 升到 437.8 —— 因为 W7 本身低于均值）
EXCLUDE_OUTLIER_WINDOWS = False
OUTLIER_NODE_MULTIPLIER = 3           # 周内地址数 > 中位数 ×3 判为异常窗

# 零地址：收发两端都剔除（mint/burn 不是市场行为）
ZERO_ADDR = "0x0000000000000000000000000000000000000000"
DEAD_ADDR = "0x000000000000000000000000000000000000dead"
BURN_ADDRS = frozenset({ZERO_ADDR, DEAD_ADDR})

# 零值转账：不影响 HHI（会被丢弃），但会进入"活跃地址"集合
# LINK 26,311 条 / UNI 2,376 条；去掉后活跃地址 -1.8% / -3.9%
DROP_ZERO_VALUE_FOR_ACTIVE = False    # 主口径保留，敏感性里给另一个数

# ============================================================ 已核实的实体
# 发行方 / 协议控制的非流通储备（Etherscan 公开标签，可复现）
ISSUER_RESERVES = {
    "LINK": frozenset({          # Chainlink: Noncirculating Supply，7 × 30,000,000
        "0x0dffd343c2d3460a7ead2797a687304beb394ce0",
        "0x5a8e77bc30948cc9a51ae4e042d96e145648bb4c",
        "0xe0b66bfc7344a80152bfec954942e2926a6fca80",
        "0x9bbb46637a1df7cadec2afca19c2920cddcc8db8",
        "0x8652fb672253607c0061677bdcafb77a324de081",
        "0x7594eb0ca0a7f313befd59afe9e95c2201a443e4",
        "0x76287e0f7b107d1c9f8f01d5afac314ea8461a04",
    }),
    "UNI": frozenset({
        "0x1a9c8182c09f50c8318d769245bea52c32be35bc",   # UNI Timelock
        "0x090d4613473dee047c3f2706764f49e0821d256e",   # UNI Token Distributor
    }),
}

# 本会话在 Etherscan / Arkham 上逐个核实过的（最高优先级，覆盖标签库）
MANUAL_ENTITIES = {
    "0x28c6c06298d514db089934071355e5743bf21d60": "Binance",     # Binance 14
    "0xf977814e90da44bfa03b6295a0616a897441acec": "Binance",     # Hot Wallet 20
    "0x5a52e96bacdabb82fd05763e25335261b270efcb": "Binance",     # Binance 28
    "0x51c72848c68a965f66fa7a88855f9f7784502a7f": "Wintermute",  # Arkham; Arbiscan: Market Maker
    "0xbc10f2e862ed4502144c7d632a3459f49dfcdb5e": "Chainlink Staking Pool",
    ZERO_ADDR: "Burn",
    DEAD_ADDR: "Burn",
}

# 实体归并时，这些名字太通用，不能当作实体（两个 MEV Bot 不是同一家）
GENERIC_LABELS = frozenset({
    "mev bot", "mev", "bot", "deployer", "contract deployer", "null address",
    "fake_phishing", "phishing", "unlabelled contract", "unlabelled",
    "unlabelled routing contract", "compromised contract", "eoa", "wallet",
    "token", "contract", "unknown", "suspected",
})
# 告警类标签打在受害地址上（如 Multichain Hack Alert 覆盖 2,923 个地址），
# 合并进去会凭空造出一个巨型参与者
ALERT_KEYWORDS = ("hack alert", "alert", "phish", "exploit", "scam",
                  "victim", "hunter", "blocked", "heist")

# ============================================================ 隐藏经纪人
# 定义：is_core 分类器概率 top1% & 余额 ≤ 中位数 & 无标签
BROKER_PROB_PCT = 99          # top 1%
# ★ 新增下限：v16 的定义会选进 counterparties=1、betweenness=0 的退化节点
#   （UNI 29 个里有 5 个，含两对 vanity 孪生地址）。论文称其"卡在关键路径上"，
#   而 betweenness=0 直接反驳该描述。
BROKER_MIN_COUNTERPARTIES = 5
BROKER_REQUIRE_POSITIVE_BETWEENNESS = True

# ★ 多种子共识 —— 解决单次运行不可复现的问题
# MPS 上 GraphSAGE 的 scatter 聚合是非确定性的（GPU 原子加法顺序不固定），
# torch.manual_seed 锁不住。实测同一 seed 两次运行：
#     LINK 经纪人 39 → 34（13%）、UNI 27 → 23（15%）、跨币交集 20 → 15（25%）
# 因此不能报告单次运行的集合。改为：每个种子各筛一次，
# 只保留出现在 ≥ BROKER_CONSENSUS_MIN_FRAC 比例种子中的节点。
# 副产物：出现频次本身就是稳定性度量，可作为附录证据。
BROKER_CONSENSUS_SEEDS = tuple(range(42, 52))   # 与 C1 同一组种子
BROKER_CONSENSUS_MIN_FRAC = 0.5                 # 投票口径：至少半数种子中出现
BROKER_CORE_SET_FRAC = 0.9                      # "高置信"子集的门槛
N2V_CONSENSUS_N = 5                             # node2vec 侧共识种子数（较慢）

# 主口径的集成方式：
#   "rank"  先把各种子的百分位【排名】平均，再切一次 top1%（标准 ensemble，
#           免受各种子概率尺度差异影响，边界抖动最小）—— 推荐
#   "vote"  各种子先切 top1% 再投票（易受硬阈值边界影响，仅作稳定性诊断）
BROKER_ENSEMBLE = "rank"

# betweenness 对照必须与 GNN 集合【等大小】才公平：
# GNN 侧经过集成/共识收紧，betweenness 侧若仍取固定百分位，两边口径不同，
# "GNN 多找出 N 个" 就没有意义。True = 取同样大小的 top-K。
BROKER_SIZE_MATCHED_BASELINE = True

# 04_c2_procrustes：单窗节点数上限。LINK 的 W7（钓鱼空投）有 55 万节点，
# 逐窗训练 GNN + k=200 betweenness 极慢且易 OOM；该窗本就不代表常态路由结构。
PROCRUSTES_MAX_NODES = 100_000

# 03_c2_weekly：精英集大小。v16 用 top-50（周际留存 53%/48%、
# 流量份额均值 0.748），但那两个数出自已废弃脚本且基于错误边权，此处重算。
ELITE_TOP_K = 50

# ============================================================ 治理层（UNI）
# 扩 C2 为三层：所有权 / 路由 / 规则。不新增 C4，不造新词。
GOV_PROPOSAL_THRESHOLD = 2_500_000        # 2.5M UNI = 0.25%
GOV_QUORUM = 40_000_000                   # 40M UNI = 4%
# 不可投票的地址（治理合约本身、销毁地址）
GOV_NON_VOTING = frozenset(ISSUER_RESERVES["UNI"]) | BURN_ADDRS
# 托管地址：币是客户的，交易所通常不代投，单列一档敏感性
GOV_CUSTODIAL_HINTS = ("binance", "coinbase", "kraken", "okx", "huobi",
                       "bitfinex", "kucoin", "bybit", "gate.io", "crypto-com")

# ============================================================ 基准
# HHI 行业基准（用于 fig_hhi_benchmark 的对照条）
# ★ 每一条都必须能在正文给出出处，键名与图注、正文的措辞保持一致。
#   SWIFT（HHI=10,000）已删除：把相关市场定义为"全球跨境银行间报文"
#   容易被质疑，而其余基准已足以支撑论证。
HHI_BENCHMARKS = {
    "US commercial banking":  935,     # \cite{ffiec2023bhc}，正文取区间下端 935--1,060
    "Canadian banking":     2_250,     # 同上文献族
    "Swedish banking":      2_400,     # 同上文献族
    "LPMCL gold clearing":  3_500,     # \cite{lbma2023clearing}
    "Visa/Mastercard":      5_150,     # \cite{nilson2024cards}
}

# Lorenz 图的传统资产基准：(名称, 文献报告的 Gini, 颜色)
# ★ 由 Gini 反解 L(p)=p^a 画示意曲线，非原始微观数据，图注必须写明。
LORENZ_BENCHMARKS = [
    ("US household wealth", 0.75, "#8D99AE"),
    ("Physical gold bullion", 0.90, "#C9A227"),
    ("Fintech corporate equity", 0.95, "#7B6D8D"),
]

DOJ_UNCONCENTRATED = 1_500
DOJ_HIGHLY_CONCENTRATED = 2_500
