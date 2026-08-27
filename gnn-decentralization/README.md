# GNN Decentralization —— 重构版分析仓库

替代 `02_gnn.ipynb`（61 cell，含多轮迭代与死代码）。v16 的实质结论经复算全部
成立，重构的目的是修掉口径不一致与一个影响面较大的建图 bug，并让每个数字可溯源。

---

## 一、先做这三件事

```bash
# 1. 数据就位
cp /path/to/link_90d_transfers.csv link_balances.csv \
   uni_90d_transfers.csv uni_balances.csv  data/

# 2. 抓一次标签快照（之后永远读本地，不再联网）
python -c "import urllib.request,shutil;shutil.copyfileobj(\
urllib.request.urlopen('https://raw.githubusercontent.com/brianleect/etherscan-labels/main/data/etherscan/combined/combinedAllLabels.json'),\
open('labels/etherscan_labels_2026-08-09.json','wb'))"

# 3. 冻结 v16 旧结果（不要删，论文要做修正前后对比）
cp -r /path/to/old/results/* results_v16_legacy/
```

## 一之补充：可选的核验文件

把 v16 的 `broker_identities.csv`（28 条已核验身份）放到 `verification/` 下，
`05_c3_roles.py` 会自动读取并报告各方法命中多少条。

```bash
mkdir -p verification
cp /path/to/broker_identities.csv verification/
```

★ 这 28 条是当初用 GNN 定义筛出后才去 Etherscan 核验的，**存在循环偏倚**，
只能作参考，不能作为"GNN 更准"的证据。脚本会自动打印这条提示。

## 二、运行顺序

```bash
python 00_data_check.py        # 闸门：有 [X] 先解决再往下
python 01_build_graph.py       # LINK+UNI 的 1w；5w/64w 用参数指定
python 02_c1.py                # C1 三证据 + 多种子 + 消融
python 03_c2_weekly.py         # C2 周度序列
python 04_c2_procrustes.py     # C2 重量版
python 05_c3_roles.py          # C3 角色 + 隐藏经纪人 + 跨币重叠
python 06_concentration.py     # 集中度网格 + 实体合并
python 07_governance.py        # 规则层（仅 UNI）
python 08_figures.py           # 图表，只读 results/
```

`01`/`02`/`05` 可加参数：`python 01_build_graph.py LINK 64w`

所有关键数字会追加到 `results/manifest.csv`（脚本、指标、数值、时间戳），
论文里每个数都能一行溯源。

---

## 三、三条硬规则（针对已发生过的三个坑）

**1. 建图只有 `common.build_graph()` 一个入口。**
v16 有 13 处 `from_pandas_edgelist`，`DiGraph` 对每对地址只保留最后一条边，
不做求和，丢掉 **LINK 48.8% / UNI 59.6%** 的转账金额。影响
`in_deg_w` / `out_deg_w` / `pagerank`；拓扑量（度、对手方、betweenness、
连通结构）不受影响。

**2. 列名一律按名字识别，禁用 `iloc` 取列。**
v16 栽过两次：UNI 余额 2 列 / LINK 4 列，`iloc[:,1]` 在 LINK 上取到
`total_received`，算出 3,918,660 个地址（真值 885,011）。
`common.find_col()` 找不到就抛错并打印真实列名，绝不静默回退。

**3. `common.load_labels()` 只读本地快照，联网直接报错。**
`brianleect/etherscan-labels` 持续更新（8/5 = 29,945 条，8/9 = 29,772 条），
`is_core` 锚点随之漂移，且**不会报错**——C1 的全部 AUC 会静默改变。
快照必须提交进仓库，论文注明抓取日期与 md5。

---

## 四、口径决策（都集中在 `config.py`）

| 选择 | 主口径 | 敏感性 |
|---|---|---|
| 人群 | 全部持币地址 | 活跃地址集 |
| 方向 | 发送方出流 | 接收方入流 |
| 时间聚合 | 周度算再平均 | 全窗汇总一次 |
| 异常窗 | 保留 | 剔除 |
| 零地址 | 收发两端剔除 | — |
| 实体解析 | 一地址=一参与者 | 同实体合并 |

要改口径，改 `config.py` 后重跑，不要去脚本里翻。

---

## 五、v16 与本仓库的差异一览

| # | 问题 | 处理 |
|---|---|---|
| P1 | 边权覆盖（丢 48.8%/59.6% 金额） | `build_graph` 求和 |
| P2 | C2 表把两个人群口径混在一行（UNI 的 Gini/top1%/top10 来自全部持币，HHI 2,336 来自活跃集） | 主表统一全部持币（LINK 135 / UNI 872） |
| P3 | Gini/top1%（入流·12窗·覆盖边权）与 Flow HHI（出流·13窗·求和）不同源 | 全部统一，其余进敏感性 |
| P4 | 标签库漂移且静默 | 本地快照 + md5 |
| P5 | 第 779 行「共享经纪人都没标签」与前一段矛盾（15 个跨币经纪人中 9 个有标签） | `05` 自动计数并提示改法 |
| P8 | `core_score` 是样本内拟合，无 CV | 结果里写明只作描述性筛选 |
| P9 | 周度 vs 全窗差 1.83 倍，未交代 | `03` 并列输出 |
| P10 | 异常窗剔除规则前后不一致 | 由 `config` 统一 |
| P11 | C2 逐窗 betweenness k=200 论文未写 | `config.BETWEENNESS_K` |
| P16 | 零值转账进入活跃地址集 | `00` 量化，`config` 开关 |
| P17 | 退化经纪人（对手方=1、介数=0，含 vanity 孪生地址） | 加下限，剔除并列出 |
| P18 | `featF_norm`(k=500) 与 `featF_fixed_k5000` 并存且无版本提示 | 文件名带参数，npz 内嵌 meta |
| P19 | W7 的 `gini_betweenness=0` 是采样退化，非"结构均等" | 图上圈出，附录说明 |
| P20 | `key_numbers_summary.json` 的 C1 部分已过期（33x/181x/562x），C2 部分是当前值 | 不再使用，由 `manifest.csv` 取代 |

## 六、C3 数字的来源（曾有歧义，已定案）

隐藏经纪人出自 **cell 41**，簇命名出自 **cell 42**，**cell 40 是死代码**。
判据：`hidden_brokers.csv` 的列名是 `core_score`，只有 cell 41 用这个变量名，
cell 40 用的是 `struct_score` —— 两者是不同的经纪人定义，不能混用。
