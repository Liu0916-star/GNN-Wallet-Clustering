# -*- coding: utf-8 -*-
"""
00_data_check.py —— 开跑前的闸门

在跑任何分析之前先确认：数据是对的、口径是清楚的、标签快照在。
任何一项 [X] 都要先解决，否则后面所有数字都不可信。

用法：python 00_data_check.py
"""
import numpy as np
import pandas as pd

import config as C
import common as cm

SCRIPT = "00_data_check"


def main():
    cm.banner("00 · 数据自检")
    all_ok = True

    # ---------- 标签快照 ----------
    print("\n[标签快照]")
    try:
        lm = cm.load_labels()
        print(f"    [OK] {C.LABEL_SNAPSHOT.name} · {len(lm):,} 条 · "
              f"md5 {cm.file_md5(C.LABEL_SNAPSHOT)}")
        print(f"    ★ 论文可复现性章节需注明这个日期与 md5")
    except FileNotFoundError as e:
        all_ok = False
        print(f"    [X] {e}")
        lm = None

    for token in C.TOKENS:
        cm.banner(token, "-")
        exp = C.EXPECT[token]

        # ---------- 文件 ----------
        for role in ("tx", "bal"):
            p = C.FILES[token][role]
            if not p.exists():
                all_ok = False
                print(f"    [X] 缺文件 {p}")
                continue
            print(f"    {p.name}  {p.stat().st_size / 1e6:8.1f} MB  "
                  f"md5 {cm.file_md5(p)}")

        if not C.FILES[token]["tx"].exists():
            continue

        # ---------- 转账（不剔零地址，先看原貌）----------
        raw = cm.load_transfers(token, drop_burn=False)
        print(f"\n  [转账]")
        all_ok &= cm.check("行数", len(raw), exp["n_tx"])
        print(f"    列识别 → from/to/value/time 全部按列名命中（未使用 iloc）")
        print(f"    时间 {raw['ts'].min()} → {raw['ts'].max()}")

        t0, t1 = pd.Timestamp(C.WINDOW_START), pd.Timestamp(C.WINDOW_END)
        oob = int(((raw["ts"] < t0) | (raw["ts"] >= t1)).sum())
        print(f"    {'[OK]' if oob == 0 else '[X] '} 窗口越界 {oob} 条 "
              f"（应为 0，SQL 是左闭右开）")
        all_ok &= (oob == 0)
        all_ok &= cm.check("周窗数", int(raw["week"].nunique()), C.N_WINDOWS)

        # ---------- mint / burn / 零值 ----------
        n_mint = int((raw["from"] == C.ZERO_ADDR).sum())
        n_burn = int(raw["to"].isin(C.BURN_ADDRS).sum())
        n_zero = int((raw["amt"] <= 0).sum())
        n_self = int((raw["from"] == raw["to"]).sum())
        print(f"\n  [口径影响项]")
        print(f"    铸造(from=0x0) {n_mint:,} | 销毁(to=burn) {n_burn:,} | "
              f"零值转账 {n_zero:,} | 自环 {n_self:,}")

        tx = cm.load_transfers(token, drop_burn=True)
        a_raw = cm.active_addresses(raw, drop_zero_value=False)
        a_nb = cm.active_addresses(tx, drop_zero_value=False)
        a_nz = cm.active_addresses(tx, drop_zero_value=True)
        print(f"    活跃地址: 原口径 {len(a_raw):,} | 去零地址 {len(a_nb):,} "
              f"| 再去零值 {len(a_nz):,}")
        all_ok &= cm.check("活跃地址（原口径）", len(a_raw), exp["n_active"])
        print(f"    → 零值转账使活跃地址多算 "
              f"{(len(a_nb) - len(a_nz)) / max(len(a_nb), 1) * 100:.1f}%"
              f"（config.DROP_ZERO_VALUE_FOR_ACTIVE="
              f"{C.DROP_ZERO_VALUE_FOR_ACTIVE}）")

        # ---------- 余额 ----------
        bal = cm.load_balances(token)
        print(f"\n  [余额]")
        all_ok &= cm.check("正余额地址", len(bal), exp["n_holders"])
        tot = bal["balance"].sum()
        print(f"    总量 {tot:,.0f}")
        neg = int((bal["balance"] < 0).sum())
        if neg:
            print(f"    [!] 负余额 {neg} 个（SQL 有 HAVING balance>0，"
                  f"出现负值说明文件与规范 SQL 不同源）")

        burned = bal.loc[bal["address"].isin(C.BURN_ADDRS), "balance"].sum()
        iss = C.ISSUER_RESERVES.get(token, frozenset())
        reserve = bal.loc[bal["address"].isin(iss), "balance"].sum()
        print(f"    销毁地址持有 {burned:,.0f}（{burned / tot * 100:.2f}%）")
        print(f"    发行方储备 {reserve:,.0f}（{reserve / tot * 100:.2f}%）"
              f" · {len(iss)} 个地址")

        # ---------- 异常窗 ----------
        out = cm.find_outlier_windows(tx)
        print(f"\n  [异常窗] {out if out else '无'}"
              f"（地址数 > 中位数 ×{C.OUTLIER_NODE_MULTIPLIER}）")
        print(f"    主口径 EXCLUDE_OUTLIER_WINDOWS="
              f"{C.EXCLUDE_OUTLIER_WINDOWS}")

        # ---------- 重复边（量化 v16 的边权 bug）----------
        agg = tx.groupby(["from", "to"], as_index=False)["amt"].sum()
        dup_rows = len(tx) - len(agg)
        wk_loss = []
        for w, g in tx.groupby("week"):
            ga = g.groupby(["from", "to"])["amt"].agg(["sum", "last"])
            wk_loss.append(1 - ga["last"].sum() / ga["sum"].sum())
        print(f"\n  [重复边] {dup_rows:,} 行 "
              f"（占 {dup_rows / len(tx) * 100:.1f}%）")
        print(f"    v16 只取每对地址的最后一笔，周均丢失金额 "
              f"{np.mean(wk_loss) * 100:.1f}%")
        print(f"    ★ 这个比例本身可作为路由层反复交互的证据（写进 C3）")

        cm.log_metric(SCRIPT, token, "n_transfers", len(raw))
        cm.log_metric(SCRIPT, token, "n_holders", len(bal))
        cm.log_metric(SCRIPT, token, "n_active_raw", len(a_raw))
        cm.log_metric(SCRIPT, token, "dup_edge_value_loss_pct",
                      round(float(np.mean(wk_loss)) * 100, 2),
                      "v16 DiGraph 覆盖导致的金额丢失")

    cm.banner("[OK] 全部通过，可以往下跑" if all_ok
              else "[X] 有未通过项，先解决再往下跑")


if __name__ == "__main__":
    main()
