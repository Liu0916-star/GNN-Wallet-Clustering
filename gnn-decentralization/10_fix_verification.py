# -*- coding: utf-8 -*-
"""
10_fix_verification.py —— 修正核验表的代币归属 + 生成待核实模板

两件事：

  A. 重算 broker_identities.csv 的 token 列
     旧表把 1inch / 0x / Bitget / Mayan / Uniswap X / Universal Router
     六个全标成 LINK，但实测归属是跨币 / 仅LINK / 仅UNI / 未入选。
     那一列记录的其实是"在哪一轮核验中被发现"，不是"在哪个币上出现"。
     现在按实测重算，并把原值保留为 discovered_in 以备溯源。

  B. 生成 verification/to_verify_template.csv
     把 09 找出的待核实地址填成与 broker_identities.csv 同构的空表，
     人工在 Etherscan 查完直接填 verified_identity / evidence 两列，
     然后 concat 回主表即可。

输出：
  verification/broker_identities_fixed.csv   （★ 人工确认后改名覆盖原表）
  verification/to_verify_template.csv

用法：python 10_fix_verification.py
"""
import numpy as np
import pandas as pd

import config as C
import common as cm

SCRIPT = "10_fix_verification"
VDIR = C.ROOT / "verification"


def main():
    cm.banner("10 · 修正核验表归属 + 生成待核实模板")

    vp = VDIR / "broker_identities.csv"
    if not vp.exists():
        print(f"  [X] 未找到 {vp}")
        return
    vb = pd.read_csv(vp)
    vb.columns = [c.strip().lstrip("\ufeff") for c in vb.columns]
    acol = next(c for c in vb.columns if "addr" in c.lower())
    vb[acol] = vb[acol].astype(str).str.strip().str.lower()
    print(f"  核验表 {len(vb)} 条，列: {list(vb.columns)}")

    # ---------- 读新经纪人集合 ----------
    sets, brok = {}, {}
    for t in C.TOKENS:
        p = C.RESULTS / f"hidden_brokers_{t}.csv"
        if not p.exists():
            print(f"  [X] 缺 {p.name}，先跑 05_c3_roles.py")
            return
        d = pd.read_csv(p)
        d["address"] = d["address"].astype(str).str.strip().str.lower()
        brok[t] = d
        sets[t] = set(d["address"])
    a, b = C.TOKENS
    print(f"  {a} {len(sets[a])} | {b} {len(sets[b])} | "
          f"交集 {len(sets[a] & sets[b])}")

    # ---------- A. 重算 token 列 ----------
    def belongs(addr):
        ia, ib = addr in sets[a], addr in sets[b]
        if ia and ib:
            return "BOTH"
        if ia:
            return a
        if ib:
            return b
        return "NEITHER"

    old_col = "token" if "token" in vb.columns else None
    if old_col:
        vb = vb.rename(columns={old_col: "discovered_in"})
        print(f"  原 token 列 → 保留为 discovered_in（记录发现轮次，非归属）")
    vb["token"] = vb[acol].map(belongs)
    vb["in_LINK"] = vb[acol].isin(sets[a])
    vb["in_UNI"] = vb[acol].isin(sets[b])

    # 出现率：跨币取两币最大
    def freq(addr):
        v = []
        for t in C.TOKENS:
            s = brok[t][brok[t]["address"] == addr]
            if len(s) and "seed_frequency" in s.columns:
                v.append(float(s["seed_frequency"].iloc[0]))
        return max(v) if v else np.nan

    vb["seed_frequency"] = vb[acol].map(freq)

    print(f"\n  [归属重算结果]")
    print(f"    {vb['token'].value_counts().to_dict()}")
    if old_col:
        chg = vb[vb["discovered_in"].astype(str).str.upper()
                 != vb["token"].astype(str).str.upper()]
        icol = next((c for c in vb.columns
                     if c.lower() in ("verified_identity", "identity",
                                      "name", "label")), None)
        print(f"\n  [归属发生变化的 {len(chg)} 条]")
        print(f"    {'身份':<36}{'旧':<8}{'新':<10}出现率")
        for r in chg.itertuples():
            nm = str(getattr(r, icol, "?"))[:34] if icol else "?"
            fq = (f"{r.seed_frequency:.0%}"
                  if pd.notna(r.seed_frequency) else "—")
            print(f"    {nm:<36}{str(r.discovered_in):<8}"
                  f"{str(r.token):<10}{fq}")

    neither = vb[vb["token"] == "NEITHER"]
    if len(neither):
        print(f"\n  [!] {len(neither)} 条已不在任何经纪人集合中（09 的 B 段）。")
        print(f"      保留在表内作为历史记录，token=NEITHER，不进论文表格。")

    # 列顺序与模板对齐，便于直接 concat
    head = ["token", "address", "in_LINK", "in_UNI", "counterparties",
            "seed_frequency", "verified_identity", "evidence", "note",
            "discovered_in"]
    cols = [c for c in head if c in vb.columns] + \
           [c for c in vb.columns if c not in head]
    vb = vb[cols]
    out = VDIR / "broker_identities_fixed.csv"
    vb.to_csv(out, index=False)
    print(f"\n  → {out.name}（★ 人工确认后改名覆盖原表）")

    # ---------- B. 待核实模板 ----------
    union = sets[a] | sets[b]
    todo = sorted(union - set(vb[acol]))
    rows = []
    for addr in todo:
        vals = {"counterparties": [], "betweenness": [], "seed_frequency": []}
        for t in C.TOKENS:
            s = brok[t][brok[t]["address"] == addr]
            if not len(s):
                continue
            for k in vals:
                if k in s.columns and pd.notna(s[k].iloc[0]):
                    vals[k].append(float(s[k].iloc[0]))
        rows.append({
            "token": belongs(addr),
            "address": addr,
            "in_LINK": addr in sets[a],
            "in_UNI": addr in sets[b],
            "counterparties": max(vals["counterparties"], default=np.nan),
            "betweenness": max(vals["betweenness"], default=np.nan),
            "seed_frequency": max(vals["seed_frequency"], default=np.nan),
            "verified_identity": "",        # ← 人工填
            "evidence": "",                 # ← 人工填（Etherscan 标签页/合约名）
            "note": "",
        })
    td = pd.DataFrame(rows)
    if len(td):
        td["prio"] = np.where(td["token"] == "BOTH", 0,
                              np.where(td["counterparties"] >= 60, 1, 2))
        td = td.sort_values(["prio", "counterparties"],
                            ascending=[True, False]).drop(columns="prio")
        tp = VDIR / "to_verify_template.csv"
        td.to_csv(tp, index=False)
        print(f"\n  [待核实 {len(td)} 个] → {tp.name}")
        print(f"    {'优先':<6}{'address':<44}{'归属':<7}{'对手方':>7}{'介数':>11}")
        for i, r in enumerate(td.itertuples()):
            pr = "🔴必查" if r.token == "BOTH" else (
                "🟠建议" if r.counterparties >= 60 else "🟡可选")
            print(f"    {pr:<6}{r.address:<44}{r.token:<7}"
                  f"{r.counterparties:>7.0f}{r.betweenness:>11.6f}")
        print(f"\n    填完 verified_identity 与 evidence 两列后：")
        print(f"      cat verification/to_verify_template.csv 追加到 "
              f"broker_identities_fixed.csv")
        print(f"      改名为 broker_identities.csv，重跑 05_c3_roles.py")
    else:
        print("\n  [OK] 无待核实地址")

    cm.log_metric(SCRIPT, "BOTH", "n_reassigned",
                  int(len(chg)) if old_col else 0)
    print(f"\n  合并命令（填完模板后执行）：")
    print(f"    python -c \"import pandas as pd;"
          f"a=pd.read_csv('verification/broker_identities_fixed.csv');"
          f"b=pd.read_csv('verification/to_verify_template.csv');"
          f"pd.concat([a,b],ignore_index=True)"
          f".to_csv('verification/broker_identities.csv',index=False)\"")
    cm.log_metric(SCRIPT, "BOTH", "n_to_verify", len(td))
    cm.banner("完成")


if __name__ == "__main__":
    main()
