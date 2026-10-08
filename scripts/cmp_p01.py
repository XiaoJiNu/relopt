"""P0-1 验收对照：snap_base（本次重跑）vs 归档 zc_xfeat_*.json（应逐位一致=同口径回归），
并列出 rc/circ3 修复口径。用法: python scripts/cmp_p01.py"""
import json
import sys

NEW = {"lawn_sameday": "results/zc_xfeat_rc_lawn_sameday.json",
       "lawn_crossday": "results/zc_xfeat_rc_lawn_crossday.json",
       "0923": "results/zc_xfeat_rc_0923.json"}
OLD = {"lawn_sameday": "/data/code/reLocation-all/accelerated_features/outputs/zc_xfeat_lawn_sameday.json",
       "lawn_crossday": "/data/code/reLocation-all/accelerated_features/outputs/zc_xfeat_lawn_crossday.json",
       "0923": "/data/code/reLocation-all/accelerated_features/outputs/zc_xfeat_0923.json"}
COLS = ("ep_mean", "ep_med", "ep_p90", "eth_mean", "eth_med", "eth_p90")

for sc in sys.argv[1:] or list(NEW):
    new, old = json.load(open(NEW[sc])), json.load(open(OLD[sc]))
    b, o = new["snap_base"], old["snap"]
    bad = [f"{k}:{b[k]}!={o[k]}" for k in COLS if abs(b[k] - o[k]) > 1e-9]
    print(f"[{sc}] base回归 vs 归档: {'OK 逐位一致' if not bad else 'DIFF ' + '; '.join(bad)}"
          f" (solve {new['solve_rate']} vs {old.get('solve_rate')})")
    for agg in ("snap", "ls", "med"):
        for v in ("base", "rc", "circ3"):
            d = new[f"{agg}_{v}"]
            print(f"   {agg}_{v:6s} ep {d['ep_mean']:.3f}/{d['ep_med']:.3f}/{d['ep_p90']:.3f}"
                  f"  eθ {d['eth_mean']:.2f}/{d['eth_med']:.2f}/{d['eth_p90']:.2f}")
