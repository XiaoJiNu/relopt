"""§3.1c/d 离线扫：在 --dump-cands 的逐候选数据上复算聚合，零重推理。
扫两个轴：守卫角 guard ∈ {25,35,45}（rc 需要更大交会角补 look-alike），
以及朝向选择器：snap(仅 top-1) / circ3(top-3 圆均值) / wmix(逐候选置信混合——
R 与 db 朝向偏差 rel|Δyaw| 大的候选回退 db 朝向，修 §3.1a 实测发现的草坪退化)。
用法: python scripts/sweep_cands.py [scene...]"""
import json
import sys

sys.path.insert(0, "/data/code/reLocation-all/accelerated_features/scripts")
sys.path.insert(0, "/data/code/reLocation-all/AsymLoc/scripts")

import numpy as np

from eval_zc_netvlad import circmean, ls_solve, med_solve  # noqa: E402
from eval_zc_xfeat_rc import yaw_err  # noqa: E402


GUARDS = (25, 35, 45, 60)
AGGS = ("snap", "ls", "med")
HS = ("base", "rc", "circ3", "wmix", "bg08", "bg15", "agree", "trig")  # agree: top-3 Δyaw 圆散布<10° 才用 rc


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def agg_by_q(cands, gt):
    """cands: dict of arrays (per-candidate rows), gt: (nq,4,4). 返回 {(guard,agg,hx): [(ep,eth)...]}"""
    qi = cands["qi"].astype(int)
    order = np.argsort(qi, kind="stable")
    qi = qi[order]
    for k in cands:
        cands[k] = cands[k][order]
    bounds = np.searchsorted(qi, np.arange(len(gt) + 1))
    out = {(g, a, x): [] for g in GUARDS for a in AGGS for x in HS}
    for q in range(len(gt)):
        lo, hi = bounds[q], bounds[q + 1]
        h_gt = np.arctan2(gt[q, 1, 2], gt[q, 0, 2])
        for g in GUARDS:
            m = cands["ang"][lo:hi] <= g
            sl = {k: cands[k][lo:hi][m] for k in cands}
            n = len(sl["ninl"])
            if n == 0:
                for key in [k for k in out if k[0] == g]:
                    out[key].append((np.inf, np.inf))
                continue
            o = np.argsort(-sl["ninl"], kind="stable")
            top = o[:3]
            pos = {}
            for a in AGGS:
                if a == "snap":
                    pos[a] = float(np.linalg.norm(sl["c"][o[0]][:2] - gt[q, :2, 3]))
                else:
                    cl = [dict(c=sl["c"][i], d=sl["d"][i], ninl=int(sl["ninl"][i])) for i in range(n)]
                    xy = (ls_solve if a == "ls" else med_solve)(cl) if n >= 2 else None
                    pos[a] = float(np.linalg.norm(np.asarray(xy) - gt[q, :2, 3])) if xy is not None else np.inf
            # 朝向：base=db量化；rc=db⊕R；circ3=rc top-3圆均值；wmix=|Δyaw_R|>25°回退db（§3.1a保险）
            drel = np.abs(wrap(sl["h_rc"] - sl["h_db"]))
            h_eff = np.where(drel > np.radians(25), sl["h_db"], sl["h_rc"])
            # bgate：R 在零基线（候选≈best 同一 db 帧）下病态——草坪 look-alike top-3 常是
            # 同一时刻附近的帧（间距 p50 ~0.15 m），交会角≈0 ⇒ Δyaw_R 纯噪声。与 best 基线
            # < thr 的候选回退 db 朝向；大交会角（0923 基线大）正常用 R。
            bl = np.linalg.norm(sl["c"][:, :2] - sl["c"][o[0]][:2], axis=1)
            circ, snapb = {}, {}
            for thr, nm in ((0.8, "bg08"), (1.5, "bg15")):
                fallback = (drel > np.radians(25)) | (bl < thr)
                heff_b = np.where(fallback, sl["h_db"], sl["h_rc"])
                circ[nm] = circmean(heff_b[top], sl["ninl"][top])
                snapb[nm] = heff_b[o[0]]
            # agree：top-3 的 Δyaw_R 圆散布(1-|R̄|) 小 → 是真实相对旋转信号；散 → 视差噪声回退 db
            dv = wrap(sl["h_rc"][top] - sl["h_db"][top])
            spread = np.degrees(np.arccos(min(1.0, abs(np.mean(np.exp(1j * dv))))))
            # trig：交会角判别——top-候选 |Δyaw_R| 中位数小 ⇒ db 全是近重复帧（草坪 look-alike），
            # R 只剩视差噪声 ⇒ 回退 base；中位数大 ⇒ 真实交会角（0923 型）⇒ 用 rc 圆均值。GT-free。
            med_dy = float(np.median(drel))
            circ = {**{"base": circmean(sl["h_db"][top], sl["ninl"][top]),
                       "rc": circmean(sl["h_rc"][top], sl["ninl"][top]),
                       "circ3": circmean(sl["h_rc"][top], sl["ninl"][top]),
                       "wmix": circmean(h_eff[top], sl["ninl"][top])}, **circ}
            circ["trig"] = circ["circ3"] if med_dy > np.radians(3) else circ["base"]
            snapb["trig"] = sl["h_rc"][o[0]] if med_dy > np.radians(3) else sl["h_db"][o[0]]
            circ["agree"] = circ["circ3"] if spread < 10 else circ["base"]
            snapb["agree"] = sl["h_rc"][o[0]] if spread < 10 else sl["h_db"][o[0]]
            for a in AGGS:
                for x in HS:
                    if a == "snap":  # snap 口径取 top-1 自身朝向（circ3 例外=文档§3.1b 多候选均值）
                        hh = (snapb[x] if x in snapb else circ["circ3"] if x == "circ3"
                              else (sl["h_db"] if x == "base" else sl["h_rc"] if x == "rc" else h_eff)[o[0]])
                    else:
                        hh = circ[x]
                    out[(g, a, x)].append((pos[a], yaw_err(hh, h_gt)))
    return out


def stat(arr):
    a = np.asarray(arr, float).reshape(-1, 2)
    if not len(a):
        return {}
    ok = np.isfinite(a[:, 0]) & np.isfinite(a[:, 1])
    if not ok.any():
        return {}
    return dict(ep_med=round(float(np.median(a[ok, 0])), 3), ep_p90=round(float(np.percentile(a[ok, 0], 90)), 3),
                eth_med=round(float(np.median(a[ok, 1])), 2), eth_p90=round(float(np.percentile(a[ok, 1], 90)), 2))


def main():
    scenes = sys.argv[1:] or ["lawn_sameday", "lawn_crossday", "0923"]
    table = {}
    for sc in scenes:
        d = np.load(f"results/zc_xfeat_rc_{sc}_cands.npz")
        res = agg_by_q({k: d[k] for k in ("qi", "c", "d", "ninl", "ang", "h_db", "h_rc")}, d["gt"])
        for key, arr in res.items():
            s = stat(arr)
            if s:
                table.setdefault(".".join(map(str, key)), {})[sc] = s
    # 汇总：三场景均值列 + 各场景
    cols = sorted(table, key=lambda k: (k.split(".")[1], k.split(".")[0]))
    row_avg = {}
    for k in cols:
        if len(table[k]) == 3:
            row_avg[k] = {m: round(float(np.mean([table[k][s][m] for s in table[k]])), 3)
                          for m in ("ep_med", "ep_p90", "eth_med", "eth_p90")}
    out = {"per_scene": table, "avg3": row_avg}
    json.dump(out, open("results/sweep_cands.json", "w"), ensure_ascii=False, indent=1)
    for k in sorted(row_avg, key=lambda k: row_avg[k]["eth_p90"]):
        a = row_avg[k]
        g, agg, hx = k.split(".")
        print(f"guard{g:>3} {agg:4s} {hx:6s}  avg ep {a['ep_med']:.3f}/{a['ep_p90']:.3f}"
              f"  eθ {a['eth_med']:.2f}/{a['eth_p90']:.2f}")


if __name__ == "__main__":
    main()
