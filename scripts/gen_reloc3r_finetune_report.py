#!/usr/bin/env python
"""Reloc3r ZC 微调实验报告生成器（方案 §6.3/§7 自动填表）。

读: reloc3r/data/zc_sim3/res_<k>_sim5_bm3_{official_odd,ft_odd,ft_all}.npz (+official all=res_<k>_sim5_bm3.npz)
    reloc3r/results/forgetting_{official,ft}.json, checkpoints/zc_finetune*/log.txt
写: relopt/docs/算法优化/ZC数据微调Reloc3r与XFeat并回测方案_Reloc3r实验报告_<ts>.md
缺项自动标「待测」，可反复重跑（幂等，每次新 ts）。A2: dvec 沿/跨 GT 轨迹分量。
用法: python gen_reloc3r_finetune_report.py [--out PATH]
"""
import argparse, json, os, sys
from datetime import datetime
import numpy as np

RELOC = '/data/code/reLocation-all/reloc3r'
SIM3 = f'{RELOC}/data/zc_sim3'
RES = f'{RELOC}/results'
DOCS = '/data/code/reLocation-all/relopt/docs/算法优化'
SCENES = [('062224', '草坪 同天'), ('081533', '草坪 跨天'), ('0923', '大楼 0923')]
Q_BAG = {'062224': '20260921_062224', '081533': '20260920_081533',
         '0923': '20260923_060415_real_robot_record_external_lz4'}
ROOT = {'062224': 'zc_fix', '081533': 'zc_fix', '0923': 'zc_0923'}


def six(d):
    return (f"{np.mean(d):.3f} / {np.median(d):.3f} / {np.percentile(d, 90):.3f}")


def load(key, suffix):
    p = f'{SIM3}/res_{key}_sim5_bm3{suffix}.npz'
    return np.load(p, allow_pickle=True) if os.path.exists(p) else None


def heading_unit(scene, name, names_all):
    """GT 轨迹朝向（map 系 xy 单位向量）：用 query bag 相邻帧差分。"""
    i = names_all.index(name)
    poses = {}
    def pose(n):
        if n not in poses:
            poses[n] = np.loadtxt(f"{RELOC}/data/{ROOT[scene]}/{Q_BAG[scene]}/seq-01/{n}.pose.txt")
        return poses[n]
    j = min(i + 1, len(names_all) - 1)
    k = max(i - 1, 0)
    v = pose(names_all[j])[:3, 3] - pose(names_all[k])[:3, 3]
    n = np.linalg.norm(v[:2])
    if n < 1e-6:
        return None
    return np.array([v[0] / n, v[1] / n, 0.0])


def a2_split(arr, scene):
    """返回 (沿轨 p90, 跨轨 p90)：dvec 在 GT 航向上的投影。names=奇/偶子集, 航向从全 bag 取。"""
    names = [str(x)[:-10] for x in arr['names']]
    all_names = sorted(n[:-10] for n in os.listdir(f"{RELOC}/data/{ROOT[scene]}/{Q_BAG[scene]}/seq-01")
                       if n.endswith('.color.png'))
    along, cross = [], []
    for dv, nm in zip(arr['dvec'], names):
        if 'dvec' not in arr.files:
            return (None, None)
        u = heading_unit(scene, nm, all_names)
        if u is None:
            continue
        a = abs(float(dv @ u))
        along.append(a)
        cross.append(float(np.linalg.norm(dv - a * u)))
    if not along:
        return (None, None)
    return (np.percentile(along, 90), np.percentile(cross, 90))


def jget(path):
    return json.load(open(path)) if os.path.exists(path) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    ts = datetime.now().strftime('%Y%m%d%H%M')
    out = a.out or f'{DOCS}/ZC数据微调Reloc3r与XFeat并回测方案_Reloc3r实验报告_{ts}.md'
    L = []
    add = L.append
    fo, ffx = jget(f'{RES}/forgetting_official.json'), jget(f'{RES}/forgetting_ft.json')
    tr = {}  # 训练摘要
    for tag, d in [('主协议(even→odd)', 'checkpoints/zc_finetune'),
                   ('上限行(all→all)', 'checkpoints/zc_finetune_all')]:
        lp = f'{RELOC}/{d}/log.txt'
        if os.path.exists(lp):
            rows = [json.loads(l) for l in open(lp) if l.strip()]
            last = rows[-1]
            tr[tag] = (len(rows) - 1, last.get('train_loss'),
                       last.get('2000 @ ZcRelpose_RelativeCameraPoseRegression_terr1_med'),
                       last.get('2000 @ ZcRelpose_RelativeCameraPoseRegression_rerr1_med'))
    fo, ffx = jget(f'{RES}/forgetting_official.json'), jget(f'{RES}/forgetting_ft.json')

    add('# ZC 数据微调 Reloc3r — 实验报告')
    add('')
    add(f'（生成 {datetime.now():%Y-%m-%d %H:%M}。方案=`ZC数据微调Reloc3r与XFeat并回测方案.md`；'
        '口径=sim5+bm3 非缓存 `eval_zc_sim3.py`；数值列=terr/rerr 的 mean/med/p90；'
        '主协议=查询帧奇偶切分(train even/eval odd)，上限行=全训全测(含泄漏，仅上界参考)。'
        'XFeat 支线不在本报告（另行）。）')
    add('')
    add('## 1. 主协议回测（奇数 eval 帧，同子集，唯一变量=权重）')
    add('')
    add('| 场景 | 权重 | n | ep mean/med/p90 (m) | eθ mean/med/p90 (°) |')
    add('| ---- | ---- | - | ----------------- | ------------------ |')
    rows_main = {}
    for key, cn in SCENES:
        for w, suf, lbl in [('off', '_official_odd', '官方基线'), ('ft', '_ft_odd', 'ZC 微调')]:
            arr = load(key, suf)
            if arr is None:
                add(f'| {cn} {key} | {lbl} | 待测 | 待测 | 待测 |')
            else:
                add(f'| {cn} {key} | {lbl} | {len(arr["terr"])} | {six(arr["terr"])} | {six(arr["rerr"])} |')
                rows_main[(key, w)] = arr
    add('')
    add('## 2. 全训全测上限行（含 pair 级泄漏，只作记忆上界，不得当泛化）')
    add('')
    add('| 场景 | 权重 | n | ep mean/med/p90 (m) | eθ mean/med/p90 (°) |')
    add('| ---- | ---- | - | ----------------- | ------------------ |')
    rows_ub = {}
    for key, cn in SCENES:
        for w, suf, lbl in [('off', '', '官方(all帧)'), ('ft', '_ft_all', '微调(all)')]:
            arr = load(key, suf)
            if arr is None:
                add(f'| {cn} {key} | {lbl} | 待测 | 待测 | 待测 |')
            else:
                add(f'| {cn} {key} | {lbl} | {len(arr["terr"])} | {six(arr["terr"])} | {six(arr["rerr"])} |')
                rows_ub[(key, w)] = arr
    add('')
    add('## 3. 灾难性遗忘检查（MegaDepth-1500 后 750 对，与训练前 750 不相交）')
    add('')
    add('| 权重 | terr med (°) | rerr med (°) | 门槛(≤官方+10%) |')
    add('| ---- | ----------- | ------------ | --------------- |')
    if fo:
        add(f"| official | {fo['terr_deg_med']:.3f} | {fo['rerr_deg_med']:.3f} | — |")
    if ffx:
        lim = fo['terr_deg_med'] * 1.1 if fo else None
        ok = ('✅' if ffx['terr_deg_med'] <= lim else '❌') if lim else '待官方'
        add(f"| finetuned | {ffx['terr_deg_med']:.3f} | {ffx['rerr_deg_med']:.3f} | {ok} |")
    else:
        add('| finetuned | 待测 | 待测 | |')
    add('')
    add('## 4. 验收判定（方案 §7）')
    add('')
    gates = []
    if rows_main:
        imp = []
        for key, _ in SCENES:
            o, f = rows_main.get((key, 'off')), rows_main.get((key, 'ft'))
            if o is not None and f is not None:
                r = 1 - np.percentile(f['terr'], 90) / np.percentile(o['terr'], 90)
                imp.append((key, r))
        if imp:
            good = sum(1 for _, r in imp if r > 0.10)
            worst = min(r for _, r in imp)
            v1 = good >= 2 and worst > -0.05
            add(f'- 域适配有效（≥2 场景 ep p90 改善>10%，其余不劣化）: {"✅" if v1 else "❌"} — 各场景 p90 改善 '
                + ', '.join(f'{k}:{r*100:+.1f}%' for k, r in imp))
            gates.append(v1)
    if fo and ffx:
        v2 = ffx['terr_deg_med'] <= fo['terr_deg_med'] * 1.1
        add(f'- 灾难性遗忘: {"✅" if v2 else "❌"}（terr med {ffx["terr_deg_med"]:.3f}° vs 官方 {fo["terr_deg_med"]:.3f}°）')
        gates.append(v2)
    if rows_main and rows_ub:
        v3 = all(np.percentile(rows_ub[(k, 'ft')]['terr'], 90) < np.percentile(rows_main[(k, 'ft')]['terr'], 90)
                 for k, _ in SCENES if (k, 'ft') in rows_ub and (k, 'ft') in rows_main)
        add(f'- 上限行显著低于主协议（证明记忆存在）: {"✅" if v3 else "❌（若接近=切分未生效，回查 parity）"}')
        gates.append(v3)
    if not gates:
        add('- 待回测数据齐后判定。')
    add('')
    add('## 5. A2 沿轨分量（dvec 沿/跨 GT 航向，p90, m）')
    add('')
    add('| 场景 | 权重 | 沿轨 | 跨轨 |')
    add('| ---- | ---- | ---- | ---- |')
    any_a2 = False
    for key, cn in SCENES:
        for w, suf, lbl in [('off', '_official_odd', '官方基线'), ('ft', '_ft_odd', 'ZC 微调')]:
            arr = rows_main.get((key, w))
            if arr is not None and 'dvec' in arr.files:
                al, cr = a2_split(arr, key)
                if al is not None:
                    add(f'| {cn} {key} | {lbl} | {al:.2f} | {cr:.2f} |')
                    any_a2 = True
    if not any_a2:
        add('| （dvec 记录自本次改动起生效；旧 npz 无该字段则留空） | | | |')
    add('')
    add('## 6. 训练过程摘要')
    add('')
    for tag, (nep, tl, te, re_) in tr.items():
        add(f'- {tag}: 完成 {nep} epoch, 末 train_loss={tl}, odd-test terr med={te:.2f}° rerr med={re_:.2f}°'
            if te else f'- {tag}: 完成 {nep} epoch')
    if not tr:
        add('- （训练日志未出现=未开始/进行中）')
    add('')
    add('## 7. 产物与复现')
    add('')
    add('- 权重: `reloc3r/checkpoints/zc_finetune/checkpoint-best.pth`（主协议）、`zc_finetune_all/`（上限行）')
    add('- 逐帧结果: `reloc3r/data/zc_sim3/res_{062224,081533,0923}_sim5_bm3{,_ft_odd,_ft_all}.npz`')
    add('- 遗忘: `reloc3r/results/forgetting_{official,ft}.json`')
    add('- 复现: `SMOKE=0 bash scripts/finetune_zc.sh` → `bash scripts/finetune_pipeline.sh`（幂等全链）')
    add('- 吞吐: AMP 修复后实测 0.49 s/it（fp32 误跑=1.12 s/it，已修 70518f4）')
    txt = '\n'.join(L) + '\n'
    os.makedirs(DOCS, exist_ok=True)
    open(out, 'w').write(txt)
    print(out)


if __name__ == '__main__':
    main()
