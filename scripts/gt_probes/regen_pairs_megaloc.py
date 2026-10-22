"""Regenerate ZC retrieval-pair npy using MegaLoc order (instead of NetVLAD), for training.
Zero GPU: candidates come from archived MegaLoc res__*.npz `order` matrices (same archive
eval_zc_sim3.py already uses for eval candidates), so train-pair distribution == deployment
retrieval distribution by construction.

Output: reloc3r/_db-q_pair_info/zcM_<qbag>_q-retrieval_db-step=1_topk=10.npy
  same object-array format as NetVLAD npy (list per query of dicts), ragged allowed.
Filter (per retrieval-pair protocol): drop pairs with GT baseline <0.15 m (ill-posed small
baseline, below /map noise) or >10 m (weak co-visibility tail).
Usage: python regen_pairs_megaloc.py
"""
import os, sys, json
import numpy as np

REPO = '/data/code/reLocation-all/reloc3r'
ML = '/data/code/reLocation-all/MegaLoc/outputs/eval_zc_megaloc'
OUT_DIR = f'{REPO}/_db-q_pair_info'
TOPK = 10
BL_MIN, BL_MAX = 0.15, 10.0
# Phase2: DATA_TAG=enu → 读 zc_*_enu，输出前缀 zcME_/zc9ME_（不覆盖旧 zcM_*）
# 亦可 POSE_ROOT=./data/zc_fix_enu 强制草坪根；大楼需成对存在 zc_0923_enu
DATA_TAG = os.environ.get('DATA_TAG', '').strip().lower()
_ENU = DATA_TAG in ('enu', 'enu_rtk', '1', 'true', 'yes')

def _roots():
    if _ENU:
        return './data/zc_fix_enu', './data/zc_0923_enu', 'zcME', 'zc9ME'
    return './data/zc_fix', './data/zc_0923', 'zc', 'zc9'

_lawn, _bldg, _pref, _pref9 = _roots()
# POSE_ROOT 仅覆盖草坪根（兼容任务书）；大楼根随 DATA_TAG
if os.environ.get('POSE_ROOT'):
    _lawn = os.environ['POSE_ROOT']

SC = {  # key: (data_root_rel, db_bag, q_bag, megalloc_npz, out_prefix)
 '062224': (_lawn, '20260921_064638', '20260921_062224',
            f'{ML}/res__mowing__same_day_062224.npz', _pref),
 '081533': (_lawn, '20260921_064638', '20260920_081533',
            f'{ML}/res__mowing__cross_day_081533.npz', _pref),
 '0923': (_bldg, '20260923_055425_real_robot_record_external_lz4',
          '20260923_060415_real_robot_record_external_lz4',
          f'{ML}/res__building_0923__query_060415.npz', _pref9),
}

def frames(root_abs, bag):
    return sorted(n[:-10] for n in os.listdir(f'{root_abs}/{bag}/seq-01') if n.endswith('.color.png'))

_POSE_CACHE = {}
def pose(root_abs, bag, name):
    k = (root_abs, bag)
    if k not in _POSE_CACHE:
        _POSE_CACHE[k] = {}
    c = _POSE_CACHE[k]
    if name not in c:
        c[name] = np.loadtxt(f'{root_abs}/{bag}/seq-01/{name}.pose.txt')
    return c[name]

def main():
    stats = {}
    for key, (root_rel, db, q, mlf, pref) in SC.items():
        root_abs = os.path.join(REPO, root_rel)
        dbn, qn = frames(root_abs, db), frames(root_abs, q)
        order = np.load(mlf)['order']
        assert order.shape == (len(qn), len(dbn)), f'{key}: order {order.shape} vs ({len(qn)},{len(dbn)})'
        rows = []
        bl_all, bl_keep = [], []
        for qi, qname in enumerate(qn):
            pq = pose(root_abs, q, qname)
            lst = []
            for rank in range(min(TOPK * 3, order.shape[1])):  # over-draw to refill after filtering
                di = int(order[qi, rank])
                pd = pose(root_abs, db, dbn[di])
                t_rel = np.linalg.norm((np.linalg.inv(pd) @ pq)[:3, 3])
                if rank < TOPK:
                    bl_all.append(t_rel)
                if BL_MIN <= t_rel <= BL_MAX:
                    lst.append({'query_path': f'{root_rel}/{q}/seq-01/{qname}.color.png',
                                'db_path': f'{root_rel}/{db}/seq-01/{dbn[di]}.color.png',
                                'query_idx': qi, 'db_idx': di})
                    if rank < TOPK:
                        bl_keep.append(t_rel)
                if len(lst) >= TOPK:
                    break
            rows.append(lst)
        # pref already carries MegaLoc marker when ENU (zcME/zc9ME); else append M
        mpref = pref if pref.endswith('M') or pref.endswith('ME') else pref + 'M'
        out = f'{OUT_DIR}/{mpref}_{q}_q-retrieval_db-step=1_topk={TOPK}.npy'
        rows = [r for r in rows if r]  # drop queries with zero valid-baseline candidates (ZcRelpose would crash on empty)
        arr = np.empty(len(rows), dtype=object)
        arr[:] = rows
        np.save(out, arr, allow_pickle=True)
        bl_all, bl_keep = np.array(bl_all), np.array(bl_keep)
        n_empty = sum(1 for r in rows if not r)
        stats[key] = dict(out=out, n_query=len(rows), n_pairs=int(sum(len(r) for r in rows)),
                          n_query_empty=n_empty,
                          topk_raw_baseline_med_p5_p95_max=[round(float(x), 2) for x in
                              (np.median(bl_all), np.percentile(bl_all, 5), np.percentile(bl_all, 95), bl_all.max())],
                          kept_frac=round(float(len(bl_keep) / len(bl_all)), 3),
                          kept_baseline_med_p5_p95=[round(float(x), 2) for x in
                              (np.median(bl_keep), np.percentile(bl_keep, 5), np.percentile(bl_keep, 95))],
                          cand_per_query_med=int(np.median([len(r) for r in rows])))
        print(json.dumps(stats[key], ensure_ascii=False))
    # overlap vs NetVLAD (map only) or vs map-MegaLoc (ENU): same frame ids, cand sets
    for key, (root_rel, db, q, mlf, pref) in SC.items():
        mpref = pref if pref.endswith('M') or pref.endswith('ME') else pref + 'M'
        mg = np.load(f'{OUT_DIR}/{mpref}_{q}_q-retrieval_db-step=1_topk=10.npy', allow_pickle=True)
        if _ENU:
            # compare to map MegaLoc pairs (zcM_/zc9M_)
            map_pref = 'zc9M' if key == '0923' else 'zcM'
            ref_path = f'{OUT_DIR}/{map_pref}_{q}_q-retrieval_db-step=1_topk=10.npy'
            label = 'overlap_map_megaloc_top10'
        else:
            map_pref = 'zc9' if key == '0923' else 'zc'
            ref_path = f'{OUT_DIR}/{map_pref}_{q}_q-retrieval_db-step=1_topk=10.npy'
            label = 'overlap_netvlad_top10'
        if not os.path.isfile(ref_path):
            print(key, 'skip overlap, missing', ref_path)
            continue
        nv = np.load(ref_path, allow_pickle=True)
        # match by frame id (paths differ between enu/map roots)
        def qfid(path):
            return int(os.path.basename(path).split('frame-')[-1][:6])
        mg_by_q = {qfid(r[0]['query_path']): {p['db_idx'] for p in r} for r in mg}
        inter = []
        for a in nv:
            sb = mg_by_q.get(qfid(a[0]['query_path']))
            if sb is None:
                continue
            sa = {p['db_idx'] for p in a}
            inter.append(len(sa & sb))
        if inter:
            stats[key + '_' + label] = dict(mean=round(float(np.mean(inter)), 2),
                                            med=int(np.median(inter)))
            print(key, label + ':', stats[key + '_' + label])
    stats_name = 'regen_pairs_megaloc_enu_stats.json' if _ENU else 'regen_pairs_megaloc_stats.json'
    json.dump(stats, open(f'{OUT_DIR}/{stats_name}', 'w'), ensure_ascii=False, indent=1)
    print('wrote', f'{OUT_DIR}/{stats_name}')

if __name__ == '__main__':
    main()
