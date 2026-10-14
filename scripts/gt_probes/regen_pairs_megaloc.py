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

SC = {  # key: (data_root_rel, db_bag, q_bag, megalloc_npz, out_prefix)
 '062224': ('./data/zc_fix', '20260921_064638', '20260921_062224',
            f'{ML}/res__mowing__same_day_062224.npz', 'zc'),
 '081533': ('./data/zc_fix', '20260921_064638', '20260920_081533',
            f'{ML}/res__mowing__cross_day_081533.npz', 'zc'),
 '0923': ('./data/zc_0923', '20260923_055425_real_robot_record_external_lz4',
          '20260923_060415_real_robot_record_external_lz4',
          f'{ML}/res__building_0923__query_060415.npz', 'zc9'),
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
        out = f'{OUT_DIR}/{pref}M_{q}_q-retrieval_db-step=1_topk={TOPK}.npy'
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
    # overlap vs NetVLAD pairs (sanity: same frames, different candidate sets)
    for key, (root_rel, db, q, mlf, pref) in SC.items():
        nv = np.load(f'{OUT_DIR}/{pref}_{q}_q-retrieval_db-step=1_topk=10.npy', allow_pickle=True)
        mg = np.load(f'{OUT_DIR}/{pref}M_{q}_q-retrieval_db-step=1_topk=10.npy', allow_pickle=True)
        mg_by_q = {r[0]['query_path']: {p['db_idx'] for p in r} for r in mg}
        inter = []
        for a in nv:
            sb = mg_by_q.get(a[0]['query_path'])
            if sb is None:
                continue
            sa = {p['db_idx'] for p in a}
            inter.append(len(sa & sb))
        stats[key + '_overlap_netvlad_top10'] = dict(mean=round(float(np.mean(inter)), 2),
                                                     med=int(np.median(inter)))
        print(key, 'overlap with NetVLAD top-10:', stats[key + '_overlap_netvlad_top10'])
    json.dump(stats, open(f'{OUT_DIR}/regen_pairs_megaloc_stats.json', 'w'), ensure_ascii=False, indent=1)

if __name__ == '__main__':
    main()
