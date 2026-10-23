"""Per-candidate relpose dump on the odd eval subset (Phase2 ENU weights) -> enables OFFLINE
aggregation sweeps (bm3 / RANSAC / pairwise-consistency / K choice / oracle) with zero GPU.

Dumps per query: candidate db indices (MegaLoc order top-K), per-candidate predicted q2d,
per-candidate implied absolute query pose, GT query pose, MegaLoc sims.
Usage: WEIGHTS=... K=10 python dump_cands_ft.py <scene_key>
"""
import os, sys, time
import numpy as np
sys.path.insert(0, '/data/code/reLocation-all/reloc3r')
import torch
from reloc3r.reloc3r_relpose import setup_reloc3r_relpose_model, inference_relpose
from reloc3r.utils.image import load_images

ML = "/data/code/reLocation-all/MegaLoc/outputs/eval_zc_megaloc"
FIX = "/data/code/reLocation-all/reloc3r/data"
OUT = f"{FIX}/zc_sim3"
SC = {
 "062224": (FIX + "/zc_fix_enu", "20260921_064638", "20260921_062224", ML + "/res__mowing__same_day_062224.npz"),
 "081533": (FIX + "/zc_fix_enu", "20260921_064638", "20260920_081533", ML + "/res__mowing__cross_day_081533.npz"),
 "0923": (FIX + "/zc_0923_enu", "20260923_055425_real_robot_record_external_lz4",
          "20260923_060415_real_robot_record_external_lz4", ML + "/res__building_0923__query_060415.npz"),
}
K = int(os.environ.get("K", "10"))
WEIGHTS = os.environ.get("WEIGHTS", "")
PARITY = os.environ.get("EVAL_PARITY", "odd")

def _parity(path, par):
    return par == "all" or int(os.path.basename(path).split("frame-")[-1][:6]) % 2 == (par == "odd")

def main(key):
    root, db, q, mlf = SC[key]
    m = np.load(mlf); order, sim = m["order"], m["sim"]
    dbn = sorted(n[:-10] for n in os.listdir(f"{root}/{db}/seq-01") if n.endswith(".color.png"))
    qn_all = sorted(n[:-10] for n in os.listdir(f"{root}/{q}/seq-01") if n.endswith(".color.png"))
    qn = [n for n in qn_all if _parity(n, PARITY)]
    qsel = np.array([i for i, n in enumerate(qn_all) if _parity(n, PARITY)])
    device = torch.device("cuda")
    from reloc3r.reloc3r_relpose import Reloc3rRelpose
    if WEIGHTS:
        model = Reloc3rRelpose(img_size=512).to(device)
        model.eval()
        ck = torch.load(WEIGHTS, map_location=device, weights_only=False)
        ck = ck.get("model", ck)
        ck = {k.replace("dec_blocks2", "dec_blocks"): v for k, v in ck.items()} if any(
            k.startswith("dec_blocks2") for k in ck) else dict(ck)
        print(model.load_state_dict(ck, strict=False))
        print(f"finetuned weights loaded: {WEIGHTS}")
    else:
        model = setup_reloc3r_relpose_model("Reloc3rRelpose(img_size=512)", device)
    db_imgs = load_images([f"{root}/{db}/seq-01/{n}.color.png" for n in dbn], size=512, verbose=False)
    q_imgs = load_images([f"{root}/{q}/seq-01/{n}.color.png" for n in qn], size=512, verbose=False)
    pose = lambda s, n: np.loadtxt(f"{root}/{s}/seq-01/{n}.pose.txt")

    PDB, PQ2D, SIM, IDX, GT = [], [], [], [], []
    t0 = time.time()
    for qi in range(len(qn)):
        sel = [int(j) for j in order[qsel[qi], :K]]
        n = len(sel)
        v1 = {"img": torch.cat([db_imgs[j]["img"] for j in sel], 0),
              "true_shape": torch.as_tensor(np.stack([db_imgs[j]["true_shape"] for j in sel])).reshape(n, 2)}
        v2 = {"img": q_imgs[qi]["img"].repeat(n, 1, 1, 1),
              "true_shape": torch.as_tensor(q_imgs[qi]["true_shape"]).repeat(n, 1)}
        with torch.no_grad():
            q2d = inference_relpose([v1, v2], model, device, use_amp=True)
        PDB.append(np.stack([pose(db, dbn[j]) for j in sel]))
        PQ2D.append(np.stack([p.cpu().numpy() for p in q2d]))
        SIM.append(sim[qsel[qi], sel].astype(np.float32))
        IDX.append(np.array(sel))
        GT.append(pose(q, qn[qi]))
        if qi % 100 == 0:
            print(f"  {qi}/{len(qn)} {time.time()-t0:.0f}s", flush=True)
    out = f"{OUT}/cands_{key}_K{K}_{os.environ.get('FT_TAG','zcOnlyENU')}_{PARITY}.npz"
    np.savez_compressed(out, pdb=np.array(PDB), pq2d=np.array(PQ2D), sim=np.array(SIM),
                        idx=np.array(IDX), gt=np.array(GT), names=np.array(qn))
    print(out, (time.time()-t0)/len(qn)*1000, "ms/frame")

if __name__ == "__main__":
    main(sys.argv[1])
