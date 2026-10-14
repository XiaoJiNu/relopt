"""M2: RTK 锚定 + 杆臂建模 + 跨包朝向偏置校正的 per-frame 真值（优化方案 §4 M2 / §6 第1步）。

产物: relopt/cache/rtk_gt_<bagkey>.npz — 与 extract 帧号(fidx)对齐的
  body_xy/body_yaw : /map 车体位姿（按 eval_zc.py extract 同规则从 mcap 复算配对，逐帧对齐）
  lever_h          : 车体系相机-车体水平杆臂（由 pose.txt 反推，逐帧常数<2cm，兼作外参自检）
  cam_enu          : RTK 锚定 + 杆臂校正后的相机中心 ENU（位置真值）
  off_deg          : /map yaw 对双天线航向的固定偏置（朝向去偏用；跨包 off 差=基准差）
口径: 沿用 gt_analysis5（共享 map->ENU 旋转 w 扫 85-95°，逐包线性 LS 求 (o,u)）。
用法: mv2d python scripts/gt_probes/build_rtk_gt.py [bagkey ...]   （零 GPU；mcap 全扫数分钟/包）
"""
import math
import os
import sys

import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from scipy.spatial.transform import Rotation as Rot

RELOPT = "/data/code/reLocation-all/relopt"
REPO = "/data/code/reLocation-all/reloc3r"
BAGS = {  # bagkey: (group, mcap, zc数据root目录)
    "20260921_064638": ("lawn", "/data/data/reLocationData/zc_data/20260921_064638.mcap", "data/zc_fix"),
    "20260921_062224": ("lawn", "/data/data/reLocationData/zc_data/20260921_062224.mcap", "data/zc_fix"),
    "20260920_081533": ("lawn", "/data/data/reLocationData/zc_data/20260920_081533.mcap", "data/zc_fix"),
    "20260923_055425_real_robot_record_external_lz4": ("bldg",
        "/data/data/reLocationData/zc_data/研发大楼数据/20260923_055425_real_robot_record_external_lz4/bag_0.mcap", "data/zc_0923"),
    "20260923_060415_real_robot_record_external_lz4": ("bldg",
        "/data/data/reLocationData/zc_data/研发大楼数据/20260923_060415_real_robot_record_external_lz4/bag_0.mcap", "data/zc_0923"),
}
TOPICS = ["/map", "/camera_01/color/image_raw", "/ll/position/gps/gnss_fix",
          "/ll/position/gps/heading_deg_stamped"]
IMG_STEP, SYNC_MAX_MS = 5, 100.0  # 与 eval_zc.py extract 一致


def scan(path):
    mt, mp, mq, it, g, ht, hy = [], [], [], [], [], [], []
    with open(path, "rb") as f:
        rd = make_reader(f, decoder_factories=[DecoderFactory()])
        for s, ch, m, dec in rd.iter_decoded_messages(topics=TOPICS):
            t = m.log_time
            if ch.topic == "/map":
                p = dec.pose.pose
                mt.append(t)
                mp.append((p.position.x, p.position.y, p.position.z))
                mq.append((p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w))
            elif ch.topic == "/camera_01/color/image_raw":
                it.append(t)
            elif ch.topic.endswith("gnss_fix"):
                g.append((t, dec.latitude, dec.longitude))
            else:
                ht.append(t)
                hy.append(float(dec.vector.y))
    Rm = Rot.from_quat(np.array(mq)).as_matrix()
    return dict(mt=np.array(mt), mp=np.array(mp), yaw=np.arctan2(Rm[:, 1, 0], Rm[:, 0, 0]),
                it=np.array(it), g=np.array(g), ht=np.array(ht), hy=np.array(hy))


def enu_at(lat, lon, lat0, lon0):
    Re, fi = 6378137.0, 1 / 298.257223563
    e2 = fi * (2 - fi)
    la0, lo0 = math.radians(lat0), math.radians(lon0)
    N0 = Re / math.sqrt(1 - e2 * math.sin(la0) ** 2)
    M = Re * (1 - e2) / (1 - e2 * math.sin(la0) ** 2) ** 1.5
    return np.stack([(np.radians(lon) - lo0) * N0 * math.cos(la0), (np.radians(lat) - la0) * M], 1)


def fit_lever(pmap, yaw, enu, th):
    """enu = e^{i th}(p + R_yaw v) + o：固定 th 后对 (o, u=e^{i th}v) 线性 LS。同 gt_analysis5。"""
    w = complex(math.cos(th), math.sin(th))
    pc = pmap[:, 0] + 1j * pmap[:, 1]
    ey = np.exp(1j * yaw)
    rhs = (enu[:, 0] + 1j * enu[:, 1]) - w * pc
    A = np.zeros((len(pmap) * 2, 4)); b = np.zeros(len(pmap) * 2)
    A[0::2, 0] = ey.real; A[0::2, 1] = -ey.imag; A[0::2, 2] = 1.0
    A[1::2, 0] = ey.imag; A[1::2, 1] = ey.real; A[1::2, 3] = 1.0
    b[0::2] = rhs.real; b[1::2] = rhs.imag
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    u = complex(sol[0], sol[1]); o = complex(sol[2], sol[3])
    resid = np.abs((w * pc + ey * u + o) - (enu[:, 0] + 1j * enu[:, 1]))
    return w, u, o, resid


def joint_fit(P):
    best = None
    for thd in np.arange(85.0, 95.01, 0.02):
        tot = [fit_lever(v["fit_p"], v["fit_yaw"], v["fit_enu"], math.radians(thd))[3] for v in P.values()]
        m = float(np.median(np.concatenate(tot)))
        if best is None or m < best[0]:
            best = (m, thd)
    thd = best[1]
    F = {n: dict(zip(("w", "u", "o", "resid"), fit_lever(v["fit_p"], v["fit_yaw"], v["fit_enu"], math.radians(thd))))
         for n, v in P.items()}
    return thd, F


def yaw_offset(d):
    """off = median( /map yaw - (-heading_rad) )；heading_deg 为 CW-from-north（同 gt_analysis5）。"""
    idx = np.clip(np.searchsorted(d["mt"], d["ht"]), 0, len(d["mt"]) - 1)
    diff = np.angle(np.exp(1j * (d["yaw"][idx] - np.radians(-d["hy"]))))
    return float(np.median(diff))


def build(bagkey):
    group, path, rootsub = BAGS[bagkey]
    d = scan(path)
    sel = d["it"][::IMG_STEP]
    i = np.clip(np.searchsorted(d["mt"], sel), 1, len(d["mt"]) - 1)
    j = np.where(np.abs(d["mt"][i] - sel) <= np.abs(d["mt"][i - 1] - sel), i, i - 1)
    ok = np.abs(d["mt"][j] - sel) <= SYNC_MAX_MS * 1e6
    fidx = np.arange(len(sel))[ok]
    body_xy, body_yaw = d["mp"][j][ok][:, :2], d["yaw"][j][ok]
    pdir = f"{REPO}/{rootsub}/{bagkey}/seq-01"
    names = sorted(n[:-len(".pose.txt")] for n in os.listdir(pdir) if n.endswith(".pose.txt"))
    assert len(names) == len(fidx), f"{bagkey}: extract 帧数 {len(names)} != 复算 {len(fidx)}"
    Pcam = np.array([np.loadtxt(f"{pdir}/{n}.pose.txt") for n in names])
    # pose.txt=cam@map; /map=body@map, 同为 map 世界系 => (Pcam - Pbody) 旋入车体系 = 常杆臂
    dxy = Pcam[:, :2, 3] - body_xy
    Rb = np.stack([np.array([[math.cos(y), math.sin(y)], [-math.sin(y), math.cos(y)]]) for y in body_yaw])  # world->body 2D 旋转
    lever_h = np.einsum("nij,nj->ni", Rb, dxy)
    if np.std(lever_h, 0).max() > 0.05:
        raise AssertionError(f"{bagkey}: cam lever std={np.std(lever_h, 0)} (外参链不稳)")
    if np.std(lever_h, 0).max() > 0.02:
        print(f"  WARN {bagkey}: cam lever std={np.std(lever_h, 0).round(3)} m (>2cm, /map 量化噪声)")
    g = d["g"][::10]
    gi = np.clip(np.searchsorted(d["mt"], g[:, 0]), 0, len(d["mt"]) - 1)
    return dict(group=group, fidx=fidx, body_xy=body_xy, body_yaw=body_yaw, lever_h=lever_h,
                cam_xy=Pcam[:, :2, 3],
                fit_p=d["mp"][gi][:, :2], fit_yaw=d["yaw"][gi], fit_latlon=g[:, 1:3],
                off_deg=yaw_offset(d), rootsub=rootsub)


def build_all(keys=None):
    groups = {}
    for k in (keys or BAGS):
        groups.setdefault(BAGS[k][0], []).append(k)
    os.makedirs(f"{RELOPT}/cache", exist_ok=True)
    for grp, kg in groups.items():
        P = {n: build(n) for n in kg}
        lat0 = min(p["fit_latlon"][:, 0].min() for p in P.values())
        lon0 = min(p["fit_latlon"][:, 1].min() for p in P.values())
        for p in P.values():
            p["fit_enu"] = enu_at(p["fit_latlon"][:, 0], p["fit_latlon"][:, 1], lat0, lon0)
        thd, F = joint_fit(P)
        print(f"[{grp}] shared map->ENU rotation = {thd:.3f} deg")
        for n, p in P.items():
            f = F[n]
            off_rad = p["off_deg"]  # yaw_offset 返回弧度
            # 拟合式对同一刚体上任一点成立: enu(pt) = w·p_map(pt) + o。
            # 相机中心直接取 pose.txt 平移 => cam_enu = w·cam_map + o（杆臂已含在 pose.txt，无需再加）。
            cam_enu = f["w"] * (p["cam_xy"][:, 0] + 1j * p["cam_xy"][:, 1]) + f["o"]
            np.savez_compressed(f"{RELOPT}/cache/rtk_gt_{n}.npz",
                                fidx=p["fidx"], body_xy=p["body_xy"], body_yaw=p["body_yaw"],
                                cam_map=p["cam_xy"], cam_enu=np.stack([cam_enu.real, cam_enu.imag], 1),
                                lever_h=p["lever_h"], off_rad=off_rad, theta_deg=thd,
                                w=[f["w"].real, f["w"].imag], o=[f["o"].real, f["o"].imag],
                                u=[f["u"].real, f["u"].imag])
            r = f["resid"]
            print(f"  {n}: resid p50/p90={np.median(r):.3f}/{np.percentile(r, 90):.3f} m "
                  f"| ant lever u=({f['u'].real:+.3f},{f['u'].imag:+.3f}) | off={math.degrees(p['off_deg']):.3f}° "
                  f"| frames={len(p['fidx'])}")


if __name__ == "__main__":
    build_all(sys.argv[1:] or None)
