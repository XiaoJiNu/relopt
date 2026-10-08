"""合成自检：recoverPose(K.T@F@K, pts_q, pts_d, K) 返回 [R|t]（第二相机=db 相对第一相机=query：
p_db = R p_q + t）⇒ 旋转组合应为 A_q = A_db @ R，射线方向 = A_db @ t̂。
脚本用已知真值（OpenCV 相机约定 + z-up 世界）验证这两个断言。"""
import numpy as np
import cv2

rng = np.random.default_rng(7)
K = np.array([[366.096, 0, 321.335], [0, 366.008, 242.173], [0, 0, 1]], float)


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def heading(A):  # eval_zc_netvlad.heading: 光轴(世界系)方位角 atan2(y, x)
    return np.arctan2(A[1, 2], A[0, 2])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# 相机系 (x右,y下,z前) -> 世界 (z上)：列=相机各轴在世界中的方向；光轴(相机z)=世界 yaw 方向
FIX = np.array([[0., 0, 1], [-1., 0, 0], [0., -1, 0]])  # cam x->-y, cam y->-z, cam z->x (右手系✓)
def make_A(yaw):
    return rot_z(yaw) @ FIX


proj = lambda A, C, X: (K @ (A.T @ (X - C).T)).T
e1, e2, d_ang = [], [], []
for _ in range(2000):
    if len(e1) >= 40:
        break
    yq, yd = rng.uniform(-np.pi, np.pi), rng.uniform(-np.pi, np.pi)
    if abs(wrap(yq - yd)) < 0.2:
        continue
    A_q, A_d = make_A(yq), make_A(yd)
    C_q = np.array([rng.uniform(-2, 2), rng.uniform(-2, 2), 0.6])
    C_d = C_q + rng.uniform(1, 3) * np.array([np.cos(rng.uniform(0, 6.3)), np.sin(rng.uniform(0, 6.3)), 0])
    X = np.c_[rng.uniform(-8, 8, 900), rng.uniform(-8, 8, 900), rng.uniform(0.2, 2.2, 900)]
    pq, pd = proj(A_q, C_q, X), proj(A_d, C_d, X)
    vis = (pq[:, 2] > 2) & (pd[:, 2] > 2)
    pq, pd = pq[vis][:, :2] / pq[vis][:, 2:3], pd[vis][:, :2] / pd[vis][:, 2:3]
    if len(pq) < 100:
        continue
    pq, pd = pq + rng.normal(0, .3, pq.shape), pd + rng.normal(0, .3, pd.shape)
    F, inl = cv2.findFundamentalMat(pq, pd, cv2.FM_RANSAC, 1.0, 0.999)
    m = inl.ravel() == 1
    ok, R, t, _ = cv2.recoverPose(K.T @ F @ K, pq[m], pd[m], K)
    if not ok:
        continue
    e1.append(np.degrees(abs(wrap(heading(A_d @ R) - yq))))        # A_db@R 复原 query 朝向
    e2.append(np.degrees(abs(wrap(heading(A_d @ R.T) - yq))))      # 取反对照
    th = t.ravel() / np.linalg.norm(t)                              # db系中 db->q 单位方向
    u_true = (C_q - C_d) / np.linalg.norm(C_q - C_d)
    d_ang.append(np.degrees(np.arccos(np.clip((A_d @ th) @ u_true, -1, 1))))

f = lambda a: f"中位 {np.median(a):5.2f}°  p90 {np.percentile(a, 90):5.2f}°"
print(f"n={len(e1)}")
print(f"  A_db @ R    query朝向误差: {f(e1)}")
print(f"  A_db @ R.T  query朝向误差: {f(e2)}")
print(f"  A_db @ t̂ vs 真db→q方向:   {f(d_ang)}")
assert len(e1) >= 20, "有效场景太少"
assert np.median(e1) < 1.0 and np.median(e1) < np.median(e2) - 5, "A_db@R 约定被否"
assert np.median(d_ang) < 5.0, "射线方向 A_db@t̂ 被否"
print("→ 与 eval_zc_xfeat_rc 的 rel_yaw(A_db@R) 和 d=A_db@t̂ 一致 ✓")
