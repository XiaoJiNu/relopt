# relopt — 量产重定位优化实现仓

实现 `docs/算法优化/量产重定位方案问题分析与优化路线_202610081143.md` 中的优化方案。
实验结果与 `docs/自研数据复现/自研数据测试topmetric效果_202609291235.md`（及同目录归档表）对比。

## 分支

| 分支 | 内容 |
| ---- | ---- |
| `main` | 文档 + 骨架 |
| `opt-a` | 方案 A：Reloc3r+MegaLoc+sim3 路线（P0-2 encoder 缓存、P1-2 sim5+bm3、P1-1 EKF 融合原型）；**汇总记录文档在本分支** `docs/汇总/` |
| `opt-b` | 方案 B：XFeat 路线（P0-1 R 组合修复 + 逐帧 dump、P1-2 top_k/守卫角 sweep） |
| `grok-bot` | ZC Phase0–2 ENU + Phase2后 P1–P7：文档、探针、ENU 旁路；`scripts/reloc3r_zc/` 镜像评测/微调入口 |

## 数据与环境（与既有复现完全相同）

- GPU：RTX 5090 Laptop；conda env `mv2d`（`/data/software/conda/anaconda3/envs/mv2d/bin/python`）
- 数据（不复制，直接引用绝对路径，与 reloc3r/accelerated_features 既有脚本同口径）：
  - 图像+位姿：`/data/code/reLocation-all/reloc3r/data/zc_fix`（草坪 db=064638，同天 q=062224，跨天 q=081533）、`.../zc_0923`（db=055425，q=060415）
  - 检索候选对：`/data/code/reLocation-all/reloc3r/_db-q_pair_info/*_topk=10.npy`
  - MegaLoc 全库相似度排名：`/data/code/reLocation-all/MegaLoc/outputs/eval_zc_megaloc/res__*.npz`
  - 权重：HF `siyan824/reloc3r-512`（本地已缓存）、XFeat（accelerated_features hubconf）
- 上游代码复用（sys.path 引入，不 fork）：`reloc3r`、`accelerated_features`、`AsymLoc/scripts`

## 产物约定

- `scripts/` 评测/导出脚本；`results/` 逐帧 npz 与聚合 json（入库，体量小）；`cache/` encoder 特征缓存（不入库，gitignore）。
- 每阶段验收门槛见优化路线文档 §4；实测数字与归档对比写入 `docs/汇总/优化实施记录*.md`（opt-a）。
