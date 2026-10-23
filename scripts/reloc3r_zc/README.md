# reloc3r_zc — Phase1/2 + P1–P7 相关入口镜像

从 `reloc3r-grok-bot`（`XiaoJiNu/reloc3r` 分支 `grok-bot`）同步进 **relopt**，便于单仓查阅与复现。

| 文件 | 作用 |
| ---- | ---- |
| `eval_zc_sim3.py` | 主评测：`POSE_TAG=enu`、`SIM3_AGG=dualgate`、`SIM3_REJ_M`、`APPLY_OFF`（默认关）、`WEIGHTS` 本地加载 |
| `finetune_zc_only.sh` | Phase1 ZC-only 微调入口 |
| `finetune_zc_only_enu.sh` | Phase2 ENU 微调入口 |
| `finetune_zc.sh` | 含 PAIRSET/LR 的微调入口（历史臂） |
| `zc_retrieval.py` | ZC 数据集（`data_root` / parity）；覆盖到 `reloc3r/datasets/` 使用 |
| `eval_forgetting.py` | 遗忘检查（`ZC_ONLY=1`） |
| `m3m4_auto.sh` | M1/M3/M4 自动回测编排（若存在） |

**来源 tip（同步时）**：`reloc3r` `grok-bot` @ `badea1e` (`badea1eaeeac9f4a6271cda95346f5870222c209`)

**运行说明**：训练/评测仍建议在 `/data/code/reLocation-all/reloc3r-grok-bot` 工作树执行（依赖 croco 子模块与 `data/` symlink）。本目录是 **relopt 内归档镜像**；改完后请两边对齐。镜像相对源去掉了 APPLY_OFF 补丁误加的二次 `load_images`。

探针与 ENU 旁路脚本在上级 `scripts/` 与 `scripts/gt_probes/`（已在 relopt 主线）。
