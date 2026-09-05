# AVIC × ApexNav 集成说明

本分支在 ApexNav 上增加了一个可选的 AVIC/SVC 视觉想象通道。ApexNav
仍负责真实观测、地图、frontier、路径规划和动作执行；外部服务只返回当前
frontier 决策使用的临时分数。服务异常时系统回退到原始 ApexNav。

## 代码边界

- `avic_bridge/controller.py`：运行在 ApexNav 环境，完成门控、真实锚点匹配、
  缓存和请求构造。
- `avic_bridge/service.py`：运行在独立 AVIC 环境，编排 Qwen、SVC 和多帧校验。
- `avic_bridge/svc_adapter.py`：惰性加载外部 Stable Virtual Camera 实现。
- `src/planner/exploration_manager/`：发布 frontier 候选并应用一次性临时分数。
- `src/planner/plan_env/`：提供真实观测覆盖度，不写入想象结果。

仓库不包含 Stable Virtual Camera 源码、Qwen 模型、模型权重或 Habitat 数据。

## 配置

先完成上游 ApexNav 的安装，再准备一个包含 SVC 依赖的独立 Python 环境和
外部 AVIC 源码目录：

```bash
cp config/avic.env.example config/avic.env
```

编辑 `config/avic.env` 中的 Conda、AVIC、vLLM 和 Hugging Face 路径。该文件
不会提交到 Git。脚本不再依赖固定用户名或 `/data/wzy/...` 路径。

## 最小验证

只验证 HTTP 协议，不加载 SVC：

```bash
./scripts/run_avic_service.sh --backend mock
```

在另一个终端使用一张本地图片发送请求：

```bash
python scripts/smoke_avic_bridge.py --image /path/to/rgb.jpg
```

执行一条完整 HM3D v2 任务：

```bash
./scripts/run_apexnav.sh \
  --dataset hm3dv2 \
  --avic-mode avic \
  --episode-manifest experiments/manifests/hm3dv2_clear_targets_50_habitat.json \
  --max-episodes 1
```

纯 ApexNav 对照只需把 `--avic-mode avic` 改为 `--avic-mode apexnav`。

## 当前算法约束

1. 只有真实语义分数进入固定中间区间、Top-2 分差较小、真实观测不足的
   frontier 才能触发想象。
2. 每个候选必须绑定 episode 历史中的真实方向锚点。
3. Qwen 只能选择固定动作轨迹的前缀长度，不能生成新动作或直接控制机器人。
4. SVC 序列必须通过可靠性和新增证据阈值，才能获得有上限的非负临时奖励。
5. 临时分数不进入 ValueMap、occupancy map 或 ESDF；失败路径保持原始分数。

完整参数可通过 `./scripts/run_apexnav.sh --help` 查看。当前实验结论见
[CURRENT_RESULTS.zh-CN.md](CURRENT_RESULTS.zh-CN.md)。

## 数据与输出策略

- `data/`、模型权重、Hugging Face 缓存始终放在 Git 之外。
- 默认日志写入 `logs/`，默认实验结果写入 `experiments/runs/`；两者均被忽略。
- `pred.mp4`、SVC 帧、回放 `.npz` 和构建目录不进入源码仓库。
- 仓库只保留可复现的代码、配置模板、episode manifest 和文字化指标结论。
