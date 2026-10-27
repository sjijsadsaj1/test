# ApexNav：Frontier 物品概率预测实验

基于 [ApexNav](https://github.com/Robotics-STAR-Lab/ApexNav) 的研究分支。当前方向是用真实图像推测 frontier 后方各类物品的独立存在概率，辅助探索选路；不生成想象图片或视频。

**目前没有证据表明该方法能稳定提高 SR。** 不把工程运行通过或单任务成功作为提升结论。

## 当前代码

- `Apexnav_1/runtime`：脚本在仓库旁建立的隔离原版对照，锁定上游 `049971b613fcd2554b034bd4a853d79cc1e61613`。
- `habitat_evaluation.py`、`src/planner/`：工作区评测与导航；保留同步观测、路径搜索、TSP恢复等已有改动。关闭模型不等于隔离原版。
- `avic_bridge/frontier_dictionary.py`：概率提示词、21类独立概率、输入输出绑定。
- `avic_bridge/frontier_crop.py`：原地图 frontier 点簇到真实图像的投影适配。
- `avic_bridge/vlm_transport.py`、`text_service.py`：独立VLM接口。当前批量实验使用本地Qwen3-VL。
- `scripts/experiment_utils.py`：任务结果解析、累计耗时读取、空闲端口选择；不再依赖旧实验驱动。

[核心流程与历史方案保留说明](docs/CORE_AND_HISTORY.zh-CN.md) · [计划书](docs/PROJECT_PLAN.zh-CN.md) · [修改日志](docs/CHANGELOG.zh-CN.md)

## 使用已有环境

仍需ROS Noetic、Habitat、原版视觉服务及对应数据集/模型。安装基础见[上游说明](https://github.com/Robotics-STAR-Lab/ApexNav#installation)；本仓库不上传模型、数据集、运行环境或实验输出。

1. 从 `config/avic.env.example` 创建本机 `config/avic.env`，设置Conda、ROS及资源路径。
2. 可从 `config/text_frontier.env.example` 创建 `config/text_frontier.env`，配置独立VLM。未创建时兼容原本机配置；不要将密钥提交到Git。
3. 本地Qwen3-VL复用已有模型根目录的 `processor/` 与 `text_encoder/`。虽然目录可能叫Qwen-Image，本流程不加载图像生成器，**这两个子目录仍是运行依赖**。
4. 部分实验入口仍采用本机 `/data/wzy/miniconda3/envs/apexnav/bin/python` 和 `.runtime/image_glm_env/bin/python`；迁移机器时需先配置/调整路径，不声称零配置可运行。

纯原版（显式任务清单）：

```bash
bash scripts/run_upstream_apexnav.sh --dataset mp3d \
  --episode-manifest experiments/manifests/mp3d_sr_dev100_seed20260911_runtime.json \
  --max-episodes 1 --reuse-existing-perception
```

词表接口单任务小测（最多2次预测，不是SR验证）：

```bash
bash scripts/run_frontier_dictionary_smoke.sh
```

固定前30任务配对入口保留，供用户明确决定扩大实验时使用；高频预测可能很慢，当前并不推荐默认扩大：

```bash
bash scripts/run_frontier_dictionary30.sh
# 已有完全匹配的原版结果可显式复用：
bash scripts/run_frontier_dictionary30.sh --reuse-baseline /absolute/path/to/previous_batch
```

批量脚本打印结果目录和tail命令，不在终端输出 `comparison.json` 内容。模型服务、ROS及导航只有执行运行脚本才启动。

## 清理范围（2026-10-27）

移除旧图生提示词试验、旧50/100任务专项入口和重复文档；细目见 [CLEANUP_MANIFEST.json](docs/CLEANUP_MANIFEST.json)。删除前在本机归档了源码；原始实验数据仍保留本地，但不上传GitHub。

为避免清理改变实验算法，共享控制器、评分协议及C++中尚有依赖的历史分支保留。旧SVC服务仅作兼容，不属于当前词表执行链。主动目标复核代码保留为默认关闭的独立模块，当前不以改进识别能力为方向。

测试（不启动模型）：

```bash
PYTHONPATH="$PWD" /data/wzy/miniconda3/envs/apexnav/bin/python -m unittest discover -s tests -v
PYTHONPATH="$PWD" /data/wzy/miniconda3/envs/apexnav/bin/python scripts/test_frontier_dictionary.py
```

保留上游 [LICENSE](LICENSE) 与 [第三方说明](THIRD_PARTY.md)。本项目是实验分支，不是ApexNav原论文结果的替代实现。
