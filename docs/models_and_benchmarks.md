# 模型与题集版本清单

核对日期：2026-10-30，Asia/Singapore。研究主线为 M1 单题自身反馈与 M2 跨任务组织学习；共享实验后置。

本文件记录本研究沿用的模型池与版本目标，供云端继续实现时核对。模型名称、公开版本目标、实际账号可用性和正式执行冻结是不同状态。本轮没有发送实验模型请求；当前轻量仓库也没有安装题集、评分库或下载镜像。模型与样本规则见[实验计划](experiment_plan.md)和[实验协议](protocol.md)。

## 1. 实验模型：4 个模型、3 家供应商

| 供应商 | 模型名称／版本说明 | 请求中的 `model` | 计划用途 | 官方资料 |
| --- | --- | --- | --- | --- |
| DeepSeek | DeepSeek-V4.1-Flash；官方当前将别名映射到 **DeepSeek-V4.1-Flash-1017** | `deepseek-flash` | G2／G3 共同池成员，G1 Strong 候选 | [模型与版本说明](https://api-docs.deepseek.com/quick_start/pricing/) |
| Xiaomi MiMo | MiMo-V2.6-Flash | `mimo-v2.6-flash` | **G1 Main 固定模型**；G2／G3 池成员，Strong 候选 | [官方模型页与 API 示例](https://mimo.mi.com/models/en-US/mimo-v2.6-flash) |
| Z.AI | GLM-5.3-Flash | `glm-5.3-flash` | G2／G3 共同池成员，G1 Strong 候选 | [官方模型代码](https://docs.z.ai/guides/vlm/glm-5.3-flash) |
| Xiaomi MiMo | MiMo-V2.6-Pro | `mimo-v2.6-pro` | G2／G3 共同池成员，G1 Strong 候选 | [官方模型页与 API 示例](https://mimo.mi.com/models/en-US/mimo-v2.6-pro) |

G1 Strong 尚未选定，从上述四候选在独立开发题上的事前规则选出；若与 Main 相同则合并执行。G2 ALO 与 G3 原生 GroupChat 使用同一模型池，角色、规划器及选择器绑定在正式合同中另行冻结。本清单没有增加专用规划模型、评分模型或 embedding 模型。GPT／Codex 云端开发助手不作为额外实验被试计入本池。

本轮沿用官方按量 API 通道的设计。MiMo Token Plan、GLM Coding Plan 和第三方转发入口有各自的凭据、服务与计费规则，不能无声明混用。新仓库不含任何密钥或可直接调用的 provider 实现。

### 模型版本仍需冻结的字段

- **API 别名不等于不可变权重快照。** `deepseek-flash` 当前的 1017 映射是本次官方文档中的版本说明，不应将它当成已经验证可单独请求的 snapshot ID。其他模型也不能仅凭名称假定服务端权重永久不变。
- 首次真实接口验证需记录 provider、入口、请求模型、返回模型、调用日期和官方可用的 release／snapshot 身份。未提供更细版本时，明确报告服务版本可变的限制。
- thinking、reasoning effort、采样参数、输出限制和计费 usage 语义须按每模型能力冻结。尤其 GLM-5.3-Flash 官方说明 thinking 只能启用，不能假定四模型均能用同一种关闭思考配置。
- 官方页面可确认名称与公开接口说明；账号权限、余额、实际返回身份和请求可达性仍待有界接口验证。每模型累计 50 CNY、总计 200 CNY 的约束继续有效，期初使用和未决预留另在私有运行存储绑定。

## 2. 题集：SWE-bench Lite 与 ProgramBench

以下为沿用的固定版本目标。SHA 是具体对象的 revision，不能在不同仓库或不同层之间互换，也不使用 `main`／`latest` 代替它们。

### SWE-bench Lite

| 字段 | 值 |
| --- | --- |
| 数据集 | `princeton-nlp/SWE-bench_Lite` |
| 数据集 revision | `6ec7bb89b9342f664a54a6e0a6ea6501d3437cc2` |
| Hugging Face config | `default` |
| 官方 split | `dev`：23 条；`test`：300 条 |
| 本研究用途 | Strong 选型从 `dev` 中选；正式 M1、M2 学习和 probe 从 `test` 中按用途不重叠选取 |
| 评分库参考版本 | `swebench==4.1.0`，在目标云环境另行安装与验收 |
| 官方固定来源 | [数据 revision](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite/tree/6ec7bb89b9342f664a54a6e0a6ea6501d3437cc2)、[该 revision 的数据卡](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite/blob/6ec7bb89b9342f664a54a6e0a6ea6501d3437cc2/README.md)、[评分库 4.1.0](https://pypi.org/project/swebench/4.1.0/) |

本研究使用 Lite。数据集 revision、`swebench` 软件包版本、题目仓库的 `base_commit` 和逐题环境镜像 digest 是四类不同身份。最终合同还需记录实际安装文件／评分源码和依赖身份；只写 `swebench==4.1.0` 不构成完整评分环境锁。

### ProgramBench

ProgramBench 必须同时记录**运行库／任务 metadata**和**行为测试资产**两层。

| 对象 | 固定版本目标 | 说明 |
| --- | --- | --- |
| 上游源码仓库 | `facebookresearch/ProgramBench` | [上游仓库](https://github.com/facebookresearch/ProgramBench) |
| 运行库／metadata 源码 commit | `895f65916bc5ba34a915d36da321fde0301125ad` | [固定源码](https://github.com/facebookresearch/ProgramBench/tree/895f65916bc5ba34a915d36da321fde0301125ad)；逐题 `task.yaml`／`tests.json` 的来源须随实际使用内容绑定 |
| 该源码声明的 Python 包版本 | `programbench==1.2.5` | [固定源码的包声明](https://github.com/facebookresearch/ProgramBench/blob/895f65916bc5ba34a915d36da321fde0301125ad/pyproject.toml)；单独一个包号不能替代源码 commit 与成员身份 |
| Hugging Face 行为测试资产仓库 | `programbench/ProgramBench-Tests` | 独立于运行库源码的测试资产仓库 |
| 测试资产 revision | `de0ddfb637590c7ecb54fa0b5301f6dc7dfbcee5` | [固定测试资产 revision](https://huggingface.co/datasets/programbench/ProgramBench-Tests/tree/de0ddfb637590c7ecb54fa0b5301f6dc7dfbcee5) |
| 本研究划分 | 自建不重叠的 Strong 开发／M1／M2 学习／probe 任务清单 | 当前读取方式从本地逐题 metadata 构造实例，不使用官方 HF 表格的 `dev`／`test` split；不能把本研究用途划分称为官方 split |

上游固定 commit 可定位，并且其 `pyproject.toml` 声明版本为 1.2.5；这不证明任意同名本地目录或任意 PyPI 1.2.5 安装与该源码字节相同。后续必须从明确来源获取、核对实际文件与评分依赖，再形成新平台的执行合同。

`de0ddfb...` 是 **HF 测试资产 revision**，不是 ProgramBench Python 包版本或 GitHub 源码 commit；`895f659...` 则是 **源码 commit**，不能替代测试资产 revision。测试资产和期望值只交给隔离评价侧，不写入新仓库或作答上下文。

## 3. 版本与样本的对应

| 阶段 | SWE-bench Lite | ProgramBench |
| --- | --- | --- |
| Strong 开发选型 | 固定 revision 的 `dev` 中 3 道新题 | 固定 metadata 范围中自建开发划分 3 道新题 |
| M1 正式比较 | 固定 revision 的 `test` 中 6 道 | 与开发划分不重叠的 6 道 |
| M2 学习 | `test` 中另选 4 道 | 与前述用途不重叠的 4 道 |
| M2 独立 probe | `test` 中另选 2 道 | 与前述用途不重叠的 2 道 |

每轨 15 个新任务身份，两轨共 30 个，其中正式部分 24 个。SWE 按评价侧参考 patch 的改动规模覆盖，ProgramBench 按上游 `difficulty` 覆盖；具体任务 ID、候补与曝光排除仍待正式冻结，不从版本清单直接生成可执行题单。

## 4. 后续逐题／逐模型锁定项

- [ ] 实际模型身份、服务通道、参数和 usage／费用上界。
- [ ] 新任务 ID、来源、暴露排除、用途和候补清单。
- [ ] SWE `base_commit`、环境身份及评分代码／依赖完整来源。
- [ ] ProgramBench 逐题 metadata、测试文件的实际哈希及独立评分来源。
- [ ] 每题候选／评价镜像的不可变 digest；镜像在独立运行存储中准备。
- [ ] 目标原生 Linux 环境的正负控制、资源与恢复接受，以及完整研究执行合同。

本清单只提供公开名称、版本目标和用途说明，不附数据、测试 blob、镜像、审计结果、账本或凭据。当前轻量安装脚本不会据此自动下载或安装评分资产。
