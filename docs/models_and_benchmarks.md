# 模型与题集版本清单

核对日期：2026-10-30，Asia/Singapore。研究主线为 M1 单题自身反馈与 M2 跨任务组织学习；共享实验后置。

本文件记录本研究沿用的模型池与版本目标，供云端继续实现时核对。模型名称、公开版本目标、实际账号可用性和正式执行冻结是不同状态。本轮没有发送实验模型请求；当前轻量仓库也没有安装题集、评分库或下载镜像。模型与样本规则见[实验计划](experiment_plan.md)和[实验协议](protocol.md)。

## 1. 实验模型：4 个模型、3 家供应商

| 供应商 | 模型名称／版本说明 | 请求中的 `model` | 计划用途 | 官方资料 |
| --- | --- | --- | --- | --- |
| DeepSeek | 请求模型名就是 `deepseek-flash`。价格以官方价格页该模型名一行为准 | `deepseek-flash` | G2／G3 共同池成员 | [价格页（中文）](https://api-docs.deepseek.com/zh-cn/quick_start/pricing) |
| Xiaomi MiMo | MiMo-V2.6-Flash | `mimo-v2.6-flash` | **G1 Main 固定模型**；G2／G3 池成员，Strong 候选 | [中文 OpenAI 兼容接口](https://mimo.mi.com/docs/zh-CN/api/chat/openai-api) |
| 智谱 BigModel | GLM-5.3-Flash | `glm-5.3-flash` | G2／G3 共同池成员，G1 Strong 候选 | [GLM-5.3-Flash（中文）](https://docs.bigmodel.cn/cn/guide/models/vlm/glm-5.3-flash) |
| Xiaomi MiMo | MiMo-V2.6-Pro | `mimo-v2.6-pro` | **G1 Strong**（2026-10-31 业主指定）；G2／G3 池成员 | [中文模型页](https://mimo.mi.com/models/zh-CN/mimo-v2.6-pro) |

G1 Strong 已由业主指定为 `mimo-v2.6-pro`，理由见[协议](protocol.md)。G2 ALO 与 G3 原生 GroupChat 使用同一模型池。规划器固定为 `deepseek-flash`；G3 的角色绑定为 Planner=`deepseek-flash`、Engineer=`mimo-v2.6-flash`、Critic=`glm-5.3-flash`、Verifier=`mimo-v2.6-pro`，选择器为 `deepseek-flash`。本清单没有增加专用规划模型、评分模型或 embedding 模型。GPT／Codex 云端开发助手不作为额外实验被试计入本池。官方标价与计费函数见 `alo_cloud.prices`。

本轮沿用官方按量 API 通道。MiMo Token Plan、GLM Coding Plan 和第三方转发入口有各自的凭据、服务与计费规则，不能无声明混用。密钥不进仓库；真实请求默认关闭。

### 中国站入口与 2026-10-31 参数决定

业主要求四个模型都走中国大陆站点，并同时启用两项参数。下表只记录官方文档能确认的内容。环境变量名保持 `DEEPSEEK_API_KEY`、`MIMO_API_KEY`、`ZAI_API_KEY`。

| 模型 | 请求 `model` | 按量 Chat Completions | 认证 | 本次多送的参数 |
| --- | --- | --- | --- | --- |
| `deepseek-flash` | `deepseek-flash` | `https://api.deepseek.com/chat/completions` | `Authorization: Bearer` | 不额外发送 thinking；中文文档写明省略 thinking 与 `reasoning_effort` 时默认开启思考 |
| MiMo-V2.6-Flash / Pro | `mimo-v2.6-flash`、`mimo-v2.6-pro` | `https://api.xiaomimimo.com/v1/chat/completions` | `api-key` | `max_completion_tokens`，不并送 `max_tokens` |
| GLM-5.3-Flash | `glm-5.3-flash` | `https://open.bigmodel.cn/api/paas/v4/chat/completions` | `Authorization: Bearer` | `thinking.type=enabled` |

依据：

- DeepSeek 中文[首次调用](https://api-docs.deepseek.com/zh-cn/)把 OpenAI `base_url` 写成 `https://api.deepseek.com`，模型名使用 `deepseek-flash`。中文[价格页](https://api-docs.deepseek.com/zh-cn/quick_start/pricing)的模型列也是这个名字。中文文档没有另给一个大陆专用主机名。认证见[对话补全](https://api-docs.deepseek.com/zh-cn/api/create-chat-completion)。
- MiMo 中文[首次调用](https://mimo.mi.com/docs/zh-CN/quick-start/first-api-call)把按量 OpenAI 兼容地址写成 `https://api.xiaomimimo.com/v1`。`token-plan-cn.xiaomimimo.com` 和 `batch-api-cn.xiaomimimo.com` 属于 Token Plan 与批量推理，不是本轮按量通道。[OpenAI 兼容接口](https://mimo.mi.com/docs/zh-CN/api/chat/openai-api)列出的模型 ID 含 `mimo-v2.6-flash` 与 `mimo-v2.6-pro`，输出上限字段是 `max_completion_tokens`（含可见输出和推理 token），curl 使用 `api-key`。该页写明有两种认证方式，但给出的请求头示例和 curl 只有 `api-key`。英文 FAQ 另写了 `Authorization: Bearer` 也可使用（<https://mimo.mi.com/docs/en-US/quick-start/faq/api-integration>）；实现跟随中文 curl，只送 `api-key`。
- 智谱中文[对话补全](https://docs.bigmodel.cn/api-reference/%E6%A8%A1%E5%9E%8B-api/%E5%AF%B9%E8%AF%9D%E8%A1%A5%E5%85%A8)的服务器是 `https://open.bigmodel.cn/api/`，路径 `/paas/v4/chat/completions`，认证为 Bearer。[GLM-5.3-Flash](https://docs.bigmodel.cn/cn/guide/models/vlm/glm-5.3-flash)的模型代码是 `glm-5.3-flash`，并写明 `thinking.type` 只能为 `enabled`。[深度思考](https://docs.bigmodel.cn/cn/guide/capabilities/thinking)说明对 GLM-5.3-FLASH 传 `disabled` 会报错。此前的 `api.z.ai` 是国际站文档入口，不再使用。模型页还推荐 `reasoning_effort: max` 和 `thinking.clear_thinking: false`；这两项不是本次业主决定，请求里不送。

token 入账：三家能核对到的 usage 都把 `total_tokens` 定义成提示加补全。DeepSeek 与 MiMo 的文档把 `reasoning_tokens` 放在 `completion_tokens_details` 里，作为补全明细；MiMo 的按量响应也会在 `usage` 顶层再给同一个计数。智谱启用思考的官方示例只有 `prompt_tokens`、`completion_tokens` 与 `total_tokens`（`8 + 239 = 247`），思考文本在 `reasoning_content`。因此两处 `reasoning_tokens` 都接受，同时出现时必须相等，并且小于等于 `completion_tokens`，不能再加一次。超出量不超过 64，或不大于 completion 的 1% 时，把 reasoning 钳到 `completion_tokens`，空内容且 `finish_reason=length` 仍按报告的 usage 记为 `empty_completion`。再大的超出仍是 unknown。账本按 `total_tokens` 扣减，且要求它等于 `prompt_tokens + completion_tokens`。明细缺失时不把 reasoning 记成测得的 0。若计数超出上述余量、钳位后两处不一致，或 `total_tokens` 对不上，该次结果视为 unknown 并停止后续派发。`content` 为 null 或空字符串、但 usage 满足上述恒等式时，按报告的 usage 结算，可见回答记为空；这不是 unknown。

四个模型的补全都包含推理。调用方要求至少 4096 时发送 16384，预留按 16384 计算。请求使用 SSE：DeepSeek 文档写明最后一个块携带 usage，并支持 `stream_options.include_usage`；智谱流式文档写明 usage 只在最后一个 chunk；MiMo 文档写明 `stream=true` 时用 SSE 增量返回 `reasoning_content` 和 `content`。读超时与沙箱命令分开，MiMo-V2.6-Flash 与 MiMo-Pro 为 600 秒，DeepSeek 与 GLM 为 300 秒。

### 模型版本仍需冻结的字段

- **请求名不等于不可变权重快照。** `deepseek-flash` 是价格页和调用页上的模型名，不是一份可以单独请求的权重快照。其他模型也不能仅凭名称假定服务端权重永久不变。
- 首次真实接口验证需记录 provider、入口、请求模型、返回模型、调用日期和官方可用的 release／snapshot 身份。未提供更细版本时，明确报告服务版本可变的限制。
- 采样、`reasoning_effort` 和 `clear_thinking` 仍未决定。GLM-5.3-Flash 的 thinking 已按业主决定开启；不能把这一项推广成四模型共用的思考开关。DeepSeek 在省略 thinking 时的默认行为以上文中文文档为准，本传输不另发该字段。
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
