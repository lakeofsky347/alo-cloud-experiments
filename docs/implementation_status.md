# 初步实现状态

日期：2026-10-30。2026-11-01 增加了 round-2 批号 `r2`（见 [protocol_freeze_r2.md](protocol_freeze_r2.md) 与 README 的运行说明）。r2 不与 r1 结果合并，也不改 r1 冻结哈希。r2 账本的冻结期初是 0：本轮每模型重新获得 50 CNY。`ALO_PRIOR_LEDGER` 不设置就不带入旧账。Pro 的 48 CNY 软停止只看这笔 r2 花费。

本仓库承载 M1／M2 的实验计划和初步代码，**不构成正式就绪包或已完成实验的证据**。研究定义见[实验计划](experiment_plan.md)与[实验协议](protocol.md)。本文件不附模型输出、运行成绩或历史审计材料。

用户指定的[行政输入目录](../study_records/README.md)已提供任务预选、有序候补、历史替换、暴露登记、M2待冻结顺序和私有存储定位／哈希。16份预算证据原件另已补齐到私有Drive并回读验证，代码仓库只记录其定位与校验信息；具体云任务的物化、期初预算绑定、完整正式realization和平台接受仍须分别完成。

## 初步实现的适用范围

代码用于继续开发三组接合、原生会话及检查点、资源计量、反馈和M2状态迁移。具体能力须以本仓库源码和重新运行的验证为准；目录或接口存在不等于对应研究条件已验收。首次云环境设置只验证轻量依赖、模块导入和离线软件测试。原生容器、基准数据和独立评分的接入属于后续阶段。

Linux执行需明确选择受支持的本地Docker后端，核对原生amd64、可用内存与磁盘、容器限额、权限和评分依赖。子进程只接收允许的环境变量；模型凭据不得进入候选容器、镜像层、源代码或测试工件。

当前能力信号的已知边界是可归因 Python 源码语法：B／TMS 的 pass／fail 不代表功能正确性。全局公开质量与资源观测独立建模。M2实验的结论必须与实际继承的信号一致。

r2 冻结路径仍使用该语法判据，不因后续代码改变已经采集的 r2 结果含义。可选 `criterion_id="repro-v1"` 才把复现脚本的修改前／修改后返回码和已改动公开测试的返回码交给 `public_quality_signal`。语法单独通过在这条判据里保持 unknown。B／TMS 仍只在明确 pass 或 fail 时更新。`routing_mode` 的 `fixed` 与 `random`、`g1-deepseek-flash`，以及固定 DAG 和轻量 s2，都是冻结日程之外的对照配置。

| 源码范围 | 本仓库包含什么 | 未接入的部分 |
| --- | --- | --- |
| `src/alo/` | 组织、权限、B／TMS、Thompson 指派、治理和状态存储的核心算法 | 实际模型传输、候选运行与基准任务 |
| `src/alo_cloud/adapters.py` | G1 工具循环、ALO 与原生 GroupChat 的服务接合代码 | 调度器在 `alo_cloud.run`，按冻结日程调用这些 adapter |
| `services.py`、`providers.py`、`sandbox.py` | 显式 `StudyServices`：四模型按量 OpenAI 兼容传输、账本预留／结算、单元限额、`compile()` 语法判据、可选 Docker 命令封装 | 真实密钥、价格、期初账本、网络许可和已接受的沙箱镜像；默认不发请求 |
| `contracts.py`、`state.py`、`checkpoints.py` | M1 顺序、M2 统计白名单与禁更新边界、检查点身份辅助函数 | 完整任务工件封存、跨主机恢复与全套 M1／M2 控制器 |
| `budget.py` | 显式期初占用、唯一请求预留、unknown 阻断；配置了 token／毫秒／工具上限时，`remaining_resources` 给出 requests、tokens、seconds、cny | 真实价格与私有期初对账、不可篡改存储和分布式配额。未配置的维度不会被填成 0 |
| `interfaces.py` | provider、sandbox、独立评分的明确接口；可执行实现位于 `services.py` | `ProviderTransport` 与 `SandboxBackend` 仍抛出未配置异常 |

`local_criterion` 对本次 `write_file` 归因的 Python 文件调用 `compile()`：只有 `SyntaxError` 是可靠失败，通过编译是可靠语法通过，非 Python、无法归因的命令写入和读取错误保持 unknown。这不是功能正确性评分。Docker 命令封装默认不执行；离线测试注入 runner，不连接 daemon。原生 GroupChat 的软件 fixture 需显式安装可选 `native` 依赖；未安装时对应测试应明确跳过。

## 正式采集前仍需完成

| 项目 | 要求 |
| --- | --- |
| 云环境 | 在目标Linux环境重新验证原生架构、Docker、资源、依赖和失败诊断 |
| 题集 | 固定数据、任务ID、暴露划分、镜像及原生评分身份；逐题正控、空候选和定向负控通过 |
| 模型 | 固定真实provider／模型身份、共同候选资格、规划器和原生GroupChat配置 |
| Strong | 业主已指定 `mimo-v2.6-pro`；开发集四候选 B0 不进入本轮付费日程 |
| 反馈 | 冻结每题公开检查，区分提交、局部语法、全局质量、资源和unknown |
| M1 | 验证首轮封存、自身B1完整恢复、题间重置、外层顺序和独立终评 |
| M2 | 验证白名单迁移、reset先验、全部学习→固定参考→终态probe顺序及probe禁更新 |
| 预算 | 在私有运行存储核对每模型累计使用与未决预留；不得按新仓库重置50 CNY上限 |
| 冻结 | 绑定代码版本、最终任务、阶段、预算、顺序、模型及评分合同后才可启动对应采集 |

`StudyServices` 在离线测试里仍可注入 sender。付费路径使用 `alo_cloud.prices.charge_microcny` 和按提示字节上限加 `max_completion_tokens` 算出的单次预留。传输仍是中国站按量入口：MiMo 发送 `max_completion_tokens`，GLM-5.3-Flash 发送 `thinking.type=enabled`；tokens 按 `prompt + completion` 入账，reasoning 明细不重复加计。空的可见回答在 usage 自洽时按报告用量结算；可解析的供应商错误按 0 结算；超时、连接中断和无法解析的 200 才保持 unknown。诊断写在运行目录的 `diagnostics/`。unknown 打断的单元可重启一次。四个模型在长请求上发送 16384 并使用 SSE。供应商读超时与命令超时分开。超时按预留最坏金额结算后日程继续，同一单元只自动重启一次。`list_files` 最多返回 200 条路径。单个 `<tool_call>` 会转成公开 JSON 工具。调度器是 `python -m alo_cloud.run`。收卷交接是 `python -m alo_cloud.collect`：每个 SWE 单元一条 prediction，ProgramBench 按单元目录放 `submission.tar.gz`。SWE 工作区从官方实例镜像的 `/testbed` 拷出后以 `--network=none` 挂回 `/testbed`：接受 HEAD 等于 `base_commit`，或 HEAD 的父提交等于 `base_commit` 且该提交只改文件模式；计分补丁相对镜像 HEAD。ProgramBench 使用 `task_cleanroom_v6` 镜像，命令前后的目录同步排除 `.git` 和 `executable`。题面包含官方 `compile.sh` / `./executable` 契约。评分容器核数取 `PROGRAMBENCH_DOCKER_CPUS`，未设置时为 `min(10, nproc)`。`--live` 才读 `DEEPSEEK_API_KEY`、`MIMO_API_KEY`、`ZAI_API_KEY` 并允许网络。账本期初按业主决定为 0，因为检出里没有数值期初占用。采样、`reasoning_effort` 和 `clear_thinking` 仍不发送。独立评分在 `alo_cloud.evaluate`，没有模型调用。离线测试通过不是模型分数，也不是采集已经开始。

离线测试通过只说明其覆盖的软件行为；零API容器控制通过只说明相应环境行为。它们不证明真实模型质量、整体研究功效或跨任务学习效果。

## 数据与秘密的存放边界

镜像、数据集、任务输入、私有评分材料、账本、检查点、审计输出和模型响应放在独立运行存储，不纳入本仓库。密钥通过执行环境的秘密机制注入并在使用后清空；不得写入配置示例、命令行参数、日志或git。

预算的累计性独立于代码仓库。若无法取得可信期初占用、来源或评分资料，应记录阻断状态并停留在零实验API准备阶段，不用空账本、虚构控制或占位题单开启正式采集。
