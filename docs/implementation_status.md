# 初步实现状态

日期：2026-10-30。2026-11-01 增加了 round-2 批号 `r2`（见 [protocol_freeze_r2.md](protocol_freeze_r2.md) 与 README 的运行说明）。r2 不与 r1 结果合并，也不改 r1 冻结哈希。r2 账本的冻结期初是 0：本轮每模型重新获得 50 CNY。`ALO_PRIOR_LEDGER` 不设置就不带入旧账。Pro 的 48 CNY 软停止只看这笔 r2 花费。

2026-11-13 增加了独立实验 `fox-single-v1`（见 [fox_single_v1.md](fox_single_v1.md)）。它是 Foxrouter 上的单模型 SWE-bench Lite 基线，输出目录与 r1／r2／r3 分开，不进入 `iter_r2_units`，不改冻结 `g2`，也不改 `r2_freeze_sha256`。六个模型使用 `fox/` 内部名，请求体里的上游 id 只进运行元数据。DeepSeek 在报告里仍写作 `deepseek-flash`。密钥只读环境变量 `FOXROUTER_API_KEY`，不写入清单或账本。这六个模型没有 CNY／USD 单价；缺价格不阻断。`--budget-cap` 默认关闭，只在 API 原样返回的 `cost` 上可选封顶。`fox/mimo-v2.6-flash` 的上下文上限默认 60000，且必须低于 70000；超限默认记 `context_limit` 并且不发送。`--split` 默认 `dev`。第一轮是 dev 23 题 × 6 模型 = 138 个单元；test 300 题推迟，仍可用 `--split test` 选择。报告和 `estimate --from-run-dir` 能按已观测的逐题 token 与原始 cost／credit（均值 ± 样本标准差）外推 test-300，并标明这是估计而不是 test 实测。运行机上的官方镜像 `swebench/sweb.eval.x86_64.*` 只读出 `/testbed`；该路径不 `docker rmi`、不 `docker image rm`、也不做任何 `prune`。离线测试不发请求。带本机镜像、Docker 29、uv 下的 Python 3.12 和密钥的机器上再跑 `--live`。

2026-11-14：fox-single 的 `--read-timeout`（默认 600 秒）是单次尝试的墙钟截止，含 HTTPS 代理的 `CONNECT`。网关或代理在套接字上持续送出少量字节时，这次尝试仍会在截止时刻中止，按已有的 `timeout` 失败种类进入原来的重试、退避和账本（含 usage 字段）。数据集、turn cap、工具、提示、评分、模型 id、记录字段和冻结的 g2／r2 哈希不变。

2026-11-14：`fox-alo-learn-v1` 把 ALO 多模型学习流接到 Foxrouter 池（见 [fox_alo_learn.md](fox_alo_learn.md)）。`--pool fox` 使用 fox-single 的客户端、别名规则、墙钟截止和原样 cost／credit／quota。池成员可由 `--models` 或 `--models-file` 选子集。国内池和 CNY 账本不走这条命令；`python -m alo_cloud.run --arm g2-learn` 会指向它，不进入 `iter_r2_units`，也不改 `r2_freeze_sha256`。六条链是 persist／reset／resources-only × forward／reverse，各用独立 run dir、run id、workdir 和容器名。链内按顺序跑，`--turn-cap` 默认 24。已完成的 `unit_id` 会跳过。persist 链结束后 `export` 对最终 profile 调用 `export_transfer_package`。离线测试不发请求。

2026-11-15：`fox-pb-v1` 的 G0 管线与官方 `895f659` 对齐。每条沙箱命令使用独立容器名，超时后 `docker rm -f`；容器名冲突不再记成 daemon 不可用。提交扫描允许把 `build/...` 安装到 `./executable`，并允许 `.pb-probe/` 里只通过正常接口运行参考二进制的探测脚本；提交物里嵌入、复制、包装、执行或读取参考字节仍然拒绝。`compile.sh` 必须在仓库根产出 `./executable`。公开信号在编译前把镜像里的参考文件移开（不读字节），再用官方同样的 `./executable` 做差分。只读审查在还原工作区前放开镜像文件的写权限，但保持参考二进制为 mode 111。这些改动不进入 `r2_freeze_sha256`。

2026-11-14：`fox-pb-v1` 是 ProgramBench 迁移臂（见 [fox_pb.md](fox_pb.md)）。池是 fox 池里的 `deepseek-flash`、`kimi-k3.1-pro`、`mimo-v2.6-flash`、`glm-5.3-flash`，不收 qwen，也不收任何带 `omni` 的别名。A1 只读导入 SWE `fox_learn export` 的冻结 profile，A2 从共同先验冷启动，A3 只带资源观测，A4 是单节点 `deepseek-flash`，A5 用同一模型担任 p1 和 p2。默认全冻结：不跨题更新，题内观测不改变指派；`--online-update` 才打开并写入清单。公开信号默认只看 `compile.sh`；`--differential-signal` 才加上对 `./executable` 的黑盒 I/O 对照。参考二进制保持 mode 111，角色契约和命令过滤拒绝对其反汇编、插桩、读字节、复制和联网。提交扫描拒绝包装、调用或嵌入该参考文件。这些开关记在运行清单的 `pb_freeze` 里，不改 `r2_freeze_sha256`。离线测试不发请求，也不跑 Docker。

本仓库承载 M1／M2 的实验计划和初步代码，**不构成正式就绪包或已完成实验的证据**。研究定义见[实验计划](experiment_plan.md)与[实验协议](protocol.md)。本文件不附模型输出、运行成绩或历史审计材料。

用户指定的[行政输入目录](../study_records/README.md)已提供任务预选、有序候补、历史替换、暴露登记、M2待冻结顺序和私有存储定位／哈希。16份预算证据原件另已补齐到私有Drive并回读验证，代码仓库只记录其定位与校验信息；具体云任务的物化、期初预算绑定、完整正式realization和平台接受仍须分别完成。

## 初步实现的适用范围

代码用于继续开发三组接合、原生会话及检查点、资源计量、反馈和M2状态迁移。具体能力须以本仓库源码和重新运行的验证为准；目录或接口存在不等于对应研究条件已验收。首次云环境设置只验证轻量依赖、模块导入和离线软件测试。原生容器、基准数据和独立评分的接入属于后续阶段。

Linux执行需明确选择受支持的本地Docker后端，核对原生amd64、可用内存与磁盘、容器限额、权限和评分依赖。子进程只接收允许的环境变量；模型凭据不得进入候选容器、镜像层、源代码或测试工件。

当前能力信号的已知边界是可归因 Python 源码语法：B／TMS 的 pass／fail 不代表功能正确性。全局公开质量与资源观测独立建模。M2实验的结论必须与实际继承的信号一致。

r2 冻结路径仍使用该语法判据，不因后续代码改变已经采集的 r2 结果含义。可选 `criterion_id="repro-v1"` 才使用复现脚本和冻结公开测试。语法单独通过在这条判据里保持 unknown。B／TMS 仍只在明确 pass 或 fail 时更新。`repro-v1` 的 pass 只在脚本修改前失败且修改后通过，或同一冻结测试集在可比的功能退出上从失败变为通过时成立。只有 `tests == 0`、测试被削弱、或依赖／环境失败，都是 unknown。unknown 不写 B／TMS，在 `repro-v1` 下也不是 DAG 节点成功。协议完成和功能 pass 分开记录。

`g1-deepseek-flash`、`g1-glm`、`g2-fixed`、`g2-random` 共用锁定的 `fair-control-v1`：同一 `repro-v1`、同一单位请求帽上的 worker 请求会计、`max_attempts=1`、规划请求 0、验证节点模型请求 0。拓扑和路由是这些臂之间仍可分开的差。`g1-glm` 固定 `glm-5.3-flash`，只在对照集合里，不在 `iter_r2_units`。`g2-fixed` 的固定模型是参数，默认仍是 `deepseek-flash`。轻量 s2 是零请求的公开复测，不是独立审查。审查角色的 `command` 落在审查前的只读快照上。合并挂载（ProgramBench 保留镜像内可执行文件）的审查会先把快照填进 `/workspace`，再把这棵树重新挂成只读后才运行命令；重新挂载失败则拒绝该命令，并且不把结果拷回快照。这些对照不在冻结日程里，也不改冻结 `g2` 的语法判据。路由在已有质量观测后按 `(model, topics)` 读取记到该角色名下的质量，不用整棵工作区的全局质量覆盖任务 cell。

`g2-learn` 与 `g2-resources` 是可选学习臂，不进入冻结 r2 日程，也不改 `r2_freeze_payload` 或 `r2_freeze_sha256`。冻结 `g2` 仍由 `run.py` 里原来的构造传入 `user_preference`、规划器、语法判据和 `max_attempts=2`，资源预测按模型汇总。`g2-learn` 直接使用 `ConstrainedThompsonPolicy`，不传偏好，也不按偏好升级到强模型，资源预测按节点 topics 过滤。`g2-resources` 传入 `learn_quality=False`，可靠的 pass/fail 也不写入 B／TMS，质量后验留在共同先验。`export_learning_state(..., strip_quality=True)` 在导出器内去掉 `credited_quality`、`quality_after` 等质量字段，并去掉 belief 与 TMS 行，只保留资源行。跨轨包 `export_transfer_package` 的键是模型 × 角色主题 × 任务类型，只保留两边共有的 `coding` / `implementation` / `testing` / `validation`，可选预先登记的 `task_family`（`swe-repair` 或 `program-reconstruction`）。`extract_features` 的 cell 标成 `non_transferable`，不进入包内；默认抽取和默认 M2 导出不变。`load_frozen_transfer` 把 SWE 侧的包装进 probe 临时库并核对源文件摘要，写回源文件会抛错。ProgramBench 目标状态不能成为 SWE 单元的父。`iter_r2_units` 不加 ProgramBench 单元。冻结 g2 这一轮不含 ProgramBench 的 p1/p2 契约；该契约在 fox-pb-v1，不改冻结哈希。以上都没有付费运行。未声明工具契约的角色（含分解失败时的 generalist）保持 #34 之前的全套公开工具；显式限制的角色仍执行契约。

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
