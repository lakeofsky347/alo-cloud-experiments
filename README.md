# ALO Cloud Experiments

供 GPT／Codex 云端继续开发的干净起点：**M1 单题自身反馈＋M2 跨任务组织学习**。
仓库包含研究计划、初步算法与状态代码、源码测试、轻量环境准备入口，以及用户指定的研究行政元数据。
付费运行控制器、官方标价和协议冻结已写入仓库。真实采集仍要密钥、公开题面和 Docker，当前不在此环境启动。

## 从哪里开始

1. 阅读[实验计划](docs/experiment_plan.md)：三类系统、题集、样本与阶段。
2. 阅读[实验协议](docs/protocol.md)：自身续答、跨题统计、probe、资源与停止规则。
3. 阅读[实现状态](docs/implementation_status.md)：已有代码的用途和剩余接合工作。
4. 按[云环境准备说明](docs/cloud_environment.md)安装和验证；首次设置可使用
   [云端准备任务](docs/cloud_setup_prompt.md)。

需要配置实验资源时，使用[模型与题集版本清单](docs/models_and_benchmarks.md)：
其中分别列出四个模型的 API 身份、SWE 数据 revision、ProgramBench 源码与测试资产 revision，
并明确实际接口、逐题镜像和评分依赖仍待冻结。

本轮任务、暴露记录和历史私有存储位置见 **[study_records/README.md](study_records/README.md)**。
其中提供原始候选30题及有序候补、旧QC与原生平台状态、暴露登记、M2候选顺序和存储定位／哈希；
候选清单不等于完整正式realization，定位索引不等于私有内容已迁移。
其中16份历史预算原件已另行补传到已核验的私有Drive目录，入口和逐文件链接见该说明；原件内容不进入代码仓库。

## 内容结构

```text
AGENTS.md                 云端开发约定
docs/                     计划、协议、实现边界与环境说明
src/                      ALO 核心及实验初步代码
tests/                    离线源码测试
scripts/                  环境准备与源码检查
study_records/            单独存放任务、暴露、历史存储的行政元数据
pyproject.toml            Python 包与可选依赖声明
requirements-cloud.txt    轻量开发依赖
```

本仓库采用新的 Git 历史。提交文本计划、源码、依赖声明、源码测试和明确授权的行政元数据。
镜像、镜像归档、题集、私评材料、模型响应、运行日志、审计结果、预算账本、
检查点、模型权重、凭据和本地环境均在仓库之外管理；单独记录的存储路径不包含这些目标内容。

## GPT／Codex 云端入口

选择本仓库建立新的云环境，按环境说明准备 Python 3.12，运行安装脚本和
离线检查。配置所需的包源访问，完成设置测试后发布环境，再从发布版本开始
开发任务。本轮设置无需模型 API keys、Docker 服务或题集下载。

当前 Codex 云环境的创建、Install script 与发布方式见
[OpenAI 官方 Cloud environments 文档](https://learn.chatgpt.com/docs/environments/cloud-environments)。
环境准备与发布、软件测试、原生评分接受、正式冻结和真实采集分别判定。

## Round 2 runbook

r2 is a new batch. It is not comparable with round 1 and must not be pooled with r1 scores, records, or ledgers. The model request name stays `deepseek-flash`. Endpoints stay the mainland-China pay-as-you-go URLs. The ledger is still cumulative: 50 CNY per model, 200 CNY total. `mimo-v2.6-pro` also soft-halts at 48 CNY between units. The freeze is [docs/protocol_freeze_r2.md](docs/protocol_freeze_r2.md).

Use a fresh run directory. Do not point r2 at an r1 run directory.

```sh
export ALO_RUN_DIR=/workspace/alo-r2-run
export ALO_BATCH=r2
python -m alo_cloud.r2 --check
python scripts/replay_transport.py
```

Pre-pull the instance images for the tasks that were not in the round-1 SWE schedule:

```sh
python -m alo_cloud.r2 --images | xargs -n 1 docker pull
```

Scored collection is SWE-bench only. ProgramBench stays out of this run. Four shards, clearing a stale halt:

```sh
CLEAR_HALT=1 NUM_SHARDS=4 ALO_BENCHMARKS=SWE-bench ALO_BATCH=r2 ALO_RUN_DIR=/workspace/alo-r2-run bash scripts/run_sharded.sh
```

`ALO_BATCH` defaults to `r2` and the default run directory for that batch is `/workspace/alo-r2-run`. Stage public task JSON at `$ALO_RUN_DIR/public_tasks/SWE-bench_Lite/<task_id>.json` before the live command. The optional ProgramBench smoke is not part of the scored schedule:

```sh
python -m alo_cloud.run --pb-smoke --dry-run --run-dir /tmp/alo-r2-pb-smoke
```

That unit is `kisielk__errcheck.dacab89`, `g1-main`, B0, 40 requests.

The r2 ledger is created with round 1's settled spend as its opening commitment, so the 50 CNY, 200 CNY, and 48 CNY Pro checks are cumulative. The frozen carry-in is deepseek 4.619, mimo-flash 3.732, glm 2.524, mimo-pro 16.955 (27.830 CNY). `ALO_PRIOR_LEDGER` can point at the round-1 `budget.sqlite3`; that file is read only, and a higher settled total replaces the frozen map. G1-strong M1 is 12 tasks, including the 6 round-1 tasks. G1-strong M2 references are not scheduled. In each G2 and G3 unit, Pro may make at most `floor(request_cap / 5)` billed calls.

Cost estimate from those caps and the round-1 mean CNY per request (deepseek 0.009, mimo-flash 0.005, glm 0.015, mimo-pro 0.0335), assuming every unit uses its full cap. This is not fit to round-1 scores. Pro's new headroom before the 48 CNY halt is 31.045 CNY. The full-cap Pro total is 26.733 CNY.

| Model | Prior | Expected new | Expected cumulative | Worst-case new | Worst-case cumulative |
| --- | ---: | ---: | ---: | ---: | ---: |
| `deepseek-flash` | 4.619 | 8.022 | 12.641 | 22.482 | 27.101 |
| `mimo-v2.6-flash` | 3.732 | 10.577 | 14.309 | 18.610 | 22.342 |
| `glm-5.3-flash` | 2.524 | 12.050 | 14.574 | 36.150 | 38.674 |
| `mimo-v2.6-pro` | 16.955 | 26.733 | 43.688 | 26.733 | 43.688 |

Expected new spend together is 57.382 CNY (85.212 cumulative). The worst-case column is each model's own ceiling, not one combined bill. Pro's expected and worst case match because the one-in-five cap binds. Cumulative Pro, 43.688 CNY, is under the 48 CNY halt by 4.312 CNY.

## 继续实现的原则

G1 固定单模型、G2 ALO、G3 原生 GroupChat 的定义保持独立；G2／G3 使用
同一异构池。M1 从自身 B0 完整状态续答 B1，题间重置；M2 只迁移白名单统计，
reset 与独立 probe 使用明确的共同先验／终态副本。当前局部 B／TMS 语法信号
不能直接解释为功能能力。

每模型 50 CNY、合计 200 CNY 的累计预算约束继续有效。新仓库不附历史账本，
也不意味着期初占用为零；真实调用前从独立私有运行存储核对累计使用和未决预留。

首次开发任务建议只完成环境和离线软件验证，然后按实现状态逐项接入。
未实现的运行接口应明确停止，不能用示例输出或脚本 fixture 冒充模型结果。
