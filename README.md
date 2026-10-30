# ALO Cloud Experiments

供 GPT／Codex 云端继续开发的干净起点：**M1 单题自身反馈＋M2 跨任务组织学习**。
仓库包含研究计划、初步算法与状态代码、源码测试和轻量环境准备入口。
完整题集执行、provider 接入和独立评分仍需实现与验收，当前不启动模型实验。

## 从哪里开始

1. 阅读[实验计划](docs/experiment_plan.md)：三类系统、题集、样本与阶段。
2. 阅读[实验协议](docs/protocol.md)：自身续答、跨题统计、probe、资源与停止规则。
3. 阅读[实现状态](docs/implementation_status.md)：已有代码的用途和剩余接合工作。
4. 按[云环境准备说明](docs/cloud_environment.md)安装和验证；首次设置可使用
   [云端准备任务](docs/cloud_setup_prompt.md)。

需要配置实验资源时，使用[模型与题集版本清单](docs/models_and_benchmarks.md)：
其中分别列出四个模型的 API 身份、SWE 数据 revision、ProgramBench 源码与测试资产 revision，
并明确实际接口、逐题镜像和评分依赖仍待冻结。

## 内容结构

```text
AGENTS.md                 云端开发约定
docs/                     计划、协议、实现边界与环境说明
src/                      ALO 核心及实验初步代码
tests/                    离线源码测试
scripts/                  环境准备与源码检查
pyproject.toml            Python 包与可选依赖声明
requirements-cloud.txt    轻量开发依赖
```

本仓库采用新的 Git 历史。只提交文本计划、源码、依赖声明和源码测试。
镜像、镜像归档、题集、私评材料、模型响应、运行日志、审计结果、预算账本、
检查点、模型权重、凭据和本地环境均在仓库之外管理。

## GPT／Codex 云端入口

选择本仓库建立新的云环境，按环境说明准备 Python 3.12，运行安装脚本和
离线检查。配置所需的包源访问，完成设置测试后发布环境，再从发布版本开始
开发任务。本轮设置无需模型 API keys、Docker 服务或题集下载。

当前 Codex 云环境的创建、Install script 与发布方式见
[OpenAI 官方 Cloud environments 文档](https://learn.chatgpt.com/docs/environments/cloud-environments)。
环境准备与发布、软件测试、原生评分接受、正式冻结和真实采集分别判定。

## 继续实现的原则

G1 固定单模型、G2 ALO、G3 原生 GroupChat 的定义保持独立；G2／G3 使用
同一异构池。M1 从自身 B0 完整状态续答 B1，题间重置；M2 只迁移白名单统计，
reset 与独立 probe 使用明确的共同先验／终态副本。当前局部 B／TMS 语法信号
不能直接解释为功能能力。

每模型 50 CNY、合计 200 CNY 的累计预算约束继续有效。新仓库不附历史账本，
也不意味着期初占用为零；真实调用前从独立私有运行存储核对累计使用和未决预留。

首次开发任务建议只完成环境和离线软件验证，然后按实现状态逐项接入。
未实现的运行接口应明确停止，不能用示例输出或脚本 fixture 冒充模型结果。
