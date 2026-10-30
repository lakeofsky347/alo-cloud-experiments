# 轻量云开发环境

本仓库先用于实现和验证 M1／M2 的代码骨架及脚本 fixture，不包含基准数据、镜像、历史审计输出或凭据。这里不创建云账号、启动服务、连接 Docker daemon 或请求模型 API。

配置流程依据 2026-10-30 核对的 [OpenAI 官方 Cloud environments 文档](https://learn.chatgpt.com/docs/environments/cloud-environments)。当前流程是在环境设置对话中准备和测试仓库，再发布；不是旧版一次性 setup 阶段或 Cursor Build 配置。

## Python 与依赖

使用 Python 3.12。默认安装 PyYAML、pytest 和本项目，足够运行核心 ALO、状态、预算及协议 fixture。AutoGen 0.2.40 和 Docker Python SDK 7.2.0 位于项目的可选 `native` extra；只有需要原生框架 fixture 时显式安装。Python SDK 的安装和导入不代表 Docker 已运行。

`requirements-cloud.txt` 是少量直接依赖候选版本，不是完整 Linux wheel/hash 锁，也没有复用 Mac 的观察依赖导出。完整解析结果和原生框架导入须在真实云环境中检查。默认不安装 SWE-bench、ProgramBench、datasets、pandas 或 pyarrow；将来接基准评分前，另建并验证那部分依赖，不能由本次轻量通过推断 PyArrow 或私评兼容。

从仓库根目录安装：

```sh
bash scripts/setup_cloud.sh
```

默认 venv 为仓库内被忽略的 `.venv`。可以指定外部目录和 Python，不改全局 Python 或系统包：

```sh
bash scripts/setup_cloud.sh --python python3.12 --venv /tmp/alo-cloud-venv
```

需要原生 GroupChat 框架 fixture 时再执行：

```sh
bash scripts/setup_cloud.sh --with-native
```

已有 venv 只有在包含 `pyvenv.cfg` 且 Python 为 3.12 时才复用；不会删除旧目录。每次安装的轻量检查结果默认写入新的系统临时目录，亦可用 `--evidence-dir <新外部目录>` 指定，任何测试或检查输出均不纳入仓库。

## 在 Codex Cloud 中准备

按官方文档选取仓库，在 setup 对话中让 Codex 确认 Python 3.12、执行安装和软件测试。可将以下命令记入 **Install script**：

```sh
bash scripts/setup_cloud.sh
```

本阶段没有服务需要启动。**Start skill** 可以说明：进入仓库，使用 `.venv/bin/python` 检查软件环境并运行脚本测试，不启动 Docker、不获取镜像、不配置 provider keys：

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/check_cloud.py --require-linux
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests -q -p no:cacheprovider -o addopts= --basetemp="$(mktemp -d /tmp/alo-cloud-pytest.XXXXXXXX)"
```

原生 fixture 需要先在 Install script 中显式加入 `--with-native`，然后在软件检查中也使用 `--with-native`。缺少可选库时，对应原生测试应明确 skip，不能记为已通过。

准备和测试完成后再 **Publish**。之后更新安装或启动约定时，按官方环境的 Edit／测试／Republish 流程处理，并在新任务中验证；现有任务保留各自的工作状态。仓库刷新不会自动重新执行安装与启动命令。[安装与启动、状态更新](https://learn.chatgpt.com/docs/environments/cloud-environments#customize-installation-and-startup)

## 网络、代理和证书

普通依赖安装需要环境允许包管理器及其下载域名。官方 Package managers preset 包含 `pypi.org`、`files.pythonhosted.org`；组织 Agent Security 限制仍同时生效。不要把安装失败直接归因于包缺失，应先区分域名访问、代理、证书与版本解析。[服务访问与包管理域名](https://learn.chatgpt.com/docs/environments/cloud-environments#connect-to-services)

安装脚本继承平台已有代理、CA 和包管理器配置，不清除 `HTTP_PROXY`／`HTTPS_PROXY`，不禁用 TLS 验证、不写系统证书、不改全局 pip 配置。软件导入检查对子进程保留所需代理/CA 配置并加 Python socket 防护；它不发出连通性探测。不要把代理或证书配置值复制到任务说明或结果里。

本阶段无需实验 API key、HTTP secret 或 provider 环境变量。以后若接真实模型，须单独定义适用的凭据入口、调用预算、网络范围和执行授权，不用本安装脚本来验证真实 API。

## 检查结果的含义

`scripts/check_cloud.py` 验证实际 Python、直接库版本及核心模块导入，并返回真实退出码。加 `--with-native` 后才导入 AutoGen 和 Docker SDK，每个模块使用独立子进程。它不会调用 Docker CLI、读取 socket、启动 daemon、下载镜像或请求 provider；也不会把未运行的基准检查标为成功。

输出中的 `cloud_environment_verified=false`、`docker_daemon_checked=false`、`formal_experiment_ready=false` 保持明确边界。本检查通过只代表软件导入通过；完整测试结果由 pytest 另行提供，正式样本、预算、基准环境、模型调用与评分仍待后续实现和验收。
