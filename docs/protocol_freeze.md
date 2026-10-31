# 付费运行冻结

日期：2026-10-31。本文件列出付费日程的任务身份、种子、组别、模型和参数。规范载荷在 `alo_cloud.freeze.freeze_payload`。CI 重新计算其 SHA-256，必须与下面这一行一致。

freeze_sha256: 9d9ea4e2bb5756a730e01c70d2b445e45cebcd8279d959a344fa4ca690387852

上一版哈希是 `9ee9ccf36dc80f535f6f42a7821bc694d5aa553147c32d3445d2bc7a10eee42d`。题单、种子、组别、模型、标价、超时和计费都没有改。这次改哈希是因为：`reasoning_tokens` 超出 `completion_tokens` 不超过 64 或不大于 completion 的 1% 时钳到 completion，按报告用量结算；能结算的工具异常交回代理，沙箱状态不明只重启该单元一次；只有全局 CNY 上限是硬停止；基础设施截断用 `requeue-interrupted` 重新入队且不增加 `unit_restarts`。

检查：`python -m alo_cloud.freeze --check`

## 模型与参数

| 角色 | 请求 `model` |
| --- | --- |
| 池顺序 | `deepseek-flash`、`mimo-v2.6-flash`、`glm-5.3-flash`、`mimo-v2.6-pro` |
| G1 Main | `mimo-v2.6-flash` |
| G1 Strong | `mimo-v2.6-pro` |
| G2 planner | `deepseek-flash` |
| G3 manager | `deepseek-flash` |
| G3 Planner / Engineer / Critic / Verifier | `deepseek-flash` / `mimo-v2.6-flash` / `glm-5.3-flash` / `mimo-v2.6-pro` |

发送的参数：四个模型都 `stream=true`。DeepSeek 另送 `stream_options.include_usage=true`。MiMo 只送 `max_completion_tokens`，GLM-5.3-Flash 送 `thinking.type=enabled`。调用方要求至少 4096 时，四个模型都实际发送 16384。不送 temperature、`reasoning_effort`、`clear_thinking`。供应商读超时：MiMo-V2.6-Flash 与 MiMo-Pro 600 秒，DeepSeek 与 GLM 300 秒，与沙箱单命令 60 秒分开。已计费超时不计入未知上限。某模型最近 40 次请求里已计费超时超过 25%，或本代已计费超时合计超过 10 CNY，才停止日程。价格按 2026-10-31 官方标价，`deepseek-flash` 用高峰价。计费向上取整到 microcny；没有缓存命中拆分时，输入全部按缓存未命中。

预算：每模型 50 CNY，合计 200 CNY。新建账本期初承诺为 0。

G2 基础种子 `20261027`。选题种子 `formal-20261028-preregistered-task-selection-v1`。

操作请求上限（不超过协议 32/16）：M1 的 G1、G2、G3 都是 B0 12 次、B1 8 次。M2 学习每次 8 次请求，固定参考 8 次，probe 为 6。

## 任务

开发题不在付费日程里。正式题与行政预选相同，两轨各 6 道 M1、4 道学习、2 道 probe。

SWE-bench Lite M1：`scikit-learn__scikit-learn-13584`，`django__django-11583`，`sympy__sympy-11870`，`django__django-16408`，`sympy__sympy-20322`，`sympy__sympy-18835`。

SWE-bench Lite 学习：`sympy__sympy-18621`，`django__django-15851`，`scikit-learn__scikit-learn-11040`，`django__django-11019`。

SWE-bench Lite probe：`django__django-13590`，`sympy__sympy-21055`。

ProgramBench M1：`kisielk__errcheck.dacab89`，`mgdm__htmlq.6e31bc8`，`esubaalew__run.0fb9dec`，`raviqqe__muffet.a882908`，`ariga__atlas.6d81150`，`unhappychoice__gittype.34b72d0`。

ProgramBench 学习：`arthursonzogni__json-tui.17a22b6`，`xorg62__tty-clock.f2f847c`，`o2sh__onefetch.e5958ce`，`johnkerl__miller.8d85b46`。

ProgramBench probe：`noborus__trdsql.d8c5ff6`，`ffmpeg__ffmpeg.360a402`。

## M2 顺序与种子

SWE order-1，首臂 persist，学习种子 `2026102800`–`2026102803`，probe 种子 `2026102900`、`2026102901`。

SWE order-2，首臂 reset，顺序为学习列表的 `[2,0,3,1]`，种子 `2026102810`–`2026102813`，probe 种子 `2026102910`、`2026102911`。

ProgramBench order-1，首臂 persist，种子 `2026102900`–`2026102903`，probe 种子 `2026103000`、`2026103001`。

ProgramBench order-2，首臂 reset，顺序 `[2,0,3,1]`，种子 `2026102910`–`2026102913`，probe 种子 `2026103010`、`2026103011`。

同一序列内按位置交替 persist 与 reset。全部学习结束之后，G1 Main、G1 Strong 和 G3 对每道学习题和 probe 各跑一次。终态 probe 的 learned / cold 先后由序列序号加 probe 位置的奇偶决定，冷启动种子是上表 probe 种子加 17。

## 评分钉扎

SWE-bench Lite revision `6ec7bb89b9342f664a54a6e0a6ea6501d3437cc2`，评分包 `swebench==4.1.0`。

ProgramBench 源码 commit `895f65916bc5ba34a915d36da321fde0301125ad`。测试资产 revision `de0ddfb637590c7ecb54fa0b5301f6dc7dfbcee5`。
