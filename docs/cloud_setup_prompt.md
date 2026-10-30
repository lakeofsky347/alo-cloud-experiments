# 首次 GPT／Codex 云环境准备任务

将下面的任务交给新仓库对应的云环境设置会话。此任务准备开发环境，
不开展模型实验。

```text
Prepare a reusable development environment for this repository.

1. Read AGENTS.md, README.md, docs/experiment_plan.md,
   docs/implementation_status.md, docs/cloud_environment.md, and
   study_records/README.md. Administrative candidate and exposure records are
   available there; they are not a formally accepted execution manifest.
2. Use Python 3.12. Run bash scripts/setup_cloud.sh and then the exact
   verification commands documented in docs/cloud_environment.md.
3. Use only the declared package dependencies. Preserve the platform's
   network proxy and certificate configuration. Do not change system security
   or silently substitute package versions after an installation failure.
4. Do not configure provider credentials, call model APIs, download benchmark
   tasks or hidden graders, pull or build container images, or start Docker.
5. Keep generated logs, reports, caches, and test outputs outside tracked
   source files. Run python3 scripts/check_repository.py and inspect the diff.
6. Report which checks actually ran, their exit status, Python/package
   versions, and any missing integration. Software checks are not benchmark
   scores, native Docker acceptance, or formal experiment readiness.
7. Record successful dependency installation in the environment's Install
   script. No service Start skill is required for the initial offline code.
   Finish and test the setup before publishing the reusable environment.
```

环境安装、环境发布、正式研究冻结与真实采集是不同的状态。本任务不创建
Strong 选型或 M1／M2 的执行队列。
