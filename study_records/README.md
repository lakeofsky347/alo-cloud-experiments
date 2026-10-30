# 本轮任务、暴露与历史存储记录

整理日期：2026-10-30，Asia/Singapore。以下行政元数据由用户明确要求加入新私有仓库，单独存放在 `2026-10-28-m1-m2/`，保留其拥有来源的位置与 SHA-256。

**已经提供可查找的候选清单和历史记录；完整正式任务 realization 仍未冻结。** 原始预选、旧平台 QC 候选替换、原生平台停止时状态不能合并成一份“已经正式接受的最新30题”。没有通过此次上传启动实验或解除其他研究门禁。

## 文件与读取顺序

| 文件 | 内容 | 使用边界 |
| --- | --- | --- |
| [task_catalog.json](2026-10-28-m1-m2/task_catalog.json) | 原始30个任务身份、用途、位置、分层、版本、选择seed、4个既有排除身份；完整有序候补 | `preselected_not_formally_realized`；候补保持原组别及列表顺序 |
| [selection_history.json](2026-10-28-m1-m2/selection_history.json) | 旧v2平台7次开发候选替换，以及后来原生平台回到原始清单后停止的行政状态 | 旧 `ready` 不是新平台接受；正式24题未在旧替换链中改动 |
| [m2_candidate_orders.json](2026-10-28-m1-m2/m2_candidate_orders.json) | 两轨各两条派生学习顺序、首臂、源码规定的seed与probe身份 | 源码派生的待冻结计划，不冒充已经存在的M2冻结manifest |
| [exposure_register.json](2026-10-28-m1-m2/exposure_register.json) | 已知真实暴露、保守排除、历史开发预留、485个唯一ID的本地记录筛查状态、仓库族重叠 | 原调查截止于10月28日，`not_found_in_scoped_records`不等于全球未暴露 |
| [storage_index.json](2026-10-28-m1-m2/storage_index.json) | 41个本机位置、8个历史Drive定位，以及新补传的16份预算原件私有Drive交付 | 原件不在代码仓库；新交付已回读校验，8个历史交付定位与其分别记录 |
| [source_index.json](2026-10-28-m1-m2/source_index.json) | 投影的源记录位置／SHA、逻辑来源ID和本目录导出文件校验表 | 追溯行政来源；不内嵌源记录全文或私有资产 |

## 任务清单的准确含义

每轨原始候选是3道Strong开发、6道M1、4道M2学习、2道独立probe；两轨共30个新任务身份，其中正式24个。`task_catalog.json` 的 `groups` 数组保留原始位置。候补共有1506个带用途／分层位置的条目，去重后473个身份；同一候补可以出现在不同用途列表，不能先全局去重或排序再执行替换。

旧v2平台曾对开发候选进行7次替换，涉及5个开发位置。后来原生平台的监督状态仍使用原始30题、没有候补替换，只保留code-minimap的一项历史种子接受，随后停止。`selection_history.json` 并列保存这些事实，不让较新的文件名替代平台适用范围。

两条M2候选顺序来自现有生成源码：`[L1,L2,L3,L4]` 与 `[L3,L1,L4,L2]`，首臂分别为persist和reset；第二条不是第一条的倒序。正式冻结时仍需绑定实际realization、平台、模型条件和所有输入。

SWE的`short/medium/long`是历史行政标签，含义为评价侧参考patch的改动规模桶，不是已经验证的任务跨度。分层和来源元数据只供研究控制与选题，不能把参考补丁或隐藏评价信息注入候选模型上下文。

## 暴露记录的准确含义

已知真实模型任务暴露包括 `sqlfluff__sqlfluff-2419` 和 `chmln__sd.87d1ba5`，继续排除于新Strong、M1、M2和probe。`sqlfluff__sqlfluff-1625`是保守QC排除，不能写成已证明的真实任务暴露；其他排除原因按登记中的证据范围处理。

13个历史开发预留身份来自原开发题和旧替换链，不等于13题都已经调用过模型，但也不能未经重新审查转入同批正式或probe用途。原候选30、候补473、首轮替换3的集合互有重叠，合并筛查为485个唯一ID，不能相加为506。

现有调查是本地留存记录范围内的历史筛查，不覆盖模型预训练、项目外使用、删除或未记录的提示。此次整理核对行政源文件与哈希，没有重新阅读所有底层ledger／checkpoint内容。SWE的M1、M2学习和probe有明显仓库族重叠：任务身份holdout不能称为新仓库外泛化。

## 历史私有存储如何定位

**预算原件已补齐到[可访问的私有Drive目录](https://drive.google.com/drive/folders/1tnfwZz6rlGjnYahX5Vac0r9N1s5oqxGk)。** r1至r5各有`manifest.json`、`provider-receipt.json`、`requests.jsonl`，根目录另有`budget-snapshot-pre-strong-v2.json`，共16份原件、902677 bytes。所有原件均已从Drive下载回读，SHA-256与本机原件逐一一致。目录与原件当前仅有owner权限，没有公开或额外分享。

使用[FILES.json](https://drive.google.com/file/d/1NFMfhltgC9nt5_l73ZyM17CGEtfkfPw8/view?usp=drivesdk)取得逐文件ID、普通私有链接、大小及SHA；[SHA256SUMS](https://drive.google.com/file/d/1OFTJMVtaV9NYCQbVZEWsrlsTGz_na0Z2/view?usp=drivesdk)只覆盖这16份原件。这两份文件及Drive中的README为新生成的索引／说明，不冒充历史原件。

用户已明确选择完整账本原件：5份`requests.jsonl`保留原有`text`／`raw_content`模型回答字段，不能删字段后仍称原件。它们仅存于已核验私有Drive目录，代码仓库只记录位置和校验元数据。预算核对不得把历史模型回答回灌新候选任务。

`storage_index.json` 的 `verified_private_budget_originals` 是这次实际交付；对应16个本机条目的 `verified_private_drive_copy` 指向逐字节相同副本。`entries` 仍保留本机绝对路径、角色和来源哈希，r1至r5父账本关系没有改写。当前Drive连接可读取原件不等于它们已经进入某个GPT云执行任务的文件系统；控制器使用前仍需物化、验证并绑定期初预算。unknown预留和祖先使用继续保留，不将新仓库空账本当成零使用。

`historical_remote_locators` 只提供8个旧交付普通folder ID与canonical URL，不包含预签名链接、上传会话或token；其历史标记与新预算交付分开。不要从旧`01-source.zip`或旧交付目录推断预算账本可用，使用上述已验证的新目录和逐文件索引。

旧服务器的停止／清理状态引用10月28日最后记录，本次没有重新连接。旧授权已经撤回，旧控制只保留为历史证据，不能从本索引恢复远端实验。目录哈希为null，单个索引文件的哈希不代表整个目录已做归档完整性验证。

## 上传范围

本目录只包含行政投影和定位／哈希。没有任务正文、参考答案、gold patch、隐藏测试、测试blob、镜像、模型回答、原始请求账本、候选检查点或密钥。原始源文件保留原字节与位置，投影不替换它们。`source_index.json` 的哈希表可核查导出文件是否变化，不证明原始资料已迁移或模型未受任何外部暴露。

下一步可以使用这些记录定位资料、核对候选与暴露范围；正式运行仍须完成目标平台接受、期初预算私有材料供给、Strong选择、实际题单／评分／反馈／模型合同与M2隔离门禁。
