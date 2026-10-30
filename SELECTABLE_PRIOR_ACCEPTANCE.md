# 可选U-Net／Mamba先验：工程验收

状态：已实现并通过有限工程验证。没有启动100000步先验或20000步展开的全量训练，原训练源码和旧权重/结果未修改。

## 已交付

- 独立selectable_prior包：原U-Net兼容分支、两个低分辨率SS2D替换的混合Mamba分支、统一VE先验与HQS接口。
- 并行训练入口train_selectable_prior.py、自动骨干选择评价入口evaluate_selectable_prior.py、原batch资源检查入口check_selectable_prior.py。
- 两份配置configs/server_128_unet.json与configs/server_128_mamba.json，仅prior.backbone不同；同一原生128数据、划分、标定、损失和更新预算。
- 文档SELECTABLE_PRIOR.md；独立扫描参考、许可证及来源哈希均随代码和运行源码快照保留。

## 验证证据

- 9项单元测试通过：扫描输出/全部输入梯度/分块状态，冻结eval时重计算及输入和噪声条件反传，A/B相同加噪输入，旧权重加载，错误骨干/阶段/数据/后端/缺失权重拒绝，逐位续训与跳过更新后的同输入重放，以及不存在续训目录时拒绝创建。
- 两种骨干、两个阶段的断点恢复，在小型CPU问题上逐位核对模型、优化器、scaler、全部随机状态、样本游标、步数和best；不是仅比较loss。
- 真实128网格：两种骨干均完成先验2次更新和完整8阶段展开2次更新；冻结先验后四组阶段参数有有限梯度。每次展开验证覆盖7场景×四孔径=28观测，CG均通过1e-5门槛。
- 正式先验batch4的128网格资源检查通过；展开batch1、累计4个真实样本的更新通过，32个阶段参数可训练。
- 旧U-Net真实先验的四个sigma、完整展开的四孔径输出，新旧接口逐元素差均为0。
- 原训练源码哈希与第7000步导入检查点记录全部一致；数据只读，不使用test。

## 本机资源试测

环境：NVIDIA GeForce RTX 2070 SUPER，torch 2.1.2+cu118，CUDA 11.8。
这些是2次先验更新和1次展开累计更新的工程样本，包含冷启动差异，不能当作稳定性能基准。

| 骨干 | 先验参数量 | 先验batch4峰值 MiB | 两次先验更新秒数 | 展开累计4样本峰值 MiB | 展开一次更新秒数 |
|---|---:|---:|---|---:|---:|
| unet | 4,461,409 | 526.5 | 0.270, 0.045 | 1178.4 | 5.222 |
| mamba | 4,241,505 | 3501.6 | 0.574, 0.517 | 1885.5 | 9.422 |

原生Mamba扫描在该有限试测中更慢、显存更高；未测官方fused后端。两种骨干参数量不同，不能将未来质量差异直接当成严格等容量机制结论。

## 最终运行与文件

- reports/selectable_prior_tests_v2.txt：9项测试完整日志。
- reports/selectable_prior_resources_v1/report.json：旧权重兼容、阶段梯度与正式batch资源数值。
- runs/selectable_unet_smoke_v3 与 runs/selectable_mamba_smoke_v3：最终源码两阶段各2步检查点、优化器、随机状态、配置和源码快照。
- runs/selectable_mamba_eval_smoke_v2：新Mamba完整展开检查点自动选择与28观测评价。
- runs/selectable_mamba_prior_eval_smoke_v1：新Mamba先验自动选择与四sigma去噪诊断。
- runs/selectable_legacy_eval_smoke_v1：旧完整U-Net检查点自动识别与28观测评价。
- v1/v2开发期短程训练目录保留其当时源码快照；因交付前增加了来源资产与输入保护，最终验收使用v3，不能拿开发期检查点直接续训最终源码。

## 使用边界

先通过配置切换骨干，再分别训练对应先验与展开阶段；不能把U-Net权重当成Mamba权重。旧U-Net权重仅作兼容初始化或评价，旧格式不在新入口直接续训。
本次短程loss、RMSE和外观只用于检查执行链路，不能用于比较去噪/重建质量、高频恢复或真实空间分辨率。正式A/B应在新增入口分别从头训练，保持有效batch、更新数、数据和先验调用预算一致，并独立报告资源成本。
