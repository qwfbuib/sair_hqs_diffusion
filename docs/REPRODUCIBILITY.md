# 复现、断点续训与Git管理

## 源码与检查点身份

原入口使用`native128-hqs-training-v1`，可选骨干入口使用`native128-selectable-prior-v1`。新检查点记录结构版本、扫描后端、模型与训练配置、数据核心文件哈希、先验权重哈希、运行源码哈希及随机状态。

续训要求身份相同。先验冻结后，展开阶段使用自身保存的`unfold/input_prior.pt`，避免外部`best.pt`后续变化影响该运行。

更换骨干、网络结构、训练预算、源码或数据内容后，不应通过改写检查点的身份字段继续旧任务。请使用新输出目录。新入口不自动将旧格式优化器迁移为新格式续训。

## Git换行与原始字节

本仓库`.gitattributes`使用`* -text`，保留文件原始字节，避免Windows/Linux自动换行转换改变源码SHA256。它不关闭文本Diff。

已有`PACKAGE_MANIFEST.json`是初始服务器包的历史清单。后续新增代码的运行身份由训练器和每个运行的`source/`快照记录，不应把历史清单当作当前完整仓库清单。

Git提交标识一个代码版本，但不会消除系统、PyTorch、CUDA、GPU或路径分隔符差异。复制到另一系统后，不能承诺旧检查点直接续训或逐位重现。新运行应记录所用提交和环境，并保留整个运行目录。

## 数据路径迁移

可选骨干入口先只读检查：

```bash
python -B diagnose_selectable_resume.py --out runs/my_run --phase unfold
```

仅当数据核心哈希、配置、源码、骨干、先验及其他身份字段全部匹配，**唯一差异为数据根路径**时，才使用数据路径迁移工具。先停止该运行的训练进程，确认`run_config.json`指向真实的新路径，再预览：

```bash
python -B relocate_selectable_resume.py --out runs/my_run --phase unfold
# 核对预览与实际迁移事实后显式执行
python -B relocate_selectable_resume.py --out runs/my_run --phase unfold --apply
```

工具创建`last.before_data_relocation.pt`备份及迁移报告，拒绝覆盖已有备份，拒绝将其他身份差异伪装成路径变化。这不是跨格式、跨骨干或跨源码版本的迁移工具。旧格式先验的服务器导入另见`README_SERVER.md`。

## 随机性与恢复

可选入口使用与初始化分离的私有加噪生成器，保存其状态以及全局随机状态、优化器、AMP scaler和样本游标。AMP跳过更新时重放同一输入与噪声。

原入口也保存优化器、scaler、步数、随机状态和游标，但没有新增入口的独立加噪流。不能把旧U-Net历史训练和新增入口的Mamba训练称作使用相同噪声抽样的消融。

`last.pt`代表最近保存的进度，`best.pt`代表验证指标最优的保存点。续训不会恢复未写入检查点的更新。

## 代码与实验资产分开管理

Git只追踪源码、配置、测试和文档。`.gitignore`排除`runs/`、`reports/`、数据、权重、临时文件、环境及日志。现有文件留在本地，并不会因忽略规则而删除。

每次实验独立保留配置、数据身份、源码快照、环境、指标、权重和失败状态。若需要分享实验资产，另行决定数据与权重的授权和存储方式，不要直接强制加入Git。

CPU单元测试不读取真实数据、不验证GPU性能。仓库的基础CI同样不能替代原生128资源检查、匹配训练预算的重建对照或独立测试集评价。

## 已知部署限制

- 训练、自动评价和测试应从仓库根目录启动，由`common.py`加载本地`vendor/`，不需要旧项目绝对路径。
- 手动导入`selectable_prior`前先`import common`，完成本地vendor路径初始化。测试按README列出的四个模块运行，避免通用discover在初始化前导入模型包。
- 根目录部分历史绘图脚本依赖`C:/Windows/Fonts/msyh.ttc`。Linux运行绘图前需要提供合适的中文字体并适配字体路径；JSON指标评价不受此限制。
- `check_selectable_prior.py`需要数据及旧基线检查点来进行真实兼容性／资源检查。这些资产不在Git仓库中。
- README_SERVER、INFERENCE_BASELINE及工程验收文档保留历史实验路径。相应运行资产没有随代码发布，不应将文档中的历史状态当作当前机器上的运行结果。
