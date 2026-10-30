# SAIR HQS Diffusion

使用物理数据一致性、HQS展开网络和可选U-Net／Mamba扩散先验，从复数可见度重建SAIR亮温图。

本仓库保存训练、推断、诊断和测试代码。**数据集、预训练权重及实验输出不包含在仓库中**，需要自行准备。当前代码采用原生128×128合成数据与VE去噪先验，不是原始DDNM采样器，也不代表已验证的真实仪器恢复系统。

## 功能与结构

- 原生128网格的复数前向观测与实值图像伴随，使用已保存的物理观测，不在展开训练时重新模拟可见度。
- 两阶段训练：先训练噪声预测先验，再冻结先验，学习8阶段HQS的32个阶段参数。
- 配置选择`unet`或`mamba`。Mamba分支是保留U-Net编码器、解码器及跳跃连接的混合骨干，只替换32×32与16×16特征上的两个模块。
- 四方向SS2D使用原生PyTorch扫描，不要求`mamba_ssm`、Triton或编译扩展。
- 支持软过滤、同权重关闭先验／过滤的诊断、后期减弱过滤实验，以及严格检查身份的断点续训。

```text
sair_hqs_diffusion/
  train_selectable_prior.py       可选骨干的两阶段训练入口
  evaluate_selectable_prior.py    按检查点身份自动构建骨干
  selectable_prior/              统一先验、混合Mamba、检查点与训练实现
  train_native_hqs.py             保留的原U-Net两阶段训练入口
  evaluate_unfold.py              旧格式完整展开网络与模块关闭诊断
  infer_fixed_prior.py            冻结先验、固定阶段参数的机制对照
  diagnose_selectable_resume.py   只读检查续训身份差异
  relocate_selectable_resume.py   仅数据路径变化时的显式迁移工具
  vendor/                        项目所需的本地模块快照
  configs/                       原入口、U-Net与Mamba配置
  docs/                          数据、复现和Git使用说明
  test_*.py                      无外部数据的CPU单元测试
```

## 环境安装

建议使用Python 3.10，在仓库根目录运行命令。先安装适合GPU与驱动的PyTorch，基础依赖不会自动替换已有PyTorch。

```bash
git clone https://github.com/qwfbuib/sair_hqs_diffusion.git
cd sair_hqs_diffusion
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell改用：.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# 可选：复现本地验证使用的CUDA 11.8构建，须先确认GPU/驱动兼容
python -m pip install -r requirements-torch-cu118.txt
python -m pip install -r requirements.txt

# 需要绘图及vendor中的额外分析模块时安装
python -m pip install -r requirements-analysis.txt
```

本地验证环境为PyTorch 2.1.2+cu118、NumPy 1.26.4、Pillow 12.3.0。安装依赖不等于目标GPU验证通过。仅运行CPU测试可单独安装：

```bash
python -m pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -B -m unittest test_fixed_prior test_filter_fade test_evaluate_unfold test_selectable_prior -v
```

## 数据契约

使用**原生128观测**的`dataset_sr_128`，不能只将256网格的标签缩小到128后混用。

| 项目 | 设置 |
|---|---|
| 空间网格 | 128×128，单通道实值亮温 |
| GT编码 | uint8，`T = 150 + 170 * g / 255` K |
| 网络归一化 | `z = (T - 235) / 85` |
| 类别划分 | train / val / test = 31 / 7 / 7类 |
| 场景数 | 21,700 / 4,900 / 4,900 |
| 孔径系数 | 1、0.75、0.5、0.35 |
| 观测 | 数据文件中保存的复数可见度，按真实测量行计算 |
| 选择与验证 | 默认val每类20场景，共140场景、560组观测 |

```text
dataset_sr_128/
  gt/<class>/<scene>.npy
  meas/<class>/<scene>.npz
  scenes.json
  geometry.npz
  summary.csv
  audit_report.json
  visibility_verification.json
  native128_generation_report.json
```

详情见[数据格式与物理假设](docs/DATASET_128.md)和[数据传输清单](dataset_transfer_manifest.json)。其中保留的历史Windows路径只用于溯源，不要求其他机器具有相同目录。不要编辑数据清单来替换历史路径，否则会改变数据身份。

## 选择先验并训练

```json
"prior": {"backbone": "unet"}
```

将值改为`"mamba"`即可切换骨干。两份配置除骨干选择外保持一致。

```bash
# U-Net：先验训练完成后自动进入展开训练
python -B train_selectable_prior.py --config configs/server_128_unet.json --data /data/dataset_sr_128 --out runs/unet_train_v1 --device cuda

# Mamba：独立训练，不能加载U-Net权重替代Mamba训练
python -B train_selectable_prior.py --config configs/server_128_mamba.json --data /data/dataset_sr_128 --out runs/mamba_train_v1 --device cuda
```

Windows只需将`/data/dataset_sr_128`替换为实际数据绝对路径，例如`"E:\Paper\Dataset64\v_generate\dataset_sr_128"`。

默认正式预算为先验100,000次有效更新、batch 4，然后冻结最佳先验，训练展开参数20,000次更新、batch 1、梯度累计4次。通过`--phase prior`或`--phase unfold`可单独运行阶段。首次输出目录必须不存在。

先做有限链路检查时，显式使用`--smoke`。它将两阶段各缩短到2次更新、batch 1及小验证子集，不代表正式实验：

```bash
python -B train_selectable_prior.py --config configs/server_128_mamba.json --data /data/dataset_sr_128 --out runs/mamba_smoke_v1 --device cuda --smoke
```

已有匹配骨干的先验时，可以直接训练一套新的展开参数：

```bash
python -B train_selectable_prior.py --config configs/server_128_mamba.json --data /data/dataset_sr_128 --phase unfold --prior /path/to/mamba_prior_best.pt --out runs/mamba_unfold_v1 --device cuda
```

更完整的分阶段命令见[可选先验使用说明](SELECTABLE_PRIOR.md)。

## 断点续训

对原命令增加`--resume`，保持骨干、配置、源码、数据与输出目录一致。例如恢复上面的Mamba pipeline：

```bash
python -B train_selectable_prior.py --config configs/server_128_mamba.json --data /data/dataset_sr_128 --out runs/mamba_train_v1 --device cuda --resume
```

- 续训读取对应阶段的`last.pt`，恢复模型、优化器、AMP、步数、样本游标及随机状态，包括独立加噪生成器。
- 独立启动的展开阶段用相同的`--phase unfold`恢复。只保存`best.pt`不等同于拥有最近的续训进度。
- 新入口拒绝旧格式检查点续训及跨骨干加载。旧U-Net检查点可作U-Net初始化或评价，旧任务仍由原入口继续。
- 旧入口`train_native_hqs.py`在缺少`last.pt`时可能从零开始，即使带有`--resume`。使用旧入口前先确认文件存在。

遇到身份不匹配时先只读诊断，不跳过校验：

```bash
python -B diagnose_selectable_resume.py --out runs/mamba_train_v1 --phase unfold
```

数据路径迁移及源码换行注意事项见[复现与Git管理](docs/REPRODUCIBILITY.md)。

## 评价与可视化

输入新格式的先验或完整展开检查点，入口会按身份选择骨干。先验检查点进行四种噪声等级的去噪诊断，展开检查点评价物理重建。

```bash
python -B evaluate_selectable_prior.py --checkpoint runs/mamba_train_v1/unfold/best.pt --data /data/dataset_sr_128 --out runs/mamba_eval_v1 --device cuda --val-per-class 20
```

输出包含`results.jsonl`、`summary.json`、`identity.json`、`status.json`，展开结果另有K单位的`predictions/`数组。指标包括RMSE、SSIM、PSNR、亮温有符号偏差、标准化观测残差及耗时。该入口仅评价val，不提供test开关。

旧先验的三组机制对照及可视化：

```bash
python -B infer_fixed_prior.py --data /data/dataset_sr_128 --prior /path/to/legacy_unet_prior.pt --out runs/fixed_prior_val_v1 --device cuda
python -B visualize_reconstructions.py --run runs/fixed_prior_val_v1 --out reports/fixed_prior_visuals_v1
```

完整展开、关闭先验、关闭过滤与减弱过滤的入口和前置结果见[推断实验说明](INFERENCE_BASELINE.md)。`evaluate_unfold.py`需要完整旧格式检查点及前期baseline目录，不是只给一个权重就能运行的通用入口。

部分历史绘图脚本写死了Windows微软雅黑路径`C:/Windows/Fonts/msyh.ttc`，Linux绘图需先适配字体。训练和JSON指标评价不依赖该字体。

## 验证范围与结果解释

CPU测试覆盖小型复数物理求解、扫描参考输出与梯度、冻结先验反传、严格权重加载、续训状态和跳过更新后的重放。CI只运行这些无外部数据测试，不启动全量训练，也不提供GPU或重建质量背书。

[历史工程验收](SELECTABLE_PRIOR_ACCEPTANCE.md)记录了当时的原生128短程验证及资源测量，其中的`runs/`和`reports/`文件不随Git发布。旧文档的日期和运行状态是历史快照，不代表此仓库附带了那些权重或当前训练状态。`PACKAGE_MANIFEST.json`也是初始服务器包的历史清单，不是本仓库当前全部文件的清单。

当前对照中的“关闭先验／关闭过滤”是同一带先验、带软过滤训练检查点的推断诊断，不能当作各结构独立训练的公平消融。更低的含噪观测残差不自动代表更真实的图像。合成亮温的改善也不等于真实SAIR空间分辨率已经提升。

## 文档与许可

- [服务器部署与旧入口](README_SERVER.md)
- [数据格式](docs/DATASET_128.md)
- [U-Net／Mamba选择、训练和评价](SELECTABLE_PRIOR.md)
- [复现、续训及Git管理](docs/REPRODUCIBILITY.md)
- [贡献说明](CONTRIBUTING.md)
- [第三方来源与许可](THIRD_PARTY_NOTICES.md)

主项目**暂不授予开源许可，保留所有权利**，见[LICENSE](LICENSE)。公开可见不等于额外授权。第三方VMamba参考及衍生部分保留原MIT许可，本项目声明不缩减第三方原有权利。
