# 上传服务器所需文件与运行方法

本包包含本机当前v3训练任务使用的训练源码，模型和训练循环保持原样。按Linux＋NVIDIA单卡服务器整理；其他系统也可直接运行Python入口。未连接或修改你的服务器，也未停止本机训练。

## 1. 上传两部分

**A. 本代码包整个文件夹**，解压后保持以下结构，尤其不要拆散vendor目录：

```text
sair_hqs_server/
  train_native_hqs.py           两阶段训练、验证、保存和续训；内含NativeData加载器
  common.py                    本地模块路径、哈希、文件工具
  torch_operator.py            可微复数前向与实伴随
  unfold_bridge.py             原生128 Y阵到HQS的适配
  configs/server_128.json       128/150–320K/8阶段/四gamma训练配置
  vendor/
    sair_dataset/              reader.py/array.py/operator.py/generate.py/training.py等
    sair_experiments/          operators.py等
    sair_unfold/               config/model/prior/physics/numerics/unfolding/losses等
  configure_data.py            设置服务器数据路径
  check_server.py              检查依赖、数据传输、前向与梯度
  import_prior_checkpoint.py   可选：迁移本机先验检查点
  requirements.txt
  requirements-torch-cu118.txt
  dataset_transfer_manifest.json
  docs/DATASET_128.md
```

三个vendor子目录按原样保留，代码量很小；部分模块虽未在本次训练调用，但被源码身份哈希记录，保留可避免不必要的检查点不兼容。

**B. 原生128数据**：从本机 `E:\Paper\Dataset64\v_generate\dataset_sr_128` 上传以下内容，建议直接复制整个原目录（约1.85GiB）：

```text
dataset_sr_128/
  gt/<class>/<stem>.npy          全部31500张128×128 uint8 GT
  meas/<class>/<stem>.npz        全部31500个场景，保存四gamma复数y
  scenes.json                   物理参数、场景ID、原类别划分
  geometry.npz                  uv、原测量行顺序与孔径indices
  summary.csv                   输入身份哈希需要
  audit_report.json             训练启动时检查验收状态
  visibility_verification.json  验收状态与输入身份哈希
  native128_generation_report.json
```

以上两个目录、六个元数据/报告文件是本训练器实际需要的集合。原数据的distribution.json、training_handoff_check.json可以一起复制，但不是训练必需。保留全部45类目录和原清单最省事；训练仍只读31个train类、验证只读7个val类，test不用于训练/选择。

不需要上传源JPEG、旧dataset_sr的256观测、dataset_sr/gt128、工作区data/full_v1多噪声目录、pilot、旧失败实验、Windows虚拟环境、.pip_tmp或__pycache__。本代码包不含大数据或模型权重，数据需要单独上传。

**不要修改数据内scenes.json等文件中的历史Windows路径**。读现成GT/y不访问源JPEG，这些路径可保留作为生成溯源；修改它们反而会破坏原数据hash。只修改训练配置的data路径即可。

## 2. 数据加载代码在哪里

实际训练使用 `train_native_hqs.py` 中的 `NativeData`：按scenes.json保持原类别划分；GT从gt读取并解码K；HQS观测通过 `vendor/sair_dataset/reader.py:DatasetReader.load` 从meas读取。模型只使用当前gamma的已保存y，训练不调用Physics.simulate重新加噪。

`vendor/sair_dataset/training.py` 另提供官方的SAIRTrainingDataset批读取接口，一并包含，但当前HQS训练入口不使用它的补零观测。不同gamma的M不同，本轮HQS使用batch1与梯度累计4，按真实行计算似然。

## 3. 环境

优先使用服务器已有、与其GPU匹配的CUDA版PyTorch环境，Python建议3.10。先确认：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python -m pip install -r requirements.txt
```

本机已验证Python3.10.21、torch2.1.2+cu118、numpy1.26.4、Pillow12.3.0。requirements.txt只安装NumPy和Pillow，避免意外覆盖服务器已配置好的GPU版torch。SciPy、Matplotlib、OpenCV、torchvision、torchaudio不是这条训练入口的必需项。

若新建与本机接近的环境，且服务器GPU/驱动支持CUDA11.8构建，可参照：

```bash
conda create -n sair128 python=3.10 -y
conda activate sair128
python -m pip install -r requirements-torch-cu118.txt
python -m pip install -r requirements.txt
```

torch2.1.2的CUDA11.8/12.1安装方式见[PyTorch官方历史版本说明](https://pytorch.org/get-started/previous-versions/#v212)。新显卡可能需要更新的torch/CUDA组合，应以该服务器的兼容性为准，并通过下面的GPU检查。跨系统、GPU或库版本不承诺逐位一致；环境会随训练记录。

## 4. 设置路径并检查

在解压后的sair_hqs_server目录运行。假设数据放在 `/data/dataset_sr_128`：

```bash
cd /path/to/sair_hqs_server
python configure_data.py --data /data/dataset_sr_128
python check_server.py --device cuda
# 可选：对31500个GT和31500个测量文件做全内容指纹检查
python check_server.py --device cuda --full-data-fingerprint
```

默认server_128.json使用`./dataset_sr_128`；如果数据就在代码目录内，可不运行configure_data。配置路径和data相对路径都按当前工作目录解析，因此按以上顺序先cd。本配置保持原31/7/7类别、150–320K、四gamma、噪声模型和训练预算。

## 5. 在服务器从头训练

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONDONTWRITEBYTECODE=1
python -u train_native_hqs.py \
  --config configs/server_128.json \
  --out runs/hqs128_server_v1 \
  --phase pipeline --device cuda
```

后台运行可使用服务器的作业调度器；普通Linux终端也可：

```bash
mkdir -p runs/hqs128_server_v1
nohup python -u train_native_hqs.py \
  --config configs/server_128.json \
  --out runs/hqs128_server_v1 \
  --phase pipeline --device cuda \
  > runs/hqs128_server_v1/stdout.log 2> runs/hqs128_server_v1/stderr.log &
echo $!
```

不要同时执行前台和后台两种启动命令。重复运行已有阶段目录时必须用--resume，或使用新目录。服务器上同一目录续训：

```bash
python -u train_native_hqs.py --config configs/server_128.json \
  --out runs/hqs128_server_v1 --phase pipeline --device cuda --resume
```

预算：先验100000次有效更新、batch4；随后自动加载best先验，冻结先验权重，训练8阶段HQS的32个阶段参数20000次，batch1累计4次。HQS阶段不是整个UNet联合微调。140个固定val场景用于选择（四gamma共560观测），最终test不运行。

## 6. 可选：接着本机已有先验继续

从头训练不用上传任何.pt。如需接着本机进度，需要从当前有效本机运行目录 `runs/hqs_native128_20261024_v3/prior/` 另外复制 `best.pt` 和 `last.pt`，例如上传到服务器 `checkpoints/local_prior/`。不要复制.tmp文件；若复制期间两检查点先后更新，导入脚本会检查step顺序，需要重复制last.pt。最佳和最近检查点各约51MiB。

Windows和服务器数据绝对路径、源码路径分隔符不同，直接--resume会被原身份校验拒绝。因此先在服务器显式迁移：

```bash
python import_prior_checkpoint.py --config configs/server_128.json \
  --checkpoints checkpoints/local_prior --out runs/hqs128_server_resume
python -u train_native_hqs.py --config configs/server_128.json \
  --out runs/hqs128_server_resume --phase pipeline --device cuda --resume
```

导入必须使用本包完全匹配的训练源码、相同模型/训练配置、未改变的原数据文件。它只重绑定服务器数据根路径和源码路径分隔符，逐项核对模型、优化器、scaler、RNG和游标未变。更换GPU/系统后可接着优化，但不是逐位重演保证。该工具只适用于当前先验阶段检查点；若本机已开始HQS阶段，迁移其检查点还需同时绑定使用的先验，不应套用本工具。

## 7. 结果位置和验证范围

运行目录status.json给阶段与步数，stdout/stderr给日志；prior和unfold内分别有best.pt、last.pt、metrics.jsonl、identity.json、environment.json。检查点每250次更新保存，验证时也保存。

本包的源码与本机v3逐文件核对；独立目录导入、原生128数据加载、前向/梯度和检查点导入/短续训在本机环境验证。服务器Linux和目标GPU仍需运行check_server实测，本次未宣称已在未知服务器验证。
