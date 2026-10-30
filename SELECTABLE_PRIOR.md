# 可选择 U-Net／Mamba 的 VE 先验

新入口与原 `train_native_hqs.py` 并行。原训练模块、原配置、输入权重和已有实验结果未修改。新增代码不需要 mamba_ssm、Triton 或编译扩展，不升级现有环境。

## 选择骨干

在配置顶层增加：

```json
"prior": {"backbone": "unet"}
```

改成 `"mamba"` 使用混合Mamba骨干。未写prior时默认为unet；未知选项报错。两份完整配置为 `configs/server_128_unet.json` 和 `configs/server_128_mamba.json`；除骨干选择外，数据、模型尺寸、训练预算和损失权重一致。

接口可直接调用：

```python
import common
from sair_unfold.config import Config
from selectable_prior import SelectablePrior, SelectableHQS

cfg = Config(image_size=128, t_min=150, t_max=320).validate()
prior = SelectablePrior(cfg, backbone="mamba")
epsilon = prior.epsilon(z_sigma, sigma)   # [B,1,128,128]
clean = prior(z_sigma, sigma)             # z_sigma - sigma * epsilon
loss = prior.dsm_loss(clean_target, generator=noise_generator)
model = SelectableHQS(cfg, backbone="mamba", projection="soft")
prediction = model(complex_visibility, physics)
```

完整模型实际使用训练配置里的所有字段，手动调用示例不用于替代训练配置。这里 z=(T-235)/85，sigma是归一化图像域噪声，物理观测仍以K计算。

## Mamba的具体结构

保留原U-Net的多尺度卷积、跳跃连接、上/下采样及输出头，仅将 `encoder.2.1` 和 `middle.1` 两个残差块替换为带噪声条件的四方向SS2D。标准配置为128×128、基础宽度32，对应32×32和16×16、128通道特征。它是Mamba–U-Net混合骨干，不是纯Mamba网络。

SS2D参数固定：d_state=16、expand=2、d_conv=3、chunk_size=32，噪声条件使用原TimeEmbedding输出。四方向为行、列及其反向；扫描跨分块保留状态。递推为FP32，输入需要梯度时做扫描重计算，包括先验冻结且处于eval模式的展开训练。

扫描来自旧项目的独立原生PyTorch适配；上游参考固定为VMamba commit `2ed52ead062a51a64521ed3871d52914bf532876`。来源、字节哈希、具体适配见 `selectable_prior/provenance.json`；MIT许可和未改写参考函数保存在 `selectable_prior/reference/`。不是官方fused内核的速度复现。

## 正式训练命令

在项目根目录运行；服务器请替换数据和Python环境路径。以下为正式预算，不是本次已执行的任务。

```powershell
# 本机示例
$py = 'C:\Users\Lenovo\.conda\envs\sair\python.exe'
$data = 'E:\Paper\Dataset64\v_generate\dataset_sr_128'

# U-Net独立运行
& $py -B train_selectable_prior.py --config configs/server_128_unet.json --data $data --out runs/selectable_unet_train_v1 --device cuda

# Mamba独立运行
& $py -B train_selectable_prior.py --config configs/server_128_mamba.json --data $data --out runs/selectable_mamba_train_v1 --device cuda

# 对同一新格式运行恢复；保留原配置、源码和数据身份
& $py -B train_selectable_prior.py --config configs/server_128_mamba.json --data $data --out runs/selectable_mamba_train_v1 --device cuda --resume
```

默认pipeline先进行100000次先验更新，再加载该运行的prior/best.pt、冻结先验，训练20000次展开更新。先验batch4；展开batch1、累计4次。`--phase prior` 或 `--phase unfold` 可分别运行。首次输出根目录必须不存在；新阶段可接入同一运行根目录。`--stop-after N` 在每阶段达到N次有效更新后受控保存停止，不改正式预算。

私有加噪生成器与模型初始化随机数分离；其状态与全局随机状态、AMP scaler、优化器、样本游标一起保存。AMP跳过更新时重放同一批输入，最多16次连续重试。严谨A/B应在新入口分别从头训练两种骨干；此前U-Net训练使用不同的加噪随机数流，旧权重可作历史参照，不能冒称同噪声抽样的架构消融。

新训练每个运行保存有效配置和source/快照；检查点记录骨干、结构版本、后端、数据核心哈希和全部相关源码哈希。源码或身份变化时拒绝续训；不要把新源码混入已经运行的source快照。原入口的续训继续使用原入口及原配置。

## 使用已有先验与旧权重

```powershell
# 使用旧U-Net先验初始化一套新的展开训练，不是恢复旧展开优化器
& $py -B train_selectable_prior.py --config configs/server_128_unet.json --data $data --phase unfold --prior runs/prior/best.pt --out runs/selectable_unet_from_existing --device cuda

# Mamba展开训练必须加载独立训练的Mamba先验
& $py -B train_selectable_prior.py --config configs/server_128_mamba.json --data $data --phase unfold --prior runs/selectable_mamba_train_v1/prior/best.pt --out runs/selectable_mamba_unfold_v1 --device cuda
```

U-Net保持原state_dict名称，旧先验和完整展开检查点可用于U-Net初始化/评价。旧格式不能在新入口直接续训；跨骨干加载、错误phase、缺失张量、错误数据身份或架构版本均报错。展开输入先验复制到该阶段的input_prior.pt，外部best以后更新不改变这次训练。显式提供不同先验恢复会被拒绝。

## 自动选择骨干的评价

```powershell
# 输入完整展开检查点：在val上评价并保存K单位重建图
& $py -B evaluate_selectable_prior.py --checkpoint runs/selectable_mamba_train_v1/unfold/best.pt --data $data --out runs/selectable_mamba_eval_v1 --device cuda

# 小规模旧完整U-Net兼容评价
& $py -B evaluate_selectable_prior.py --checkpoint runs/unfold_val_step7000_v1/unfold_frozen.pt --data $data --out runs/selectable_legacy_eval_new --device cuda --val-per-class 1
```

输入先验检查点时自动做四个sigma的独立去噪诊断；输入展开检查点时按训练时projection进行重建。所有评价仅选原val类，每类默认20场景，最多20；不提供test开关。输出目录拒绝覆盖，检查点冻结复制，模型自动设为eval/no_grad。

评价亦检查训练源码身份；若源码版本不同，使用检查点保存的source/，不自动忽略不匹配。source快照可以配合绝对配置/数据/检查点路径使用。跨系统路径分隔符或环境变更不承诺直接续训或逐位一致。

## 工程检查与当前验收

```powershell
& $py -B -m unittest test_selectable_prior -v

# 显式缩短为两阶段各2更新、batch1、小验证子集；不能当正式实验
& $py -B train_selectable_prior.py --config configs/server_128_mamba.json --data $data --out runs/selectable_mamba_smoke_new --device cuda --smoke

# 真实旧权重兼容，以及完整128网格原batch配置的资源检查
& $py -B check_selectable_prior.py --data $data --out reports/selectable_resources_new --device cuda --batch 4
```

单元测试包括独立扫描参考的输出与全部输入梯度、跨分块状态、冻结eval扫描重计算、噪声条件和A/B同随机输入、旧权重兼容、跨骨干拒绝、断点恢复及跳过更新后的重放一致性。

本次只完成工程检查和资源测量，未启动全量Mamba训练，也未证明Mamba质量更好、速度更快或高频/空间分辨率提升。具体运行目录和资源数值见 `SELECTABLE_PRIOR_ACCEPTANCE.md`。

两种骨干参数量不同；相同数据和更新预算的A/B属于骨干整体方案比较，不是严格等参数或等计算量对照。原生PyTorch扫描的资源结果也不能外推到官方加速内核。
