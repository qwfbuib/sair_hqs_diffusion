# 原生128网格可见度数据使用说明

日期：2026-10-24。格式仍为`sair-scene-v2`，使用现有生成器的`--n-grid 128`。这里的“128”表示GT与前向求和的网格大小，不表示阵列具有128个横向分辨单元。数据生成/验收状态以新目录中的报告为准。

**当前状态：全量已生成并通过验收，可以直接读取。** 31,500张GT、31,500个场景测量文件、126,000条观测；文件字节合计1,988,047,382（约1.85 GiB，不含文件系统分配簇开销）。无需为开始训练重跑生成器。

## 选择正确的数据根目录

|数据根目录|可见度来自什么|GT存放|默认读取/训练目标|
|---|---|---|---|
|`E:\Paper\Dataset64\v_generate\dataset_sr`|256×256模拟亮温|`gt`存256；`gt128`存补充128标签|256；`target_size=128`只改标签，不改观测|
|`E:\Paper\Dataset64\v_generate\dataset_sr_128`|**128×128模拟亮温**|`gt`存128|128，观测与标签来自同一128网格|
|`E:\Paper\Dataset64\v_generate\dataset_sr_128_pilot`|同一128模型，每类前40张|`gt`存128|128，开发/验收用子集|

需要满足`y=G128(T128)+noise`时，使用`dataset_sr_128`。不能通过给旧`dataset_sr`加`target_size=128`获得这套新观测。新根目录内的128 GT仍保存在`gt`，不另建同名`gt128`副本。内置接口不从新根目录返回256标签。

## 源图与物理设置

- 源图：`E:\Paper\Dataset64\NWPU-RESISC45_multiscale\128`，直接检查128×128尺寸并用Pillow转灰度，不缩放、不插值。
- uint8 GT编码0～255；按`T=150+170*g/255`解码为模拟亮温K。现有`dataset_sr/gt128`是同场景标签，全量验收另外逐文件确认两份GT的数值一致。
- Y阵61根天线，0.875波长间距，四个gamma为1/.75/.5/.35；视场方向余弦宽度1.05。
- 沿用归一化几何权重、`F=1`均匀方向图假设、忽略条纹洗涤的理想化SAIR模型，不宣称完整MIRAS仪器仿真。
- 物理噪声复RMS为`350/sqrt(19e6*1.2)`，约0.07330 K；实/虚部分别为该值除以sqrt(2)。新可见度由128 GT重新前向计算，不从旧测量缩放或截取。
- 种子20261022、完整清单scene_index、gamma顺序与256版本对齐。因此对应场景使用相同的随机噪声种子，而不是两个独立噪声试验；两套无噪信号一般不同。跨网格比较应记录这种配对关系。
- 31/7/7类别互斥划分不变，train/val/test仍为21,700/4,900/4,900场景；训练及扩散先验只用train。

阵列和孔径不变，所以每条可见度的基线数不因128/256网格切换而改变。约束维数比例要使用128²=16,384重新计算：

|gamma|复测量行M|独立复基线|实约束上限|null_fraction下界|
|---|---:|---:|---:|---:|
|1.00|1830|1260|2520|0.84619140625|
|0.75|1635|1065|2130|0.8699951171875|
|0.50|1098|537|1074|0.9344482421875|
|0.35|753|267|534|0.9674072265625|

这仍是实方程预算给出的下界，不是实际矩阵秩测量，也不是未约束“像素”的比例。

## 文件与读取

```text
dataset_sr_128/
  scenes.json
  geometry.npz
  summary.csv
  gt/<class>/<name>.npy         (128,128) uint8
  meas/<class>/<name>.npz       四gamma可见度与噪声元数据
  distribution.json
  audit_report.json
  visibility_verification.json
  training_handoff_check.json
  native128_generation_report.json
```

每场景NPZ的`y_g100/y_g075/y_g050/y_g035`仍为`(1,1830)/(1,1635)/(1,1098)/(1,753)`、complex64。第一维为噪声重复次数，默认1。u/v和孔径行索引只存于共享`geometry.npz`，格式与256版本一致。

```python
from sair_dataset.reader import DatasetReader
from sair_dataset.training import SAIRTrainingDataset

root = r"E:\Paper\Dataset64\v_generate\dataset_sr_128"
reader = DatasetReader(root)
scene = "airplane/airplane_001"
obs = reader.load(scene, gamma=1.0, repeat=0)
T128 = reader.truth(scene)  # (128,128)，float64，单位K，已解码
assert obs["n_grid"] == 128

train = SAIRTrainingDataset(root, split="train")  # 也可显式target_size=128
sample = train[0]
assert sample["target_k"].shape == (1,128,128)  # float32，K，无需再解码
assert sample["inputs"]["physics"]["n_grid"] == 128
assert sample["target_n_grid"] == 128
# target_z为同尺寸float32、0～1；模型只接收sample["inputs"]。
```

DataLoader批处理后`target_k/target_z`为`(B,1,128,128)`，可见度实/虚输入仍补零为`(B,1830,2)`并提供mask。相位、权重和物理损失必须采用新根目录的N=128参数，不能沿用256的离散算子。

## 复现与验收

以下生成命令要求输出目录尚不存在；已有数据不应覆盖。输出目录名可以替换为新的独立位置。

```powershell
python .\generate.py --input "E:\Paper\Dataset64\NWPU-RESISC45_multiscale\128" --output dataset_sr_128 --n-grid 128 --per-class 0 --gammas "1.0,0.75,0.5,0.35" --batch 8 --seed 20261022 --noise-repeats 1 --bt-min 150 --bt-max 320
python .\audit_dataset.py --data dataset_sr_128 --compare-pilot dataset_sr_128_pilot --workers 4
python .\verify_visibility.py --data dataset_sr_128 --report dataset_sr_128/visibility_verification_new.json --workers 4
python .\check_training_data.py --data dataset_sr_128 --target-size 128 --report experiment_runs/native128_training_check_new.json
python .\visualize.py --data dataset_sr_128 --scene "airplane/airplane_001" --gamma 1.0 --out visualizations_128
```

验收顺序：1,800场景pilot生成→结构/物理/全部噪声重放→分带量纲检查→31,500场景全量生成→全部文件结构检查、与pilot逐位比对→全量独立前向及126,000条观测噪声重放→训练接口检查→与已有gt128逐文件比对。源代码、环境、命令、日志及旧数据校验值记录在`experiment_runs/native128_generation_20261024_v1`。64项回归测试包含原生128生成、独立验收、批大小逐位一致和训练接口。

本轮不更改旧`dataset_sr`及其观测；旧256数据仍可按原方式使用。原格式及训练规范的完整交接见[DATASET_HANDOFF.md](DATASET_HANDOFF.md)。

## 本轮实际验收结果

- 64项代码测试通过，包含原生128网格端到端用例。
- 全量结构验收`status=pass`、warnings为空；试生成1,800场景与全量对应文件逐位一致。
- 独立核验全部31,500场景及126,000条含噪观测，全部源图灰度一致、种子噪声重放逐位一致；最大独立前向差`3.7236209404178557e-6 K`，零基线300 K归一化误差为0。
- 新`dataset_sr_128/gt`与已有`dataset_sr/gt128`的31,500个标签全部一致；两个版本的有序uv及孔径索引一致，场景ID、scene_index和31/7/7类别划分一致。
- 新根目录的Reader返回`(128,128)` float64 K，训练`target_k`返回`(1,128,128)` float32 K；实际24条训练样本检查通过，输入`physics.n_grid=128`。
- 全量生成567.185秒；含试生成、验收、全量独立重放及旧GT对齐检查的流水线1246.267秒，约20分46秒。
- 原256版本的scenes.json、geometry.npz、summary.csv及gt128_report.json校验值保持不变。

完整报告：[native128_generation_report.json](dataset_sr_128/native128_generation_report.json)、[结构验收](dataset_sr_128/audit_report.json)、[全量独立核验](dataset_sr_128/visibility_verification.json)、[训练接口](dataset_sr_128/training_handoff_check.json)。

图例：[128亮温与可见度示例](experiment_runs/native128_generation_20261024_v1/figures/airplane/airplane_001_g100_n0.png)。全量GT/观测内容指纹：`65ebc30138b0e5482cdac4cb67ec522133e5a52396e1f113be75f5abde5afd0d`。
