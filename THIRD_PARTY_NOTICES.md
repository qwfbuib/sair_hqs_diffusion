# 第三方来源与许可

## VMamba参考与SS2D适配

- 上游仓库：[MzeroMiko/VMamba](https://github.com/MzeroMiko/VMamba)
- 固定提交：`2ed52ead062a51a64521ed3871d52914bf532876`
- 上游文件：[vmamba.py](https://github.com/MzeroMiko/VMamba/blob/2ed52ead062a51a64521ed3871d52914bf532876/vmamba.py)
- 许可证：MIT，`Copyright (c) 2024 MzeroMiko`
- 原许可全文：[VMAMBA_LICENSE.txt](selectable_prior/reference/VMAMBA_LICENSE.txt)
- 保存的独立参考：[vmamba_reference.py](selectable_prior/reference/vmamba_reference.py)
- 来源、参考函数与字节哈希：[provenance.json](selectable_prior/provenance.json)

`selectable_prior/ss2d.py`是遵循固定参考递推和四方向路由的本地原生PyTorch适配。它采用分块关联扫描，并调整了输入需要梯度时的重计算条件，包括冻结且处于eval模式的先验。该实现不是官方fused内核，不应将本地速度测量当作官方VMamba性能。

本项目的“保留所有权利”声明不缩减这些第三方材料已经通过MIT授予的权利。

## 项目内部快照

`vendor/sair_dataset`、`vendor/sair_experiments`和`vendor/sair_unfold`保存训练所依赖的项目模块快照。`vendor`目录名不表示这些模块自动具有第三方开源许可。初始服务器包的来源哈希记录在`PACKAGE_MANIFEST.json`中。

`provenance.json`里的历史本机路径仅用于记录迁移来源，不是运行依赖，不要为了替换路径而改变已有检查点绑定的来源文件。

## 数据与运行依赖

数据集、图像、检查点及实验输出不包含在此代码发布中。若使用NWPU-RESISC45来源图像构造合成亮温，请另外遵循原数据集的使用条件。本仓库的主代码或第三方代码许可不构成数据许可。

PyTorch、NumPy、Pillow、Matplotlib和SciPy通过依赖安装获取，仍适用各自的许可证；本仓库没有重新发布它们的安装包。
