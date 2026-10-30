"""Paired diagnostic of removing the prior from an already trained checkpoint."""
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import common
from verify_filter_fade import comparison


def report(out,reference):
    info=json.loads((out/'identity.json').read_text(encoding='utf-8'))
    oldinfo=json.loads((reference/'identity.json').read_text(encoding='utf-8'))
    if any(info[k]!=oldinfo[k] for k in ('checkpoint_sha256','data_core','scenes','gammas','local_spectral_scales')):
        raise ValueError('Checkpoint, data, selection or operator scaling differ')
    if hashlib.sha256((out/'unfold_frozen.pt').read_bytes()).hexdigest()!=info['checkpoint_sha256']:
        raise ValueError('Frozen checkpoint differs')
    if any(common.sha(common.ROOT/p)!=h for p,h in info['training_identity']['code'].items()):
        raise ValueError('Training source changed')
    newrows=[json.loads(s) for s in (out/'results.jsonl').read_text().splitlines()]
    oldrows=[json.loads(s) for s in (reference/'results.jsonl').read_text().splitlines()]
    if len(newrows)!=560 or any(r['status']!='ok' or r['prior_calls']!=0 for r in newrows):
        raise ValueError('Incomplete zero-prior run')
    if any(t['prior_visibility_change_rms']!=0 or t['filter_cg_residual']!=0 for r in newrows for t in r['trace']):
        raise ValueError('Prior or filter was active')
    without=json.loads((out/'summary.json').read_text())['trained_without_prior_diagnostic']['class_gamma_equal']
    original=json.loads((reference/'summary.json').read_text())
    comp={mode:comparison([r for r in oldrows if r['mode']==mode],newrows)
          for mode in ('trained_soft','trained_filter_off_diagnostic')}
    common.write_json(out/'paired_comparisons.json',comp)
    lines=['# 第7000步展开检查点：关闭先验诊断','',
           '这是在已经按带先验、带软过滤配置训练的同一检查点上，测试时关闭整个先验修正。不是独立训练的无先验网络。',
           '保留初始数据步与八阶段学习到的beta；mu、sigma、alpha因修正为零而不再影响输出。没有另一个学习图像结构的CNN分支。',
           '同一140场景、四孔径，共560次，全部成功。与完整模型的检查点哈希、数据身份、场景列表、孔径和实际算子归一化尺度逐项相同。未读取test或训练模型。','',
           '## 结果','',
           '| 配置 | K-RMSE ↓ | SSIM ↑ | 标准化残差能量 | 秒/次 |',
           '|---|---:|---:|---:|---:|']
    entries=[('同一训练检查点关闭先验',without),
             ('扩散先验＋无过滤〔同权重诊断〕',original['trained_filter_off_diagnostic']['class_gamma_equal']),
             ('完整训练模型：扩散先验＋软过滤',original['trained_soft']['class_gamma_equal'])]
    for label,r in entries:
        lines.append(f"| {label} | {r['rmse_k']:.6f} | {r['ssim']:.6f} | {r['standardized_residual_energy']:.6f} | {r['seconds']:.4f} |")
    lines+=['','时间包含阶段诊断，且不同配置来自不同运行时段，仅供参考。较低含噪观测残差并非自动意味着图像更真实。','',
            '## 先验加入后的配对变化','',
            '差值=带先验配置−关闭先验；RMSE/残差负值、SSIM正值为改善。区间按类别内场景重采样，绑定同一场景四孔径，不含模型选择和训练种子不确定性。','',
            '| 带先验配置 | 指标 | 平均差 | 95%场景重采样区间 | 改善组数 |',
            '|---|---|---:|---|---:|']
    for mode,label in [('trained_soft','完整模型'),('trained_filter_off_diagnostic','先验＋无过滤诊断')]:
        for key in ('rmse_k','ssim','standardized_residual_energy'):
            p=comp[mode][key];lo,hi=p['scene_stratified_bootstrap_95_interval']
            lines.append(f"| {label} | {key} | {p['mean_delta']:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {p['improved_pairs']}/{p['pairs']} |")
    lines+=['','## 核验与边界','',
            f"- 最大CG真实相对残差：{max(r['max_cg_relative_residual'] for r in newrows):.8g}，门槛1e-5。",
            '- 所有记录prior_calls=0；各阶段先验引起的可见度变化与过滤CG残差均为0。',
            '- 单元测试用禁止调用的先验模块验证没有调用网络，输出与保留同一阶段参数的prior_kind=none模型逐元素完全一致。',
            '- 当前诊断评价的是既有模型对先验的依赖，不能保证先验优于重新训练或重新优化的传统正则化方法。',
            '- 现有模型先验训练/展开训练均已使用相应train数据；此处为原验证子集上的诊断，不是独立test性能。',
            '- 原训练模块与输入权重未修改；只扩展独立评价入口的模式选择，默认原两模式保持兼容。','',
            '## 同样代表场景的重建总览','',
            '沿用原每类别中位误差附近样本、gamma=0.5。统一亮温色标150–320 K，原数组未截断、转置或翻转。','',
            '![关闭先验对比](overview.png)','',
            '## 运行方式','',
            '```powershell',
            'python -B evaluate_unfold.py --checkpoint runs/unfold/best.pt --baseline runs/fixed_prior_val_v1 --out runs/unfold_no_prior_new --device cuda --modes trained_without_prior_diagnostic',
            'python -B report_without_prior.py --out runs/unfold_no_prior_new --reference runs/unfold_val_step7000_v1',
            '```']
    font='C:/Windows/Fonts/msyh.ttc';font_manager.fontManager.addfont(font)
    plt.rcParams['font.family']=font_manager.FontProperties(fname=font).get_name()
    base=Path(info['baseline'])
    selected=json.loads((base/'visualizations/selection.json').read_text())['representatives']
    new={(r['scene'],r['gamma']):r for r in newrows}
    old={(r['scene'],r['gamma'],r['mode']):r for r in oldrows}
    titles=('真值','训练检查点关闭先验〔诊断〕','加入先验、关闭过滤〔诊断〕','完整训练模型：先验＋过滤')
    fig,axes=plt.subplots(7,4,figsize=(12,18.9),layout='constrained')
    for i,scene in enumerate(selected):
        gt=(150+170*np.load(Path(info['data'])/'gt'/(scene+'.npy')).astype(float)/255).astype(np.float32)
        arrays=[gt,np.load(out/'predictions/trained_without_prior_diagnostic/0.5'/(scene+'.npy')),
                np.load(reference/'predictions/trained_filter_off_diagnostic/0.5'/(scene+'.npy')),
                np.load(reference/'predictions/trained_soft/0.5'/(scene+'.npy'))]
        records=[None,new[(scene,.5)],old[(scene,.5,'trained_filter_off_diagnostic')],old[(scene,.5,'trained_soft')]]
        for j,a in enumerate(arrays):
            ax=axes[i,j];im=ax.imshow(a,cmap='inferno',vmin=150,vmax=320,interpolation='nearest',origin='upper')
            ax.set_xticks([]);ax.set_yticks([])
            if i==0:ax.set_title(titles[j],fontsize=10,pad=9)
            if j==0:
                ax.set_ylabel(scene.split('/')[0].replace('rectangular_farmland','farmland').replace('thermal_power_station','power station'),fontsize=9)
                ax.set_xlabel(scene.split('/')[-1],fontsize=8)
            else:
                ax.set_xlabel(f"RMSE {records[j]['rmse_k']:.2f} K · SSIM {records[j]['ssim']:.3f}",fontsize=9)
    fig.colorbar(im,ax=axes.ravel().tolist(),label='亮温 / K',shrink=.5,pad=.015)
    fig.suptitle('同一训练展开检查点：先验作用诊断 · 第7000步 · gamma=0.5\n三组复用同一学习阶段参数；移除先验并非独立训练的无先验方法',fontsize=13)
    fig.savefig(out/'overview.png',dpi=150);plt.close(fig)
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(without_prior=without,paired=comp,report=str(out/'REPORT.md'))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    args=parser.parse_args()
    report(args.out.resolve(),args.reference.resolve())
