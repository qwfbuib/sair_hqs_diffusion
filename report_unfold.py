"""Report independently evaluated complete unfolded weights."""
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager


def report(out):
    identity=json.loads((out/'identity.json').read_text(encoding='utf-8'))
    status=json.loads((out/'status.json').read_text())
    if status.get('status')!='complete' or status.get('failed'):
        raise ValueError('Need a complete failure-free evaluation')
    summary=json.loads((out/'summary.json').read_text())
    rows=[json.loads(s) for s in (out/'results.jsonl').read_text().splitlines()]
    base=Path(identity['baseline'])
    previous=json.loads((base/'summary.json').read_text())
    oldrows=[json.loads(s) for s in (base/'results.jsonl').read_text().splitlines()]
    root=out.parent.parent
    changed=[name for name,value in identity['training_identity']['code'].items()
             if hashlib.sha256((root/name).read_bytes()).hexdigest()!=value]
    if changed: raise ValueError('Training source changed: '+str(changed))
    if hashlib.sha256((out/'unfold_frozen.pt').read_bytes()).hexdigest()!=identity['checkpoint_sha256']:
        raise ValueError('Frozen checkpoint changed')
    modes=[('固定参数＋未过滤先验',previous['prior_unfiltered']['successful_only_class_gamma_equal']),
           ('固定参数＋软过滤',previous['prior_soft']['successful_only_class_gamma_equal']),
           ('训练后完整展开网络',summary['trained_soft']['class_gamma_equal']),
           ('同一训练检查点关闭过滤〔机制诊断〕',summary['trained_filter_off_diagnostic']['class_gamma_equal'])]
    lines=['# 完整展开网络：第7000步检查点本地验证','',
           f"导入检查点SHA256：`{identity['checkpoint_sha256']}`。完整加载214个状态张量，包括先验与全部32个学习阶段参数。",
           f"与旧基线的先验{identity['same_prior_tensor_count']}个张量逐一相同；先验SHA256：`{identity['prior_sha256']}`。",
           '140场景、四孔径，每模式560次；完整训练配置与同权重关闭过滤的机制诊断，共1120次，均成功。不读取test、不训练或调参。','',
           '## 同先验的结果比较','',
           '| 方法 | K-RMSE ↓ | SSIM ↑ | 标准化残差能量 | 秒/次 |',
           '|---|---:|---:|---:|---:|']
    for name,m in modes:
        lines.append(f"| {name} | {m['rmse_k']:.6f} | {m['ssim']:.6f} | {m['standardized_residual_energy']:.6f} | {m['seconds']:.4f} |")
    lines+=['','未过滤诊断使用为soft配置训练的同一阶段参数，不能当作独立训练并充分调优的无过滤方法。旧基线计时来自先前运行，均包含阶段诊断，跨时段速度仅供参考。标准化残差不是越小越好。','',
            '## 独立复测与服务器记录','',
            f"- 服务器best记录的验证K-RMSE：{identity['server_best_validation']:.9f}。",
            f"- 本地复测K-RMSE：{summary['trained_soft']['class_gamma_equal']['rmse_k']:.9f}；本地减服务器差：{summary['local_minus_server_rmse_k']:+.9f} K。",
            f"- 服务器：{identity['imported_environment']['gpu']}，torch {identity['imported_environment']['torch']}；本地：{identity['device_name']}，torch {identity['torch']}。不承诺跨环境逐位一致。",
            f"- 最大CG真实相对残差：{max(r['max_cg_relative_residual'] for r in rows):.8g}，阈值1e-5。",
            '- 检查点内外identity、数据核心哈希、验证场景列表、原训练源码哈希均匹配。严格加载测试通过，缺失阶段参数会报错，不会静默退回初始化参数。','',
            '## 配对差异','',
            '差值=完整训练模型−旧固定基线；RMSE和残差负值、SSIM正值表示改善。区间按类别内场景重采样，绑定同场景四孔径；不包含训练种子及模型选择的不确定性。','',
            '| 对照 | 指标 | 平均差 | 95%场景重采样区间 | 改善组数 |',
            '|---|---|---:|---|---:|']
    for mode,name in [('prior_unfiltered','固定未过滤'),('prior_soft','固定软过滤')]:
        for key,label in [('rmse_k','RMSE K'),('ssim','SSIM'),('standardized_residual_energy','残差能量')]:
            value=summary['paired_comparisons'][mode][key]
            lo,hi=value['scene_stratified_bootstrap_95_interval']
            lines.append(f"| {name} | {label} | {value['mean_delta']:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {value['improved_pairs']}/{value['pairs']} |")
    lines+=['','## 训练后关闭过滤的机制诊断','']
    for key in ('rmse_k','ssim','standardized_residual_energy'):
        p=summary['filter_off_vs_trained_soft'][key]
        lines.append(f"- {key}：关闭过滤−原训练配置，平均差 {p['mean_delta']:+.6f}，关闭过滤改善 {p['improved_pairs']}/{p['pairs']}组。")
    lines+=['','这个实验只检验当前训练检查点对过滤的依赖，不能据此单独证明soft结构优于独立训练的none结构。','',
            '## 学习到的阶段参数','',
            '| 阶段 | beta | mu | sigma | alpha |','|---|---:|---:|---:|---:|']
    for k in range(8):
        values=[identity['learned_parameters'][key][k] for key in ('beta','mu','sigma','alpha')]
        lines.append('| '+str(k+1)+' | '+' | '.join(f'{v:.6f}' for v in values)+' |')
    lines+=['','初始mu为0.2→0.01，alpha恒为0.5；末阶段mu已经学到约0.043、alpha约0.80。mu变大意味着同一观测奇异方向的过滤减弱；但beta、sigma、alpha也同时改变，不能把总收益归因于mu一个参数。','',
            '## 各孔径 K-RMSE','',
            '| gamma | 固定未过滤 | 固定软过滤 | 完整训练模型 | 训练后关闭过滤 |',
            '|---|---:|---:|---:|---:|']
    for g in identity['gammas']:
        v=[np.mean([r['rmse_k'] for r in oldrows if r['gamma']==g and r['mode']==m])
           for m in ('prior_unfiltered','prior_soft')]
        v.extend(summary[m]['by_gamma'][str(g)]['rmse_k'] for m in ('trained_soft','trained_filter_off_diagnostic'))
        lines.append('| '+str(g)+' | '+' | '.join(f'{x:.6f}' for x in v)+' |')
    lines+=['','## 结论边界','',
            '前期固定参数实验只评价了冻结先验加未训练阶段参数；本次才评价导入的训练后展开网络。两者必须区分。',
            '导入的是训练中第7000次更新的best，不是20000次训练结束后的最终定稿；此验证子集也参与服务器选择best，属于复测和机制诊断，不是独立测试。',
            '原输入权重、配置信息、训练源码和旧结果未修改。下一步可在训练结束后按相同入口复测最终检查点；如果研究过滤的独立贡献，再安排各结构独立训练的公平比较。','',
            '## 重建总览','',
            '复用原总览的每类中位误差附近代表场景（gamma=0.5），不根据新方法的效果重新选图。统一150–320 K色标，原数组未裁剪、转置或翻转。','',
            '![代表样本对比](overview.png)','',
            '## 复用入口','',
            '```powershell',
            'python -B evaluate_unfold.py --checkpoint runs/unfold/best.pt --baseline runs/fixed_prior_val_v1 --out runs/unfold_val_new --device cuda',
            'python -B report_unfold.py --out runs/unfold_val_new',
            '```']
    selection=json.loads((base/'visualizations/selection.json').read_text())['representatives']
    old={(r['scene'],r['gamma'],r['mode']):r for r in oldrows}
    new={(r['scene'],r['gamma'],r['mode']):r for r in rows}
    font='C:/Windows/Fonts/msyh.ttc';font_manager.fontManager.addfont(font)
    plt.rcParams['font.family']=font_manager.FontProperties(fname=font).get_name()
    titles=('真值','固定参数＋未过滤先验','训练后完整展开网络','同权重关闭过滤〔诊断〕')
    fig,axes=plt.subplots(7,4,figsize=(12,18.9),layout='constrained')
    for i,scene in enumerate(selection):
        gt=(150+170*np.load(Path(identity['data'])/'gt'/(scene+'.npy')).astype(float)/255).astype(np.float32)
        arrays=[gt,np.load(base/'predictions/prior_unfiltered/0.5'/(scene+'.npy')),
                np.load(out/'predictions/trained_soft/0.5'/(scene+'.npy')),
                np.load(out/'predictions/trained_filter_off_diagnostic/0.5'/(scene+'.npy'))]
        info=[None,old[(scene,.5,'prior_unfiltered')],new[(scene,.5,'trained_soft')],new[(scene,.5,'trained_filter_off_diagnostic')]]
        for j,a in enumerate(arrays):
            ax=axes[i,j];im=ax.imshow(a,cmap='inferno',vmin=150,vmax=320,interpolation='nearest',origin='upper')
            ax.set_xticks([]);ax.set_yticks([])
            if i==0:ax.set_title(titles[j],fontsize=11,pad=9)
            if j==0:
                ax.set_ylabel(scene.split('/')[0].replace('rectangular_farmland','farmland').replace('thermal_power_station','power station'),fontsize=9)
                ax.set_xlabel(scene.split('/')[-1],fontsize=8)
            else:
                ax.set_xlabel(f"RMSE {info[j]['rmse_k']:.2f} K · SSIM {info[j]['ssim']:.3f}",fontsize=9)
    fig.colorbar(im,ax=axes.ravel().tolist(),label='亮温 / K',shrink=.5,pad=.015)
    fig.suptitle('完整展开网络复测 · 第7000步 · gamma=0.5\n同一先验、同一观测；末列是同权重关闭过滤的机制诊断',fontsize=14)
    fig.savefig(out/'overview.png',dpi=150);plt.close(fig)
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(report=str(out/'REPORT.md'),trained=summary['trained_soft']['class_gamma_equal'],
                         diagnostic=summary['trained_filter_off_diagnostic']['class_gamma_equal'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    report(parser.parse_args().out.resolve())
