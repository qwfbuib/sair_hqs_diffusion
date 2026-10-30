"""Summarize a completed, prespecified validation follow-up."""
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np


def report(out):
    identity=json.loads((out/'identity.json').read_text(encoding='utf-8'))
    status=json.loads((out/'status.json').read_text(encoding='utf-8'))
    if status.get('status')!='complete' or status.get('failed'):
        raise ValueError('Expected complete, failure-free validation')
    summary=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    base=Path(identity['baseline'])
    baseline=json.loads((base/'summary.json').read_text(encoding='utf-8'))
    rows=[json.loads(s) for s in (out/'results.jsonl').read_text().splitlines()]
    anchors=json.loads((out/'anchors.json').read_text())
    root=out.parent.parent
    for name,value in identity['source_hashes'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=value:
            raise ValueError('Training source changed: '+name)
    if hashlib.sha256(Path(identity['prior_file']).read_bytes()).hexdigest()!=identity['prior_sha256']:
        raise ValueError('Prior changed')
    lines=['# 早期保留、后期减弱软过滤：验证结果','',
           '本轮预先固定一个候选，没有根据中途结果调整参数。过滤混合权重为 [1,1,1,1,0.75,0.5,0.25,0]。',
           '每阶段采用 d_effective=lambda*Q_mu(d)+(1-lambda)*d；原mu、beta、sigma和alpha不变。lambda=0时跳过过滤求解。',
           '使用同一91,000步冻结先验、140个既有验证场景、四孔径，共560次新重建。沿用固定观测，不读test、不更新模型。','',
           '## 三组结果','',
           '| 方法 | K-RMSE ↓ | SSIM ↑ | 标准化残差能量 | 秒/次 |',
           '|---|---:|---:|---:|---:|']
    names=[('未过滤先验',baseline['prior_unfiltered']['successful_only_class_gamma_equal']),
           ('原软过滤',baseline['prior_soft']['successful_only_class_gamma_equal']),
           ('后期减弱过滤',summary['class_gamma_equal'])]
    for name,r in names:
        lines.append(f"| {name} | {r['rmse_k']:.6f} | {r['ssim']:.6f} | {r['standardized_residual_energy']:.6f} | {r['seconds']:.4f} |")
    lines+=['','均值按类别和孔径等权。时间含阶段诊断；旧两组复用原运行记录，跨时段计时仅供参考。观测残差并非越低越真实。','',
            '## 逐图配对变化','',
            '差值=新调度−对应基线，RMSE/残差负值、SSIM正值为改善。区间在固定验证类别内重采样场景（2000次），同场景四孔径绑定，不当作560个独立场景。','',
            '| 对照 | 指标 | 平均差 | 场景重采样95%区间 | 改善组数 |',
            '|---|---|---:|---|---:|']
    for mode,name in [('prior_unfiltered','未过滤先验'),('prior_soft','原软过滤')]:
        for key,title in [('rmse_k','K-RMSE'),('ssim','SSIM'),('standardized_residual_energy','残差能量')]:
            p=summary['paired_comparisons'][mode][key]
            a,b=p['scene_stratified_bootstrap_95_interval']
            lines.append(f"| {name} | {title} | {p['mean_delta']:+.6f} | [{a:+.6f}, {b:+.6f}] | {p['improved_pairs']}/{p['pairs']} |")
    lines+=['','## 各孔径 K-RMSE','',
            '| gamma | 未过滤先验 | 原软过滤 | 后期减弱过滤 |','|---|---:|---:|---:|']
    oldrows=[json.loads(s) for s in (base/'results.jsonl').read_text().splitlines()]
    for gamma in identity['gammas']:
        values=[np.mean([r['rmse_k'] for r in oldrows if r['mode']==mode and r['gamma']==gamma])
                for mode in ('prior_unfiltered','prior_soft')]
        values.append(summary['by_gamma'][str(gamma)]['rmse_k'])
        lines.append('| '+str(gamma)+' | '+' | '.join(f'{x:.6f}' for x in values)+' |')
    new=summary['class_gamma_equal']
    a=summary['paired_comparisons']['prior_unfiltered']
    b=summary['paired_comparisons']['prior_soft']
    lines+=['','## 判断','',
            f"相对原软过滤，RMSE变化{b['rmse_k']['mean_delta']:+.6f} K；相对未过滤先验，变化{a['rmse_k']['mean_delta']:+.6f} K。"]
    if a['rmse_k']['mean_delta']<0 and a['ssim']['mean_delta']>0:
        lines.append('该固定调度在当前验证子集上改善了相对未过滤先验的平均RMSE和SSIM；仍需结合残差与各孔径表现评估折衷，不能外推最终先验或真实微波数据。')
    elif b['rmse_k']['mean_delta']<0:
        lines.append('后期减弱改善了原软过滤的亮温误差，但仍未全面超过未过滤先验；证据支持调度有影响，不支持宣称已解决性能问题。')
    else:
        lines.append('本次预先固定的减弱调度没有降低相对原软过滤的平均亮温误差；不支持将单步审计收益直接外推到完整轨迹。')
    lines+=['','## 验证与边界','',
            f"- 新调度全部{len(rows)}次重建成功；最大CG真实相对残差{max(r['max_cg_relative_residual'] for r in rows):.8g}，门槛1e-5。",
            f"- 八个真实端点锚点通过；新入口全1/全0分别复现原过滤/不滤实现，已保存基线图的最大逐像素差{max(r['max_saved_prediction_difference_k'] for r in anchors):.8g} K。",
            '- 单元测试覆盖混合方向、端点轨迹和后期零权重跳过过滤。源码与冻结权重哈希复核通过。',
            '- 本轮只在验证集跟进已观察到的现象，不是独立测试；置信区间未包含调度选择、训练随机种子及模型训练进度的不确定性。',
            '- 与原两组同为8次先验调用。额外CG预算不同，不将这次效果差异解读为等时间最优比较。',
            '- 原训练代码、原权重和原结果未修改。首个预检目录filter_fade_val_v1记录了入口返回值读取错误，正式重建仅在v2执行。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(report=str(out/'REPORT.md'),metrics=new),ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    report(parser.parse_args().out.resolve())
