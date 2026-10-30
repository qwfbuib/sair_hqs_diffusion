"""Offline gallery and fixed-scale scientific figures from saved predictions."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

MODES = ('physics_only', 'prior_unfiltered', 'prior_soft')
LABELS = ('仅物理更新', '物理更新＋先验', '物理更新＋先验＋软过滤')
FONT = 'C:/Windows/Fonts/msyh.ttc'


def build(run, output):
    identity = json.loads((run/'identity.json').read_text(encoding='utf-8'))
    status = json.loads((run/'status.json').read_text(encoding='utf-8'))
    if status.get('status') != 'complete' or status.get('failed') != 0:
        raise ValueError('Expected completed, failure-free comparison')
    rows = [json.loads(s) for s in (run/'results.jsonl').read_text(encoding='utf-8').splitlines()]
    records = {(r['scene'], float(r['gamma']), r['mode']): r for r in rows}
    scenes = identity['scenes']; gammas = identity['gammas']
    expected = len(scenes)*len(gammas)*len(MODES)
    if len(records) != expected:
        raise ValueError('Missing or duplicate result rows')
    output.mkdir(parents=True, exist_ok=False)
    font_manager.fontManager.addfont(FONT)
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=FONT).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    fonts = {s: ImageFont.truetype(FONT, s) for s in (18, 20, 22, 26)}
    temp_cmap = matplotlib.colormaps['inferno']
    error_cmap = matplotlib.colormaps['magma']
    delta_cmap = matplotlib.colormaps['RdBu_r']
    catalog = []
    cached = {}
    max_rmse_difference = 0.0

    def mapped(array, low, high, cmap, size=(256,256)):
        rgba = cmap(np.clip((array-low)/(high-low), 0, 1), bytes=True)
        return Image.fromarray(rgba[:,:,:3]).resize(size, Image.Resampling.NEAREST)

    def bar(canvas, draw, x, y, low, high, cmap, caption, width=256):
        strip=np.tile(np.linspace(low,high,256),(12,1))
        canvas.paste(mapped(strip,low,high,cmap,(width,12)),(x,y))
        draw.text((x,y+15),f'{low:g}',font=fonts[18],fill='#424b5c')
        draw.text((x+width,y+15),f'{high:g}',anchor='ra',font=fonts[18],fill='#424b5c')
        draw.text((x+width//2,y+15),caption,anchor='ma',font=fonts[18],fill='#424b5c')

    for scene in scenes:
        gt_u8 = np.load(Path(identity['data'])/'gt'/(scene+'.npy'),allow_pickle=False)
        if gt_u8.shape != (128,128) or gt_u8.dtype != np.uint8:
            raise ValueError('Invalid reference image')
        gt=(150+170*gt_u8.astype(np.float64)/255).astype(np.float32)
        for gamma in gammas:
            preds=[np.load(run/'predictions'/mode/str(float(gamma))/(scene+'.npy'),allow_pickle=False)
                   for mode in MODES]
            for mode,pred in zip(MODES,preds):
                if pred.shape != gt.shape or not np.isfinite(pred).all():
                    raise ValueError('Invalid prediction')
                recomputed=float(np.sqrt(np.mean((pred.astype(float)-gt.astype(float))**2)))
                difference=abs(recomputed-records[(scene,float(gamma),mode)]['rmse_k'])
                max_rmse_difference=max(max_rmse_difference,difference)
                if difference > 1e-5:
                    raise ValueError('Prediction/GT pairing does not match recorded RMSE')
            canvas=Image.new('RGB',(1280,840),'#f5f7fb'); draw=ImageDraw.Draw(canvas)
            draw.text((28,12),f'{scene}   |   孔径 γ={gamma:g}',font=fonts[26],fill='#15243a')
            xs=(28,348,668,988)
            titles=('真值亮温图',)+LABELS
            for col,(array,title) in enumerate(zip([gt]+preds,titles)):
                x=xs[col]
                draw.text((x,58),title,font=fonts[20],fill='#15243a')
                canvas.paste(mapped(array,150,320,temp_cmap),(x,90))
                if col:
                    r=records[(scene,float(gamma),MODES[col-1])]
                    draw.text((x,352),f"RMSE {r['rmse_k']:.3f} K · SSIM {r['ssim']:.3f}",font=fonts[18],fill='#424b5c')
                else:
                    draw.text((x,352),'原生 128×128 网格',font=fonts[18],fill='#424b5c')
            bar(canvas,draw,28,388,150,320,temp_cmap,'亮温 / K',width=576)
            draw.text((668,390),'所有重建共用色标；越界颜色饱和，原数值保留。',font=fonts[18],fill='#424b5c')
            delta=preds[2]-preds[1]
            draw.text((28,454),'软过滤 − 未过滤：亮温差',font=fonts[20],fill='#15243a')
            canvas.paste(mapped(delta,-2,2,delta_cmap),(28,488))
            for col,pred in enumerate(preds,1):
                draw.text((xs[col],454),'绝对误差 |重建 − 真值|',font=fonts[20],fill='#15243a')
                canvas.paste(mapped(np.abs(pred-gt),0,30,error_cmap),(xs[col],488))
            bar(canvas,draw,28,758,-2,2,delta_cmap,'差值 / K')
            bar(canvas,draw,348,758,0,30,error_cmap,'绝对误差 / K',width=576)
            draw.text((988,758),'误差 ≥30 K 为最亮色',font=fonts[18],fill='#424b5c')
            draw.text((988,788),'差值超出 ±2 K 时饱和',font=fonts[18],fill='#424b5c')
            relative=Path('comparisons')/(scene+f'_g{gamma:g}.png')
            (output/relative).parent.mkdir(parents=True,exist_ok=True)
            canvas.save(output/relative,optimize=False)
            catalog.append(dict(scene=scene,category=scene.split('/')[0],gamma=gamma,
                                image=relative.as_posix()))
            if gamma == .5:
                cached[scene]=(gt,preds)

    # One example per class, chosen near that class's median prior RMSE at gamma=.5.
    representatives=[]
    for category in sorted({s.split('/')[0] for s in scenes}):
        group=[s for s in scenes if s.startswith(category+'/')]
        values=[records[(s,.5,'prior_unfiltered')]['rmse_k'] for s in group]
        median=float(np.median(values))
        representatives.append(min(group,key=lambda s:(abs(records[(s,.5,'prior_unfiltered')]['rmse_k']-median),s)))
    fig,axes=plt.subplots(len(representatives),4,figsize=(12,2.7*len(representatives)),layout='constrained')
    for i,scene in enumerate(representatives):
        gt,preds=cached[scene]
        for j,image in enumerate([gt]+preds):
            ax=axes[i,j]
            im=ax.imshow(image,cmap=temp_cmap,vmin=150,vmax=320,interpolation='nearest',origin='upper')
            ax.set_xticks([]);ax.set_yticks([])
            if i==0:ax.set_title(titles[j],fontsize=12,pad=9)
            if j==0:
                category=scene.split('/')[0].replace('rectangular_farmland','farmland').replace('thermal_power_station','power station')
                ax.set_ylabel(category,fontsize=9)
                ax.set_xlabel(scene.split('/')[-1],fontsize=8)
            else:
                r=records[(scene,.5,MODES[j-1])]
                ax.set_xlabel(f"RMSE {r['rmse_k']:.2f} K · SSIM {r['ssim']:.3f}",fontsize=9)
    fig.colorbar(im,ax=axes.ravel().tolist(),label='亮温 / K',shrink=.5,pad=.015)
    fig.suptitle('SAIR 重建对比 · γ=0.5 · 每类中位误差附近样本\n同一亮温色标 150–320 K；91,000 步冻结先验',fontsize=14)
    fig.savefig(output/'overview.png',dpi=150)
    plt.close(fig)
    first=next(i for i,c in enumerate(catalog) if c['scene']==representatives[0] and c['gamma']==.5)
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SAIR 重建结果浏览</title><style>
body{margin:0;background:#edf1f7;color:#16243b;font:16px "Microsoft YaHei",sans-serif}main{max-width:1280px;margin:auto;padding:24px}h1{font-size:26px;margin:0 0 10px}p{line-height:1.7}select,button{font:inherit;padding:9px;border:1px solid #ccd4df;border-radius:7px;background:white;color:inherit}nav{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:18px 0}.panel{background:white;padding:16px;border-radius:12px}img{width:100%;height:auto;display:block}small{color:#526176}.legend{font-size:14px}a{color:#125fb2}label{display:flex;gap:6px;align-items:center}
</style><main><h1>SAIR 重建结果浏览</h1><p>140 个验证场景 · 四个孔径 · 三组重建 · 冻结先验 91,000 步。图像取自已有结果，没有重新推断。</p>
<nav><label>类别<select id="category"></select></label><label>场景<select id="scene"></select></label><label>孔径 γ<select id="gamma"></select></label><button id="prev">上一场景</button><button id="next">下一场景</button><span id="position"></span></nav>
<div class="panel"><img id="comparison" alt="真值、三组重建、绝对误差与软过滤差值对比"></div>
<p class="legend">上排：真值、仅物理更新、加入先验、加入先验与软过滤。下排：软过滤引起的亮温差、三组绝对误差。亮温统一为 150–320 K，误差统一为 0–30 K；颜色饱和仅影响显示，原数组未裁剪。数组行列方向保持原存储顺序，未转置或翻转。</p>
<p><a href="overview.png">打开各类别代表样本总览</a> · <a id="download" download>下载当前对比图</a></p>
<small>总览每类选择 γ=0.5 下“未过滤先验”RMSE 最接近该类中位数的场景。浏览页包含全部结果，不按效果筛选。灰度代理模拟亮温，尚非真实微波测温验证；输出像素数不代表已证明的空间分辨率。</small></main>
<script>const cases=__DATA__,initial=__INITIAL__;
const cat=document.querySelector('#category'),scene=document.querySelector('#scene'),gamma=document.querySelector('#gamma');
const unique=a=>[...new Set(a)];function fill(el,items){el.replaceChildren(...items.map(v=>{const o=document.createElement('option');o.value=v;o.textContent=v;return o}))}
fill(cat,unique(cases.map(c=>c.category)));fill(gamma,[1,.75,.5,.35]);
function sceneList(){fill(scene,unique(cases.filter(c=>c.category===cat.value).map(c=>c.scene)))}
function render(){const item=cases.find(c=>c.scene===scene.value&&c.gamma===Number(gamma.value));document.querySelector('#comparison').src=item.image;document.querySelector('#download').href=item.image;document.querySelector('#position').textContent=`${scene.selectedIndex+1} / ${scene.options.length}`}
cat.onchange=()=>{sceneList();render()};scene.onchange=gamma.onchange=render;
function move(d){scene.selectedIndex=(scene.selectedIndex+d+scene.options.length)%scene.options.length;render()}
document.querySelector('#prev').onclick=()=>move(-1);document.querySelector('#next').onclick=()=>move(1);
cat.value=cases[initial].category;sceneList();scene.value=cases[initial].scene;gamma.value=cases[initial].gamma;render();
</script></html>'''
    html=html.replace('__DATA__',json.dumps(catalog,ensure_ascii=False)).replace('__INITIAL__',str(first))
    (output/'index.html').write_text(html,encoding='utf-8')
    (output/'selection.json').write_text(json.dumps(dict(
        representatives=representatives,selection='per-class closest to median prior_unfiltered RMSE at gamma=.5',
        input_run=str(run),prior_sha256=identity['prior_sha256'],
        cases=len(catalog),prediction_pairings_verified=expected,max_rmse_difference=max_rmse_difference,
        brightness_limits_k=[150,320],error_limits_k=[0,30],delta_limits_k=[-2,2],
        orientation='original array order, origin upper; no flip or transpose'),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(output=str(output),cases=len(catalog),representatives=representatives,
                         max_rmse_difference=max_rmse_difference),ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    build(args.run.resolve(),args.out.resolve())
