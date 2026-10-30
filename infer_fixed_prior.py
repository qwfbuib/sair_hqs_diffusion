"""Frozen VE/HQS mechanism comparison; not DDPM/DDNM posterior sampling."""
import common
import argparse
import json
import math
import shutil
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from train_native_hqs import (NativeData, Config, HQSUnfold, physics_bank,
                              seed_all, code_hashes, SCHEMA)

MODES = ('physics_only', 'prior_unfiltered', 'prior_soft')


def metrics(pred, truth, residual, sigma):
    # 11x11 Gaussian SSIM, sigma=1.5, valid interior, population covariance.
    pred, truth = pred.double(), truth.double()
    axis = torch.arange(-5, 6, device=pred.device, dtype=pred.dtype)
    g = torch.exp(-axis.square() / (2 * 1.5**2)); g = g / g.sum()
    kernel = (g[:, None] * g[None, :])[None, None]
    smooth = lambda a: F.conv2d(a, kernel)
    px, py = smooth(pred), smooth(truth)
    vx, vy = smooth(pred.square()) - px.square(), smooth(truth.square()) - py.square()
    cov = smooth(pred * truth) - px * py
    ssim = ((2*px*py + 1.7**2)*(2*cov + 5.1**2) /
            ((px.square()+py.square()+1.7**2)*(vx+vy+5.1**2))).mean()
    error = pred-truth
    rmse = float(error.square().mean().sqrt())
    energy = float((residual.abs().double()/sigma).square().mean())
    return dict(rmse_k=rmse, bias_k=float(error.mean()),
                psnr_db=20*math.log10(170/rmse) if rmse else None,
                psnr_is_infinite=rmse == 0, ssim=float(ssim),
                standardized_residual_energy=energy,
                residual_rms_k=math.sqrt(energy)*sigma)


def summarize(rows):
    result = {}
    keys = ('rmse_k', 'bias_k', 'ssim', 'psnr_db', 'standardized_residual_energy',
            'seconds', 'forward_calls', 'adjoint_calls')
    for mode in MODES:
        selected = [r for r in rows if r['mode'] == mode]
        good = [r for r in selected if r['status'] == 'ok']
        groups = {}
        for r in good:
            groups.setdefault((r['scene'].split('/')[0], r['gamma']), []).append(r)
        result[mode] = dict(attempted=len(selected), succeeded=len(good),
                            failed=len(selected)-len(good),
                            complete=bool(selected) and len(selected)==len(good),
                            successful_only_class_gamma_equal={k: float(np.mean([
                                np.mean([r[k] for r in group if r[k] is not None])
                                for group in groups.values()
                                if any(r[k] is not None for r in group)]))
                                for k in keys} if good else {})
    return result


def sync(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()


@torch.no_grad()
def run(args):
    if args.val_per_class not in range(1, 21):
        raise ValueError('val-per-class must be 1..20; this entrypoint never evaluates test')
    seed_all(20261023)
    out = common.new_output(args.out, [args.data, args.prior])
    # Copy once; all subsequent loads use this immutable run-local checkpoint.
    frozen = out/'prior_frozen.pt'
    before = common.sha(args.prior)
    shutil.copyfile(args.prior, frozen)
    if before != common.sha(frozen) or before != common.sha(args.prior):
        raise RuntimeError('Prior changed during snapshot; retry with a new output directory')
    ckpt = torch.load(frozen, map_location='cpu', weights_only=False)
    if ckpt.get('schema') != SCHEMA or ckpt.get('phase') != 'prior':
        raise ValueError('Expected native128 prior checkpoint')
    cfg = Config(**ckpt['identity']['model_config']).validate()
    if cfg.stages != 8 or cfg.image_size != 128:
        raise ValueError('Expected the agreed eight-stage native128 configuration')
    data = NativeData(args.data, args.val_per_class)
    if ckpt['identity']['data_core'] != data.identity:
        raise ValueError('Checkpoint and dataset identity differ')
    model = HQSUnfold(cfg, 'diffusion', 'soft').to(args.device).eval()
    model.prior.load_state_dict(ckpt['model'], strict=True)
    model.requires_grad_(False)
    gammas = (1., .75, .5, .35)
    identity = dict(method='fixed VE/HQS ablation, not original DDNM',
                    prior_sha256=before, prior_step=ckpt['step'], config=cfg.to_dict(),
                    data_core=data.identity, data=str(data.root), scenes=data.selected_val,
                    gammas=gammas, source_hashes=code_hashes(),
                    inference_sha256=common.sha(__file__), torch=torch.__version__,
                    device=args.device, cuda=torch.version.cuda,
                    device_name=torch.cuda.get_device_name() if args.device.startswith('cuda') else 'CPU',
                    seed=20261023, test_evaluated=False)
    common.write_json(out/'identity.json', identity)
    source = out/'source'; source.mkdir()
    for name in [*identity['source_hashes'], Path(__file__).name]:
        target = source/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(common.ROOT/name, target)
    # Prior-only diagnostic: fixed noise, no observation inputs to the network.
    diagnostics = []
    rng = torch.Generator(device=args.device).manual_seed(20261023)
    for scene in data.selected_val:
        truth = data.batch([scene], args.device)
        clean = cfg.normalize(truth)
        noise = torch.randn(clean.shape, device=args.device, generator=rng)
        for sigma in (.02, .1, .25, .5):
            noisy = clean+sigma*noise
            restored = cfg.denormalize(model.prior(noisy, sigma))
            if not bool(torch.isfinite(restored).all()):
                raise RuntimeError('Non-finite prior output')
            diagnostics.append(dict(scene=scene, sigma=sigma,
                before_rmse_k=float((sigma*noise*cfg.scale).square().mean().sqrt()),
                after_rmse_k=float((restored-truth).square().mean().sqrt()),
                bias_k=float((restored-truth).mean())))
    common.write_json(out/'prior_diagnostics.json', diagnostics)
    print('Prior diagnostics complete', flush=True)
    bank = physics_bank(data, cfg, gammas, args.device)
    numerical = []
    for gamma, physics in bank.items():
        x = torch.randn(1,1,128,128,device=args.device)
        r = torch.complex(torch.randn(1,physics.operator.measurements,device=args.device),
                          torch.randn(1,physics.operator.measurements,device=args.device))
        lhs = (physics.operator(x).conj()*r).real.sum()
        rhs = (x*physics.operator.adjoint(r)).sum()
        err = float((lhs-rhs).abs()/torch.maximum(lhs.abs(), rhs.abs()).clamp_min(1e-6))
        truth,y,_ = data.batch([data.selected_val[0]],args.device,gamma)
        # Explicitly checks the nonzero 235 K offset and the 85 K scaling.
        a = physics(cfg.normalize(truth))-physics.encode(y)
        b = (physics.operator(truth)-y)/physics.sigma/physics.spectral_scale
        # Subtracting large offset visibilities loses float32 digits; measure
        # error against the original signal, not a near-zero residual element.
        norm_error=float(torch.linalg.vector_norm(a-b)/torch.maximum(
            torch.linalg.vector_norm(physics(cfg.normalize(truth))),
            torch.linalg.vector_norm(physics.encode(y))).clamp_min(1.0))
        if norm_error > 1e-5:
            raise RuntimeError(f'Normalization check failed: {norm_error}')
        if err > 1e-4: raise RuntimeError('Real adjoint check failed')
        numerical.append(dict(gamma=gamma,adjoint_relative_error=err,
                              normalization_max_error=float((a-b).abs().max()),
                              normalization_relative_signal_error=norm_error))
        # Count actual calls, including trace diagnostics; evaluation metrics excluded.
        physics.calls = dict(forward_calls=0,adjoint_calls=0)
        for method,key in [('forward','forward_calls'),('adjoint','adjoint_calls')]:
            original = getattr(physics.operator,method)
            def counted(*a,_original=original,_key=key,_p=physics,**kw):
                _p.calls[_key] += 1
                return _original(*a,**kw)
            setattr(physics.operator,method,counted)
    common.write_json(out/'numerical_checks.json', numerical)
    rows=[]
    total=len(data.selected_val)*len(gammas)*len(MODES)
    with (out/'results.jsonl').open('w',encoding='utf-8') as log:
        for scene in data.selected_val:
            for gamma in gammas:
                truth,y,sigma=data.batch([scene],args.device,gamma)
                physics=bank[gamma]
                for mode in MODES:
                    physics.cg_records=[]
                    physics.calls=dict(forward_calls=0,adjoint_calls=0)
                    row=dict(scene=scene,gamma=gamma,mode=mode)
                    sync(args.device); start=time.perf_counter()
                    try:
                        z,trace=model(y,physics,projection='none' if mode=='prior_unfiltered' else 'soft',
                                      disable_prior=mode=='physics_only',trace=True)
                        sync(args.device)
                        row.update(seconds=time.perf_counter()-start,**physics.calls)
                        pred=cfg.denormalize(z)
                        if not bool(torch.isfinite(pred).all()): raise RuntimeError('Non-finite output')
                        row.update(metrics(pred,truth,physics.operator(pred)-y,sigma))
                        row.update(status='ok',trace=trace,
                                   max_cg_relative_residual=max(physics.cg_records),
                                   prior_calls=0 if mode=='physics_only' else cfg.stages)
                        path=out/'predictions'/mode/str(gamma)/(scene+'.npy')
                        path.parent.mkdir(parents=True,exist_ok=True)
                        np.save(path,pred.cpu().numpy()[0,0],allow_pickle=False)
                    except RuntimeError as exc:
                        row.update(status='failed',error=str(exc),seconds=time.perf_counter()-start,
                                   **physics.calls)
                    rows.append(row)
                    log.write(json.dumps(row,allow_nan=False)+'\n'); log.flush()
                    common.write_json(out/'status.json',dict(completed=len(rows),total=total))
                    if len(rows)%12==0: print(f'{len(rows)}/{total}',flush=True)
    summary=summarize(rows)
    common.write_json(out/'summary.json',summary)
    common.write_json(out/'status.json',dict(status='complete',completed=len(rows),total=total,
                                            failed=sum(r['status']!='ok' for r in rows)))
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',required=True)
    parser.add_argument('--prior',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--val-per-class',type=int,default=20)
    run(parser.parse_args())
