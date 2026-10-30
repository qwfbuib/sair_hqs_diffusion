"""Read-only identity diagnosis. Does not resume training or edit checkpoints."""
import argparse
import json
from pathlib import Path


def differences(saved, current, prefix='identity'):
    if isinstance(saved, dict) and isinstance(current, dict):
        result = []
        for key in sorted(saved.keys() | current.keys()):
            name = prefix + '.' + key
            if key not in saved:
                result.append((name, '<missing>', current[key]))
            elif key not in current:
                result.append((name, saved[key], '<missing>'))
            else:
                result.extend(differences(saved[key], current[key], name))
        return result
    return [] if saved == current else [(prefix, saved, current)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True, help='Existing original run directory')
    p.add_argument('--phase', choices=['prior', 'unfold'], default='unfold')
    p.add_argument('--config', help='Defaults to run_config.json in the run directory')
    p.add_argument('--data', help='Only supply if the failing command used this override')
    args = p.parse_args()
    # Keep this file beside train_selectable_prior.py, not inside the hashed package.
    import torch
    import common
    from train_native_hqs import Config, NativeData
    from selectable_prior.models import architecture
    from selectable_prior.checkpoints import SCHEMA, source_hashes
    from selectable_prior.training import backbone_from

    out = Path(args.out).resolve()
    config = json.loads(Path(args.config or out/'run_config.json').read_text(encoding='utf-8'))
    if args.data:
        config['data'] = args.data
    cfg = Config(**config['model']).validate()
    data = NativeData(config['data'], config['training']['val_scenes_per_class'])
    checkpoint = out/args.phase/'last.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    prior = out/'unfold/input_prior.pt'
    if args.phase == 'unfold' and not prior.is_file():
        prior = out/'prior/best.pt'
    expected = dict(model_config=cfg.to_dict(), training=config['training'],
                    data_root=str(data.root), data_core=data.identity, phase=args.phase,
                    architecture=architecture(backbone_from(config)),
                    prior_sha256=common.sha(prior) if args.phase=='unfold' else None,
                    code=source_hashes(), noise_seed=cfg.seed+314159,
                    noise_policy='private generator; AMP retries replay same samples and noise')
    diff = differences(state.get('identity', {}), expected)
    if state.get('schema') != SCHEMA:
        diff.insert(0, ('schema', state.get('schema'), SCHEMA))
    print('Checkpoint:', checkpoint)
    print('Saved step:', state.get('step'))
    print('Difference count:', len(diff))
    for field, saved, current in diff:
        print('\nFIELD:', field)
        print('SAVED:', json.dumps(saved, ensure_ascii=False))
        print('CURRENT:', json.dumps(current, ensure_ascii=False))
    saved_code = state.get('identity', {}).get('code', {})
    snapshot = out/'source'
    snapshot_ok = bool(saved_code) and all(
        (snapshot/name).is_file() and common.sha(snapshot/name)==h
        for name, h in saved_code.items())
    print('\nSaved source snapshot matches checkpoint:', snapshot_ok)
    if not diff:
        print('Identity matches for these arguments. Compare with the exact failing command.')
    print('Read-only check finished. No training or checkpoint changes performed.')


if __name__ == '__main__':
    main()
