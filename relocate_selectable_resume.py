"""Relocate only data_root in an existing last.pt; training must be stopped."""
import argparse
import copy
import json
import os
from pathlib import Path
from datetime import datetime, timezone


def relocation_identity(saved, expected):
    candidate = copy.deepcopy(saved)
    candidate['data_root'] = expected['data_root']
    if candidate != expected:
        raise ValueError('Other identity fields differ. Refusing data-path-only relocation.')
    return candidate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    p.add_argument('--phase', choices=['prior', 'unfold'], default='unfold')
    p.add_argument('--apply', action='store_true', help='Back up and update last.pt after checks')
    args = p.parse_args()
    import torch
    import common
    from train_native_hqs import Config, NativeData, atomic_json
    from selectable_prior.models import architecture
    from selectable_prior.checkpoints import SCHEMA, source_hashes
    from selectable_prior.training import backbone_from

    out = Path(args.out).resolve()
    config = json.loads((out/'run_config.json').read_text(encoding='utf-8'))
    cfg = Config(**config['model']).validate()
    data = NativeData(config['data'], config['training']['val_scenes_per_class'])
    checkpoint = out/args.phase/'last.pt'
    original_sha = common.sha(checkpoint)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if state.get('schema') != SCHEMA or state.get('phase') != args.phase:
        raise ValueError('Unsupported checkpoint schema or phase')
    prior = out/'unfold/input_prior.pt'
    if args.phase == 'unfold' and not prior.is_file():
        raise ValueError('Missing frozen input_prior.pt; refusing to guess the prior')
    expected = dict(model_config=cfg.to_dict(), training=config['training'],
                    data_root=str(data.root), data_core=data.identity, phase=args.phase,
                    architecture=architecture(backbone_from(config)),
                    prior_sha256=common.sha(prior) if args.phase=='unfold' else None,
                    code=source_hashes(), noise_seed=cfg.seed+314159,
                    noise_policy='private generator; AMP retries replay same samples and noise')
    old_identity = state['identity']
    updated = relocation_identity(old_identity, expected)
    print('Saved step:', state['step'])
    print('Old data_root:', old_identity['data_root'])
    print('New data_root:', updated['data_root'])
    print('All other identity fields, including data_core and code: MATCH')
    if old_identity == updated:
        print('Already matches. No changes made.')
        return
    if not args.apply:
        print('Preview only. Stop training, then add --apply to back up and update last.pt.')
        return
    backup = checkpoint.with_name('last.before_data_relocation.pt')
    report = checkpoint.with_name('data_relocation.json')
    if backup.exists() or report.exists():
        raise FileExistsError('Relocation backup/report already exists. Refusing to overwrite.')
    # Exclusive backup creation preserves the original checkpoint byte for byte.
    with checkpoint.open('rb') as src, backup.open('xb') as dst:
        import shutil
        shutil.copyfileobj(src, dst)
    if common.sha(backup) != original_sha or common.sha(checkpoint) != original_sha:
        raise ValueError('Checkpoint changed during backup. Stop concurrent training.')
    temp = checkpoint.with_name(f'last.relocate.{os.getpid()}.tmp')
    state['identity'] = updated
    try:
        torch.save(state, temp)
        check = torch.load(temp, map_location='cpu', weights_only=False)
        if check['identity'] != expected or check['step'] != state['step']:
            raise ValueError('Serialized checkpoint verification failed')
        if not all(torch.equal(v, check['model'][k]) for k, v in state['model'].items()):
            raise ValueError('Serialized model differs')
        atomic_json(report, dict(utc=datetime.now(timezone.utc).isoformat(), step=state['step'],
                                changed_field='identity.data_root', old=old_identity['data_root'],
                                new=updated['data_root'], backup=str(backup),
                                before_sha256=original_sha, after_sha256=common.sha(temp)))
        if common.sha(checkpoint) != original_sha:
            raise ValueError('Checkpoint changed before commit. Stop concurrent training.')
        os.replace(temp, checkpoint)
    finally:
        temp.unlink(missing_ok=True)
    print('Updated last.pt. Original backup:', backup)
    print('Model, optimizer, scaler, step, cursor and RNG states were retained.')
    print('Resume with your original training command. Training itself has not started.')


if __name__ == '__main__':
    main()
