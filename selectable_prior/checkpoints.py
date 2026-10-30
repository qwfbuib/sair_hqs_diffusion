from pathlib import Path
import torch
import common
from train_native_hqs import Config, code_hashes, SCHEMA as LEGACY_SCHEMA
from .models import architecture, SelectablePrior, SelectableHQS

SCHEMA='native128-selectable-prior-v1'


def source_hashes():
    result=code_hashes()
    paths=list((common.ROOT/'selectable_prior').rglob('*.py'))
    paths += [common.ROOT/'selectable_prior/provenance.json',
              common.ROOT/'selectable_prior/reference/VMAMBA_LICENSE.txt']
    paths += [common.ROOT/'train_selectable_prior.py', common.ROOT/'evaluate_selectable_prior.py',
              common.ROOT/'infer_fixed_prior.py']
    for p in paths:
        result[str(p.relative_to(common.ROOT))]=common.sha(p)
    return result


def load_model(state, expected_backbone=None, expected_phase=None, data_core=None,
               device='cpu', allow_legacy=True):
    phase=state.get('phase')
    if phase not in ('prior','unfold') or (expected_phase and expected_phase!=phase):
        raise ValueError('Checkpoint phase differs')
    identity=state['identity']
    if state.get('schema')==SCHEMA:
        arch=identity['architecture']
        backbone=arch['backbone']
        if arch!=architecture(backbone):
            raise ValueError('Architecture version or backend differs')
    elif state.get('schema')==LEGACY_SCHEMA and allow_legacy:
        backbone='unet';arch=architecture(backbone)
    else:
        raise ValueError('Unsupported checkpoint schema; legacy resume is not allowed')
    if expected_backbone and expected_backbone!=backbone:
        raise ValueError('Cross-backbone checkpoint loading is not allowed')
    if data_core is not None and identity['data_core']!=data_core:
        raise ValueError('Checkpoint data identity differs')
    cfg=Config(**identity['model_config']).validate()
    projection=identity.get('training',{}).get('projection','soft')
    model=(SelectablePrior(cfg,backbone) if phase=='prior' else
           SelectableHQS(cfg,backbone,projection))
    model.load_state_dict(state['model'],strict=True)
    return model.to(device),cfg,arch


def checked_snapshot(source, target):
    import shutil
    source,target=Path(source),Path(target)
    before=common.sha(source)
    if target.exists():
        if common.sha(target)!=before: raise ValueError('Frozen input checkpoint differs')
    else:
        shutil.copyfile(source,target)
    if common.sha(target)!=before or common.sha(source)!=before:
        raise ValueError('Checkpoint changed during snapshot')
    return before
