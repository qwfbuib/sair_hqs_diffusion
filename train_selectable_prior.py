"""Parallel native128 training entrypoint. Legacy training files are unchanged."""
import common
import argparse
from selectable_prior.training import main

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    p.add_argument('--phase',choices=['pipeline','prior','unfold'],default='pipeline')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--data');p.add_argument('--prior')
    p.add_argument('--resume',action='store_true');p.add_argument('--stop-after',type=int)
    p.add_argument('--smoke',action='store_true',help='Explicit two-update-per-phase engineering profile')
    main(p.parse_args())
