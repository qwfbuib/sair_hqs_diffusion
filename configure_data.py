"""Set only the server data root; leave model, split and training budgets intact."""
import argparse
import json
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--data',required=True);p.add_argument('--config',default='configs/server_128.json')
a=p.parse_args();root=Path(a.data).expanduser().resolve()
meta=json.loads((root/'scenes.json').read_text(encoding='utf-8'))
if meta.get('n_grid')!=128 or meta.get('status')!='complete':raise ValueError('Complete native128 dataset required')
path=Path(a.config);cfg=json.loads(path.read_text(encoding='utf-8'));cfg['data']=str(root)
path.write_text(json.dumps(cfg,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print('Configured native128 data:',root)
