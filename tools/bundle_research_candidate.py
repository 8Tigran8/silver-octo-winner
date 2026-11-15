"""Bind selected full public-data fits to a local, hash-verified inference bundle."""
from pathlib import Path
import json,hashlib,os
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    selection_path=ROOT/'reports/reliability/selection.json';s=json.loads(selection_path.read_text())
    if not s.get('candidate'):raise ValueError('No selected research candidate')
    out=ROOT/'reports/reliability/full_bundle.json'
    if out.exists():raise ValueError('Bundle already frozen')
    base=ROOT/'reports'/('reliability' if s['family']=='long' else 'spell_trial')
    allowed=json.loads((ROOT/'assets/v4/manifest.json').read_text())['model_training_games'];models=[]
    for seed in s['seeds']:
        folder=base/'models'/f'seed{seed}_foldfull';meta=json.loads((folder/'meta.json').read_text());file=folder/'model.pt'
        if set(meta['train'])!=set(allowed) or meta['excluded'] or meta['model_sha256']!=sha(file):raise ValueError('Full fit membership or checkpoint mismatch')
        models.append({'seed':seed,'path':os.path.relpath(file,out.parent),'sha256':sha(file),'metadata_path':os.path.relpath(folder/'meta.json',out.parent),'metadata_sha256':sha(folder/'meta.json')})
    bundle={'candidate':s['candidate'],'family':s['family'],'rule':s['rule'],'models':models,'training_games':allowed,'selection_sha256':sha(selection_path),'full_fit_only_public_external_games':True,'release_status':'RESEARCH CANDIDATE ONLY; independent confirmation/inherited-lineage audit pending; current v4 package unchanged','user_hold':'No submission/messages/push/publication'}
    out.write_text(json.dumps(bundle,indent=2)+'\n');print('Frozen local bundle:',s['candidate'],len(models),'models; no competition fitting')


if __name__=='__main__':main()
