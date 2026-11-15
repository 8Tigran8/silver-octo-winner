"""Freeze only selected public-external trained weights in a portable local bundle."""
from pathlib import Path
import json,shutil,hashlib,argparse
ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--bundle',type=Path,default=ROOT/'reports/reliability/full_bundle.json');parser.add_argument('--output',type=Path,default=ROOT/'assets/research-v5');args=parser.parse_args()
    original=args.bundle;bundle=json.loads(original.read_text());dest=args.output
    if dest.exists():raise ValueError('Portable research bundle already exists')
    dest.mkdir();models=[]
    for item in bundle['models']:
        name=f"seed{item['seed']}";folder=dest/name;folder.mkdir()
        for source,target,digest in [(item['path'],'model.pt',item['sha256']),(item['metadata_path'],'meta.json',item['metadata_sha256'])]:
            file=original.parent/source
            if hashlib.sha256(file.read_bytes()).hexdigest()!=digest:raise ValueError('Frozen source hash changed')
            shutil.copy2(file,folder/target)
        models.append(dict(item,path=name+'/model.pt',metadata_path=name+'/meta.json'))
    bundle['models']=models;(dest/'bundle.json').write_text(json.dumps(bundle,indent=2)+'\n')
    shutil.copy2(original.parent/'selection.json',dest/'selection.json')
    (dest/'README.md').write_text('# Local research candidate\n\nSelected temporal weights trained only on the29public external games listed in bundle.json. No organizer input or fitted target statistics. Hashes and metadata are verified before inference.\n\nStatus: research candidate; verified v4 stays the default. Independent confirmation and inherited full-fold lineage review are pending. User hold: no submission, push, publication or outbound messages.\n')
    print('Portable local research assets frozen; v4 default unchanged')


if __name__=='__main__':main()
