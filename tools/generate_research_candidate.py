"""Generate a local-only candidate CSV without target ground-truth scoring."""
import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Preserve existing candidates; choose a new output path')
    os.environ['MTAH_DATA_DIR']=str(a.data_dir.resolve())
    from mtah import atd,v4,research_candidate
    from mtah.methods import boost,boost_train
    from mtah.config import HALVES,N_FRAMES,D,LABELS
    from mtah.submission import write
    import numpy as np
    data=atd.load();assets=v4.load();bundle,states=research_candidate.load_bundle(a.bundle)
    segs=[(first,last+1) for _,first,last in HALVES];pd,ph=research_candidate.probabilities(data,segs,assets,states,bundle['rule'])
    params=dict(boost_train.decoder_params(),**json.loads((ROOT/'mtah/methods/v2_params.json').read_text())['decoder'])
    play=[(max(first,boost.KICKOFF.get(i,first)),last) for i,(first,last) in enumerate(segs)]
    codes=boost.label(pd,ph,play,N_FRAMES,params,boost.kick_frames(data,params))
    for i,(first,last) in enumerate(segs):codes[first:max(first,boost.KICKOFF.get(i,first))]=D
    a.output.parent.mkdir(parents=True,exist_ok=True);write(str(a.output),np.asarray(LABELS)[codes]);print('Local candidate generated; target quality not assessed',flush=True)


if __name__=='__main__':main()
