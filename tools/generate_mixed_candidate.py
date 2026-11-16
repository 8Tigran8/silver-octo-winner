"""Generate the frozen mixed-source candidate without organizer GT scoring."""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Do not overwrite an existing candidate')
    os.environ['MTAH_DATA_DIR']=str(a.data_dir.resolve())
    from mtah import atd,v4,mixed_candidate
    from mtah.methods import boost,boost_train
    from mtah.config import HALVES,N_FRAMES,D,LABELS
    from mtah.submission import write
    import numpy as np
    data=atd.load();
    if len(data['frame_ids']) != N_FRAMES:raise ValueError('Tactical frame count mismatch')
    bundle,state=mixed_candidate.load_bundle(a.bundle);parts=[(first,last+1) for _,first,last in HALVES]
    pd,ph=mixed_candidate.probabilities(data,parts,v4.load(),state)
    params=dict(boost_train.decoder_params(),**json.loads((ROOT/'mtah/methods/v2_params.json').read_text())['decoder'])
    play=[(max(first,boost.KICKOFF.get(i,first)),last) for i,(first,last) in enumerate(parts)]
    codes=boost.label(pd,ph,play,N_FRAMES,params,boost.kick_frames(data,params))
    for i,(first,last) in enumerate(parts):codes[first:max(first,boost.KICKOFF.get(i,first))]=D
    a.output.parent.mkdir(parents=True,exist_ok=True);write(str(a.output),np.asarray(LABELS)[codes]);print('Frozen candidate generated; target ground truth not assessed',flush=True)


if __name__=='__main__':main()
