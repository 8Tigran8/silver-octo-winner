"""Generate the validated recovery candidate from ATD only; never score target GT."""
import argparse,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Do not overwrite a completed candidate')
    os.environ['MTAH_DATA_DIR']=str(a.data_dir.resolve())
    import numpy as np
    from mtah import atd,recovery_candidate
    from mtah.methods import boost
    from mtah.config import HALVES,N_FRAMES,D,LABELS
    from mtah.submission import write
    data=atd.load()
    if len(data['frame_ids'])!=N_FRAMES:raise ValueError('Tactical frame count mismatch')
    bundle,state=recovery_candidate.load_bundle(a.bundle);parts=[(a,b+1) for _,a,b in HALVES]
    pd,ph=recovery_candidate.probabilities(data,parts,state)
    params=bundle['decoder'];play=[(max(a,boost.KICKOFF.get(i,a)),b) for i,(a,b) in enumerate(parts)]
    codes=boost.label(pd,ph,play,N_FRAMES,params,boost.kick_frames(data,params))
    for i,(first,last) in enumerate(parts):codes[first:max(first,boost.KICKOFF.get(i,first))]=D
    a.output.parent.mkdir(parents=True,exist_ok=True);write(str(a.output),np.asarray(LABELS)[codes])
    print('Recovery candidate generated; target ground truth not assessed',flush=True)


if __name__=='__main__':main()
