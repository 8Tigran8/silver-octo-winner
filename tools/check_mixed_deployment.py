"""Exact deployment/scoring equivalence on previously used PFFdev3821, no GT."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from mtah import mixed_candidate,v4,degrade,expanded_team
from mtah.methods import boost
from pff_development import geometry,spans
from pff_mixed_score import decoder
from pff_reader import sha


def main():
    dest=ROOT/'reports/pff_mixed/deployment_equivalence.json'
    if dest.exists():raise ValueError('Preserve completed deployment-equivalence check')
    bundle_path=ROOT/'assets/pff-mixed-candidate/bundle.json';bundle,state=mixed_candidate.load_bundle(bundle_path);assets=v4.load()
    data,_=geometry('3821');parts=spans(data['period']);dummy=np.where(data['period']>0,0,3).astype(np.int8)
    noisy=degrade.degrade(dict(data,gt=dummy),seed=7,params=expanded_team.BLIND);noisy.pop('gt');params=decoder();differences={}
    original=ROOT/'reports/pff_mixed/pffdev_3821_predictions.npz'
    with np.load(original,allow_pickle=False) as z:
        for scenario,observed in [('natural_detected',data),('label_blind_seed7',noisy)]:
            control,candidate=mixed_candidate.paired_probabilities(observed,parts,assets,state)
            for name,probs in [('v4',control),(bundle['candidate'],candidate)]:
                pred=boost.label(*probs,parts,len(dummy),params);differences[scenario+'/'+name]=int(np.sum(pred!=z[scenario+'__'+name]))
    if any(differences.values()):raise ValueError('Deployment differs from the scored frozen model')
    dest.write_text(json.dumps({'status':'PASS','bundle_sha256':sha(bundle_path),'scope':'Previously used dev3821; both sensor cases; no reference labels or organizer score','different_frame_labels':differences,'original_predictions_sha256':sha(original),'inference_source_sha256':sha(ROOT/'mtah/mixed_candidate.py')},indent=2)+'\n');print('Exact deployment/scoring labels PASS',differences,flush=True)


if __name__=='__main__':main()
