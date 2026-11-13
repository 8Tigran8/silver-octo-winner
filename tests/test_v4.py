import json,hashlib,sys,tempfile,zipfile,io
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from mtah import v4
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from wcsf_reader import geometry,unanimous_reference


class V4Tests(unittest.TestCase):
    def test_frozen_bundle_contains_only_external_trained_models(self):
        manifest=json.loads((Path(v4.ROOT)/'assets/v4/manifest.json').read_text())
        self.assertEqual(len(manifest['model_training_games']),29);self.assertFalse(manifest['competition_data_used_for_fitting'])
        for item in manifest['files']:self.assertEqual(v4.digest(Path(v4.ROOT)/'assets/v4'/item['path']),item['sha256'])
        self.assertTrue(all(not any(s in game for s in ['BAR-ATH','wcsf','pff','metrica_3']) for game in manifest['model_training_games']))

    def test_actual_frozen_predictor_is_reference_blind(self):
        n=700;p=np.zeros((n,22,3),np.float32);p[:,:11,0]=.25;p[:,11:,0]=.75;p[:,:,1]=np.linspace(.1,.9,22)
        data={'players':p,'ball':np.tile([.5,.5,0.],(n,1)),'slot_team':np.repeat([0,1],11),'pitch':(105.,68.),'gt':np.zeros(n,np.int8)}
        assets=v4.load();a,b=v4.probabilities(data,[(0,350),(350,n)],assets);data['gt'][:]=3;c,d=v4.probabilities(data,[(0,350),(350,n)],assets)
        np.testing.assert_array_equal(a,c);np.testing.assert_array_equal(b,d);self.assertTrue(np.isfinite(a).all() and np.isfinite(b).all())

    def test_external_reference_ambiguity_never_uses_a_majority(self):
        df=pd.DataFrame({'timestamp_ms':[0,0,100,100,100,200,200],'AliveFlag':[0,0,1,1,1,1,0],'PossessionFlag':[1,2,1,1,2,1,1]})
        values,conflicts=unanimous_reference(df);self.assertEqual(values.tolist(),[2,3,3]);self.assertEqual(conflicts,2)

    def test_external_geometry_ignores_label_columns(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            def write_labels(value):
                records=[]
                for t in [0,100,200]:
                    for group in [1,2]:
                        sign=1 if group==1 else -1
                        for identity in [1,2]:records.append({'x':sign*(identity*10+t/100),'y':sign*identity,'group_id':group,'timestamp_ms':t,'x_ball':sign*(2+t/100),'y_ball':sign*3,'player_id':identity,'AliveFlag':value,'PossessionFlag':value})
                for name in ['positions_FIRST_HALF.csv.zip','positions_SECOND_HALF.csv.zip']:
                    with zipfile.ZipFile(root/name,'w') as z:z.writestr(name[:-4],pd.DataFrame(records).to_csv(index=False))
            write_labels(1);a,meta=geometry(root);write_labels(0);b,changed=geometry(root)
            self.assertEqual(meta,changed)
            for key in a:np.testing.assert_array_equal(a[key],b[key])


if __name__=='__main__':unittest.main()
