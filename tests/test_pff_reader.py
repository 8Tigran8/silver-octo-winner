"""Check annotation separation and conservative PFF reference construction."""
import bz2
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from pff_reader import load_geometry,reference,tracking_reference


def event(start,sequence,home,kind='OTB',action='PA'):
    return {'startTime':start,'eventTime':start,'sequence':sequence,
            'gameEvents':{'gameEventType':kind,'period':1,'homeTeam':home,'initialNonEvent':False},
            'possessionEvents':{'possessionEventType':action,'nonEvent':False}}


class PffReaderTests(unittest.TestCase):
    def geometry_fixture(self,path,changed_annotations=False,conflict=False,gap=False):
        meta=[{'fps':25,'homeTeam':{'id':'h'},'awayTeam':{'id':'a'},'stadium':{'pitches':[{'length':100,'width':60}]}}]
        roster=[{'shirtNumber':1,'team':{'id':'h'}},{'shirtNumber':1,'team':{'id':'a'}}]
        (path/'meta.json').write_text(json.dumps(meta));(path/'roster.json').write_text(json.dumps(roster))
        rows=[]
        for i in range(26):
            if gap and i==14:continue
            row={'frameNum':i,'videoTimeMs':i*40,'period':1,
                 'homePlayers':[{'jerseyNum':1,'visibility':'VISIBLE','x':10,'y':0,'speed':1}],
                 'awayPlayers':[{'jerseyNum':1,'visibility':'ESTIMATED','x':-10,'y':0,'speed':1}],
                 'balls':[{'visibility':'VISIBLE' if i!=10 else 'ESTIMATED','x':i,'y':0,'z':0}],
                 'game_event':{'home_ball':not changed_annotations}}
            rows.append(row)
            if i==5:
                duplicate=copy.deepcopy(row);duplicate['game_event']={'game_event_type':'OUT'}
                if conflict:duplicate['balls'][0]['x']+=5
                rows.append(duplicate)
        with bz2.open(path/'tracking.bz2','wt') as f:
            for row in rows:f.write(json.dumps(row)+'\n')
        return load_geometry(path/'tracking.bz2',path/'meta.json',path/'roster.json')

    def test_annotations_cannot_affect_geometry_and_estimates_stay_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);a,info=self.geometry_fixture(path)
            b,_=self.geometry_fixture(path,changed_annotations=True)
            for key in a:np.testing.assert_array_equal(a[key],b[key])
            self.assertTrue(np.isnan(a['ball'][10]).all())
            self.assertTrue(np.isnan(a['players'][:,1]).all())
            self.assertEqual(info['duplicate_rows'],1)
            self.assertNotIn('gt',a)

    def test_conflicting_duplicate_coordinates_become_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            data,info=self.geometry_fixture(Path(tmp),conflict=True)
            self.assertTrue(np.isnan(data['ball'][5]).all())
            self.assertEqual(info['conflicting_geometry_frames'],1)

    def test_missing_tracking_frame_is_not_excluded_from_period(self):
        with tempfile.TemporaryDirectory() as tmp:
            data,_=self.geometry_fixture(Path(tmp),gap=True)
            self.assertEqual(len(data['time']),26)
            self.assertEqual(data['period'][14],1)
            self.assertTrue(np.isnan(data['ball'][14]).all())

    def test_only_explicit_out_creates_dead_and_period_resets_reference(self):
        times=np.arange(76)/25;period=np.where(times<1,1,np.where(times>=2,2,0)).astype(np.int8)
        rows=[event(0,1,True),event(.1,None,None,'SUB'),event(.2,None,None,'FOUL'),event(.5,None,None,'OUT')]
        restart=event(2.2,2,False);restart['gameEvents']['period']=2;rows.append(restart)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'events.json';p.write_text(json.dumps(rows));gt,_=reference(p,times,period)
        self.assertTrue((gt[:12]==0).all())
        self.assertTrue((gt[13:25]==2).all())
        self.assertTrue((gt[25:55]==3).all())
        self.assertTrue((gt[55:]==1).all())

    def test_conflicting_deliberate_owners_are_unknown_not_majority_voted(self):
        rows=[event(0,1,True),event(.2,1,True),event(.4,1,False)]
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'events.json';p.write_text(json.dumps(rows));gt,info=reference(p,np.arange(26)/25,np.ones(26,np.int8))
        self.assertTrue((gt==3).all());self.assertEqual(info['clear_owner_conflicts'],1)

    def test_rebound_opening_is_unknown_until_deliberate_control(self):
        rows=[event(0,1,False,action='RE'),event(.2,1,True)]
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'events.json';p.write_text(json.dumps(rows));gt,_=reference(p,np.arange(26)/25,np.ones(26,np.int8))
        self.assertTrue((gt[:5]==3).all());self.assertTrue((gt[5:]==0).all())

    def test_primary_state_machine_ignores_positions_and_preserves_side_event_state(self):
        rows=[]
        for i in range(26):
            e=({'game_event_type':'FIRSTKICKOFF','home_ball':False} if i==0 else
               {'game_event_type':'SUB','home_ball':None} if i==5 else
               {'game_event_type':'FOUL','home_ball':None} if i==7 else
               {'game_event_type':'OUT','home_ball':None} if i==10 else
               {'game_event_type':'OTB','home_ball':True} if i==20 else None)
            rows.append({'frameNum':i,'period':1,'game_event':e,'balls':[{'x':i*100,'y':-999}]})
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'tracking.bz2'
            def save():
                with bz2.open(path,'wt') as f:
                    for r in rows:f.write(json.dumps(r)+'\n')
            save();times=np.arange(26)/25;period=np.ones(26,np.int8)
            gt,shadow,_=tracking_reference(path,times,period,25)
            for row in rows:row['balls']=[{'x':-1e20,'y':1e20}]
            save();changed,_,_=tracking_reference(path,times,period,25)
        np.testing.assert_array_equal(gt,changed)
        np.testing.assert_array_equal(gt,shadow)
        self.assertTrue((gt[:10]==1).all());self.assertTrue((gt[10:20]==2).all());self.assertTrue((gt[20:]==0).all())

    def test_reference_duplicates_mask_conflict_without_majority_vote(self):
        rows=[{'frameNum':0,'period':1,'game_event':{'game_event_type':'FIRSTKICKOFF','home_ball':True}},
              {'frameNum':1,'period':1,'game_event':{'game_event_type':'OTB','home_ball':True}},
              {'frameNum':1,'period':1,'game_event':{'game_event_type':'OTB','home_ball':False}},
              {'frameNum':2,'period':1,'game_event':None}]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'tracking.bz2'
            with bz2.open(path,'wt') as f:
                for row in rows:f.write(json.dumps(row)+'\n')
            gt,shadow,info=tracking_reference(path,np.arange(3)/25,np.ones(3,np.int8),25)
        np.testing.assert_array_equal(gt,[0,3,1]);np.testing.assert_array_equal(shadow,[0,1,1])
        self.assertEqual(info['conflicting_reference_frames'],1)


if __name__=='__main__':unittest.main()
