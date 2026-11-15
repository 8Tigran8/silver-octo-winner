"""Guard against aggregate wins hiding reserve-group or coverage failures."""
import copy
import json
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from pff_final import final_decision


class PffFinalTests(unittest.TestCase):
    def setUp(self):
        self.rules=json.loads((ROOT/'docs/evaluation/policy.json').read_text())['reserve']['final_gate']
        self.ids=sum(self.rules['groups'].values(),[])

    def records(self,gains=None):
        gains=gains or {g:.03 for g in self.ids};rows=[]
        for g in self.ids:
            for scenario in ['natural_detected','label_blind_seed7']:
                for reference in ['canonical','sdk_shadow']:
                    for candidate in ['v4','source_seed20261113']:
                        gain=gains[g] if candidate!='v4' else 0
                        rows.append({'game':g,'scenario':scenario,'reference':reference,'candidate':candidate,
                            'spell_f1_mean':.6+gain,'macro_f1':.7,
                            'per_class':{c:{'f1':.7} for c in ['H','A','D']}})
        return rows

    def test_negative_country_group_blocks_large_aggregate_improvement(self):
        gains={g:.1 if g in self.rules['groups']['A'] else -.01 for g in self.ids}
        result=final_decision(self.records(gains),self.ids,self.rules)
        self.assertTrue(result['checks']['natural_gain'])
        self.assertFalse(result['checks']['group_H'])
        self.assertTrue(result['status'].startswith('FAIL'))

    def test_missing_or_duplicated_reserve_cell_cannot_pass(self):
        rows=self.records()
        with self.assertRaises(ValueError):final_decision(rows[:-1],self.ids,self.rules)
        rows[-1]=copy.deepcopy(rows[0])
        with self.assertRaises(ValueError):final_decision(rows,self.ids,self.rules)

    def test_reference_sensitivity_and_macro_are_independent_guards(self):
        rows=self.records()
        self.assertTrue(final_decision(rows,self.ids,self.rules)['status'].startswith('PASS'))
        for row in rows:
            if row['candidate']=='source_seed20261113' and row['reference']=='sdk_shadow':row['spell_f1_mean']=.58
        result=final_decision(rows,self.ids,self.rules)
        self.assertTrue(result['checks']['natural_gain'])
        self.assertFalse(result['checks']['sensitivity_natural_detected'])
        rows=self.records()
        for row in rows:
            if row['candidate']=='source_seed20261113':row['macro_f1']=.68
        self.assertFalse(final_decision(rows,self.ids,self.rules)['checks']['macro_guard'])


if __name__=='__main__':unittest.main()
