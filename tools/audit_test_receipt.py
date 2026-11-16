"""Machine-readable unittest execution receipt, including failures and skips."""
import unittest,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
class Receipt(unittest.TextTestResult):
 def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.records=[]
 def record(self,t,status,detail=''):self.records.append({'test':t.id(),'purpose':t.shortDescription() or t.id().split('.')[-1].replace('_',' '),'input_and_expected':'See concrete fixture and assertions in '+t.__class__.__module__.replace('.','/')+'.py','actual':status,'detail':detail,'command':f'python -m unittest {t.id()}'})
 def addSuccess(self,t):super().addSuccess(t);self.record(t,'PASS')
 def addFailure(self,t,e):super().addFailure(t,e);self.record(t,'FAIL',self._exc_info_to_string(e,t))
 def addError(self,t,e):super().addError(t,e);self.record(t,'ERROR',self._exc_info_to_string(e,t))
 def addSkip(self,t,r):super().addSkip(t,r);self.record(t,'NOT VERIFIED',r)
suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),top_level_dir=str(ROOT));r=unittest.TextTestRunner(verbosity=1,resultclass=Receipt).run(suite);out=ROOT/'reports/independent_audit/test_receipt.json';out.write_text(json.dumps({'run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'skipped':len(r.skipped),'records':r.records},indent=2)+'\n');sys.exit(not r.wasSuccessful())
