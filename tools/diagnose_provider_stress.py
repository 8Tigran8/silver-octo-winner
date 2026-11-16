"""Descriptive one-factor probes after the registered combined stress pilot.

No weights or inference policies are changed. Uses the same outer exclusions.
"""
import json
import time
import gc
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import evaluate_provider_stress as audit

ROOT = audit.ROOT
OUT = ROOT / 'reports/provider_stress_decomposition'
PLAN = ROOT / 'docs/experiments/provider-stress/decomposition_plan.json'


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT / 'results.json').exists():
        raise ValueError('Do not overwrite completed diagnostic')
    plan = json.loads(PLAN.read_text())
    config = json.loads((ROOT / 'tools/provider_stress_presets.json').read_text())
    before = audit.frozen_attestation()
    records, exports, provenance = [], [], {}
    audit.torch.set_num_threads(1)
    audit.torch.use_deterministic_algorithms(True)
    with threadpool_limits(1):
        for source_name, cases in plan['cases'].items():
            source = config['sources'][source_name]
            for item in source['raw_files'].values():
                if audit.sha(ROOT / 'external' / item['path']) != item['sha256']:
                    raise ValueError('Raw source changed')
            loader = audit.stress.load_idsse if source['provider'] == 'idsse' else audit.stress.load_skillcorner
            original, native = loader(source, ROOT / 'external', OUT)
            game = source['fold_key']
            inherited, control, candidate, proof = audit.models_for(source['fold'], game)
            provenance[game] = proof
            original_hashes = audit.array_hashes(original)
            for name, params in cases.items():
                started = time.perf_counter()
                data, stats = audit.stress.make_copy(original, native, source['source_hz'], params, plan['seed'])
                checks = audit.stress.verify(original, data, stats, native)
                labels = audit.predict(data, inherited, control, candidate)
                for candidate_name, pred in labels.items():
                    records.append(dict(game=game, scenario=name, candidate=candidate_name,
                                        **audit.metric(original['gt'], pred)))
                exports.append(dict(game=game, scenario=name, parameters=params, checks=checks,
                                    array_sha256=audit.array_hashes(data)))
                np.savez_compressed(OUT / f'{source_name}__{name}_predictions.npz', **labels)
                if audit.array_hashes(original) != original_hashes:
                    raise ValueError('Canonical source changed')
                print(game, name, {r['candidate']:round(r['spell_f1_mean'],5) for r in records[-2:]},
                      'seconds', round(time.perf_counter()-started,2), flush=True)
                del data, labels
                gc.collect()
            del original, native, inherited, control, candidate
            gc.collect()
    (OUT / 'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    result=dict(plan_sha256=audit.sha(PLAN), runner_sha256=audit.sha(__file__),
                shared_runner_sha256=audit.sha(Path(audit.__file__)),
                scope=plan['reason'],records=records,exports=exports,
                provenance_sha256=audit.sha(OUT/'provenance.json'),
                frozen_before=before,frozen_after=audit.frozen_attestation(),
                model_changed=False,final_reserve_used=False)
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
