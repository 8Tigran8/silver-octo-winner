"""Rebuild every inherited stage with complete outer-fold isolation.

The selected neural candidate is frozen before this audit. No new candidate,
threshold, seed or reserve selection is performed here.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys
import time

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cloudpickle
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from mtah import expanded_team as expanded, possession_context as context, stack, features2, degrade
from mtah import temporal, v4
from mtah.methods import boost, boost_train
import temporal_trial as trial
import evaluate_temporal as prior
from quality_audit import sha, read_feature, model_path, guard_competition_data
from legacy_skillcorner_reader import period_spans
from decoder_development import metric

OUT = ROOT / 'reports/strict_fullfold'
DOC = ROOT / 'docs/experiments/strict-fullfold'
PLAN = DOC / 'plan.json'
AUDIT = trial.AUDIT
HEAD = ROOT / 'reports/v4_search'


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + '\n')


def original_names():
    return sorted(g.split('/')[1] for g in trial.plan()['training_games'] if g.startswith('original/'))


def fold_games(fold):
    return trial.game_split(f'fold{fold}')


def split_first(excluded):
    excluded = tuple(sorted(excluded))
    old = model_path(AUDIT, excluded)
    return old if old.with_suffix('.json').exists() else OUT / 'stage1' / ('exclude__' + '__'.join(excluded))


def load_first(excluded):
    dest = split_first(excluded)
    meta = json.loads(dest.with_suffix('.json').read_text())
    if set(meta['train']) != set(original_names()) - set(excluded) or set(meta['excluded']) != set(excluded):
        raise ValueError('First-stage exclusion mismatch')
    if sha(dest.with_suffix('.pkl')) != meta['model_sha256']:
        raise ValueError('First-stage hash mismatch')
    with dest.with_suffix('.pkl').open('rb') as stream:
        return cloudpickle.load(stream), meta


def register():
    if PLAN.exists():
        raise ValueError('Do not rewrite the registered audit')
    inherited = json.loads((HEAD/'models/all_nineteen.json').read_text())['train']
    if inherited != expanded.games():
        raise ValueError('Inherited training registry changed')
    assets = {}
    for fold in range(5):
        for family, folder in [('control', ROOT/'reports/temporal/models'/f'fold{fold}'),
                               ('candidate', ROOT/'reports/source_balance/models'/f'seed20261113_fold{fold}')]:
            meta = json.loads((folder/'meta.json').read_text())
            train, excluded = fold_games(fold)
            if meta['train'] != train or meta['excluded'] != excluded or sha(folder/'model.pt') != meta['model_sha256']:
                raise ValueError('Neural checkpoint fold/hash mismatch')
            assets[f'{family}_fold{fold}'] = {'model_sha256': meta['model_sha256'], 'meta_sha256': sha(folder/'meta.json')}
    write(PLAN, {
        'registered_utc': datetime.now(timezone.utc).isoformat(),
        'source_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'purpose': 'Close inherited whole-fold isolation gap without selecting another model.',
        'candidate': 'source_seed20261113',
        'control': 'v4 architecture reconstructed under the same complete-fold exclusions',
        'historical_control': 'Frozen v4 aggregate scores remain immutable; report separately, not as strict-paired control.',
        'fold_plan_sha256': sha(trial.PLAN), 'policy_sha256': sha(ROOT/'docs/evaluation/policy.json'),
        'inherited_training_games': inherited, 'neural_assets': assets,
        'first_stage': 'Same nine original-game registry/model/augmentation. Exclude every original game in outer fold. Reuse exact matching audited checkpoint only after train/model hash checks.',
        'stack': 'Within each outer train set, fit from inner game-out predictions. Every inner first-stage fit also excludes the entire outer fold. Same stack parameters, no tuning.',
        'team': 'Same inherited nineteen-game registry, features/scenarios/stride/HGB; remove every outer-fold game. No new training games.',
        'features': 'Verify cached training/evaluation period spans equal independently observed geometry/metadata spans. Predictors never receive reference labels. Full25Hz features; unchanged fivefold neural checkpoints/normalization.',
        'scenarios': 'Unchanged complete29-game/67-cell grid. Decode within independent full periods, same cost9/durations.',
        'decisions': 'Apply original evaluation-1.0 numeric/provider guards both against immutable historical v4 and against reconstructed strict control. No threshold relaxation. No candidate/seed/ensemble switching after audit.',
        'scope': 'Retrospective research audit only; no untouched final test. PFF final remains unopened.',
        'organizer_data': 'Blocked from audit. No competition input, GT, model fitting or score.',
        'sending': 'No submission/publication unless complete quality/release requirements are actually met; no outbound message in this audit.',
    })
    print('Audit registered:', PLAN, flush=True)


def attest_periods():
    rows = []
    protocol = json.loads((AUDIT/'protocol.json').read_text())
    for name in ['mtah/features2.py','mtah/features.py','mtah/degrade.py','mtah/stack.py','mtah/methods/boost.py']:
        if sha(ROOT/name) != protocol['source_sha256'][name]:
            raise ValueError('Inherited source differs: '+name)
    for game in trial.plan()['training_games']:
        family, name = game.split('/')
        if family == 'original':
            file = ROOT/'external/conv'/(name+'.npz')
            if sha(file) != protocol['converted_inputs_sha256'][name]:
                raise ValueError('Original converted input changed')
            with np.load(file) as z:
                geometry = {k:z[k] for k in ['ball','players']}
                spans = trial.observed_periods(geometry)
                label_spans = trial.metrics.valid_runs(z['gt'])
        else:
            # Source metadata defines periods; possession phases do not.
            cache = ROOT/'reports/v2_search/skillcorner'/(name+'.npz')
            with np.load(cache) as z:
                spans = period_spans(z['period'])
                label_spans = trial.metrics.valid_runs(z['gt'])
        cached = json.loads((trial.folder(game, trial.scenarios(game)[0])/'meta.json').read_text())['full_segments']
        if [list(s) for s in spans] != cached or spans != label_spans:
            raise ValueError('Cached feature spans differ from independent periods: '+game)
        rows.append({'game':game,'independent_periods':[list(s) for s in spans], 'cached_spans_equal':True})
    write(OUT/'period_attestation.json', {'plan_sha256':sha(PLAN),'status':'PASS','rows':rows})
    print('Independent period attestation PASS:29games', flush=True)


def fit_first(excluded):
    guard_competition_data()
    excluded = tuple(sorted(excluded))
    dest = split_first(excluded)
    if dest.with_suffix('.json').exists():
        _, meta = load_first(excluded)
        return {'excluded':excluded,'cached':True,'sha256':meta['model_sha256']}
    started = time.perf_counter()
    train = [n for n in original_names() if n not in excluded]
    with threadpool_limits(1):
        rows = []
        for name in train:
            f = read_feature(AUDIT,name)
            rows.append(boost.training_rows(f['X'],f['Xs'],f['gt']))
        del f
        model = boost.fit(rows)
        del rows
    dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.with_suffix('.pkl').open('wb') as stream:
        cloudpickle.dump(model,stream)
    meta = {'train':train,'excluded':list(excluded),'plan_sha256':sha(PLAN),
            'model_sha256':sha(dest.with_suffix('.pkl')),'seconds':time.perf_counter()-started}
    write(dest.with_suffix('.json'),meta)
    return {'excluded':excluded,'cached':False,'seconds':round(meta['seconds'],1)}


def fit_inherited(fold):
    guard_competition_data()
    dest = OUT/'folds'/f'fold{fold}'
    if (dest/'meta.json').exists():
        meta = json.loads((dest/'meta.json').read_text())
        if meta['plan_sha256'] != sha(PLAN) or sha(dest/'models.pkl') != meta['model_sha256']:
            raise ValueError('Stale inherited fold checkpoint')
        return fold, 'cached'
    train, test = fold_games(fold)
    excluded = [g.split('/')[1] for g in test if g.startswith('original/')]
    original_train = [n for n in original_names() if n not in excluded]
    inherited_train = [g for g in expanded.games() if g not in test]
    started = time.perf_counter()
    with threadpool_limits(1):
        rows = []; ancestry = []
        for name in original_train:
            inner_excluded = tuple(sorted(excluded+[name]))
            model, meta = load_first(inner_excluded)
            f = read_feature(AUDIT,name)
            pd, ph = boost.predict(model,f['X'],f['Xs'])
            rows.append(stack.rows(pd,ph,f['gt']))
            ancestry.append({'inner_test':'original/'+name,'train':['original/'+n for n in meta['train']],
                             'model_sha256':meta['model_sha256']})
        stage2 = stack.fit(rows)
        del rows, f, model
        parts = []; input_hashes = {}
        for game in inherited_train:
            path = HEAD/'rows'/(trial.stem(game)+'.npz')
            metadata = json.loads(path.with_suffix('.json').read_text())
            if sha(path) != metadata['rows_sha256']:
                raise ValueError('Inherited team rows changed')
            input_hashes[game] = metadata['rows_sha256']
            with np.load(path) as z:
                parts.append((z['X'],z['y']))
        team = context.fit(parts)
        del parts
        stage1, base = load_first(excluded)
    for item in ancestry:
        if set(item['train']) & set(test) or item['inner_test'] in item['train']:
            raise ValueError('Nested outer/inner leak')
    if set(inherited_train) & set(test):
        raise ValueError('Team outer-fold leak')
    dest.mkdir(parents=True,exist_ok=True)
    with (dest/'models.pkl').open('wb') as stream:
        cloudpickle.dump({'stage1':stage1,'stage2':stage2,'team':team},stream)
    write(dest/'meta.json', {
        'plan_sha256':sha(PLAN),'fold_plan_sha256':sha(trial.PLAN),'excluded':test,
        'stage1_train':['original/'+n for n in base['train']], 'stage1_sha256':base['model_sha256'],
        'stack_train':['original/'+n for n in original_train], 'inner_provenance':ancestry,
        'team_train':inherited_train,'team_rows_sha256':input_hashes,
        'model_sha256':sha(dest/'models.pkl'),'seconds':time.perf_counter()-started,
    })
    return fold, round(time.perf_counter()-started,1)


def training(jobs):
    if not (OUT/'period_attestation.json').exists():
        attest_periods()
    exclusions = set()
    for fold in range(5):
        _, test = fold_games(fold)
        outer = tuple(sorted(g.split('/')[1] for g in test if g.startswith('original/')))
        exclusions.add(outer)
        for name in original_names():
            if name not in outer:
                exclusions.add(tuple(sorted(outer+(name,))))
    with ProcessPoolExecutor(max_workers=jobs,mp_context=get_context('spawn')) as pool:
        for future in as_completed([pool.submit(fit_first,e) for e in sorted(exclusions)]):
            print('first',future.result(),flush=True)
    with ProcessPoolExecutor(max_workers=jobs,mp_context=get_context('spawn')) as pool:
        for future in as_completed([pool.submit(fit_inherited,f) for f in range(5)]):
            print('inherited',future.result(),flush=True)


def game_features(game, scenario):
    family, name = game.split('/')
    if family == 'original':
        f = read_feature(AUDIT,name,scenario)
        spans = json.loads((trial.folder(game,'matched_seed0')/'meta.json').read_text())['full_segments']
        return f['X'], f['Xs'], f['cols'].tolist(), f['gt'], spans
    with np.load(ROOT/'reports/v2_search/skillcorner'/(name+'.npz')) as z:
        clean = {k:z[k] for k in z.files}
    spans = period_spans(clean['period'])
    data = clean if scenario == 'natural_detected' else degrade.degrade(clean,seed=7,params=expanded.BLIND)
    gt = data.pop('gt')
    x, cols = features2.build(data,spans)
    xs, swapped = features2.build(data,spans,swap=True)
    if cols != swapped:
        raise ValueError('Feature columns differ')
    return x,xs,cols,gt,spans


def neural_states(fold):
    """Verify the actual weights and train-only normalizers, not just a declaration."""
    plan = json.loads(PLAN.read_text())
    train, test = fold_games(fold)
    mean, scale = trial.normalization(trial.sequences(train))
    states = {}; proofs = {}
    for name, kind, folder in [('strict_v4','control',ROOT/'reports/temporal/models'/f'fold{fold}'),
                                ('strict_source','candidate',ROOT/'reports/source_balance/models'/f'seed20261113_fold{fold}')]:
        info = json.loads((folder/'meta.json').read_text())
        frozen = plan['neural_assets'][f'{kind}_fold{fold}']
        if info['train'] != train or info['excluded'] != test or sha(folder/'meta.json') != frozen['meta_sha256'] or sha(folder/'model.pt') != frozen['model_sha256']:
            raise ValueError('Neural lineage/hash differs from registered model')
        state = torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
        if not np.array_equal(mean,state['mean'].numpy()) or not np.array_equal(scale,state['scale'].numpy()):
            raise ValueError('Normalizer differs from complete-fold training-only recomputation')
        model = temporal.Temporal(); model.load_state_dict(state['state_dict']); model.eval()
        states[name] = (model,mean,scale)
        proofs[name] = dict(info,normalizer_recomputed_train_only=True)
    return states, proofs


def assert_component_isolation(proof, test):
    """Every direct and transitive fit must exclude all outer games."""
    omitted = set(test)
    if set(proof['excluded']) != omitted:
        raise ValueError('Incomplete outer exclusions')
    for stage in ['stage1_train','stack_train','team_train']:
        if set(proof[stage]) & omitted:
            raise ValueError('Outer-fold leakage in '+stage)
    for inner in proof['inner_provenance']:
        if set(inner['train']) & (omitted | {inner['inner_test']}):
            raise ValueError('Transitive nested leakage')


def probabilities(models, x, xs, cols, spans, states):
    """No reference/label input exists on the prediction path."""
    pd, ph = boost.predict(models['stage1'],x,xs)
    pd, ph = stack.predict(models['stage2'],pd,ph)
    x,_ = context.extend(x,cols,spans); xs,_ = context.extend(xs,cols,spans)
    ph = .5*ph+.5*context.predict(models['team'],x,xs)
    downspans = [((a+4)//5,(b+4)//5) for a,b in spans]
    result = {}
    for name, (model,mean,scale) in states.items():
        nd, nh = temporal.probabilities(model,x[::5],xs[::5],downspans,mean,scale,device='cpu')
        nd, nh = v4.neural_upsample(nd,nh,spans,len(x))
        result[name] = (.5*pd+.5*nd,.5*ph+.5*nh)
    return result


def score():
    guard_competition_data()
    dest = OUT/'results.json'
    if dest.exists():
        raise ValueError('Do not overwrite completed audit scores')
    params = dict(boost_train.decoder_params(),**json.loads((ROOT/'mtah/methods/v2_params.json').read_text())['decoder'])
    records = []; lineage = {}
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    with threadpool_limits(1):
        for fold in range(5):
            folder = OUT/'folds'/f'fold{fold}'
            proof = json.loads((folder/'meta.json').read_text())
            if sha(folder/'models.pkl') != proof['model_sha256'] or proof['plan_sha256'] != sha(PLAN):
                raise ValueError('Inherited artifact changed')
            with (folder/'models.pkl').open('rb') as stream:
                models = cloudpickle.load(stream)
            train, test = fold_games(fold)
            assert_component_isolation(proof,test)
            states, neural_proof = neural_states(fold)
            lineage[f'fold{fold}'] = dict(proof, train=train, plan_sha256=sha(trial.PLAN), audit_plan_sha256=sha(PLAN),
                                         neural=neural_proof, complete_transitive_outer_isolation=True)
            for game in test:
                family, name = game.split('/')
                scenarios = ['matched_seed0','label_blind_seed1','stress_seed1'] if family == 'original' else ['natural_detected','label_blind_seed7']
                for scenario in scenarios:
                    x,xs,cols,gt,spans = game_features(game,scenario)
                    predicted = probabilities(models,x,xs,cols,spans,states)
                    del x,xs
                    for candidate, (pd,ph) in predicted.items():
                        pred = boost.label(pd,ph,spans,len(gt),params)
                        records.append({'candidate':candidate,'game':game,'match':name,'family':family,'fold':fold,'scenario':scenario,**metric(gt,pred)})
                print('strict audit',game,'complete',flush=True)
    write(dest, {'plan_sha256':sha(trial.PLAN),'audit_plan_sha256':sha(PLAN),'score_source_sha256':sha(Path(__file__)),
                 'scope':'Whole-fold isolation in every learned component;29previously reused research games, not untouched confirmation.', 'records':records})
    write(OUT/'provenance.json',lineage)
    print('Complete strict audit scores:',dest,flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['register','train','score'])
    p.add_argument('--jobs',type=int,choices=[1,2],default=2)
    args = p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.mode == 'register': register()
    elif args.mode == 'train': training(args.jobs)
    else: score()


if __name__ == '__main__': main()
