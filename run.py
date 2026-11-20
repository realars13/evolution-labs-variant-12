"""Run reproducible experiment series and regenerate reports and figures."""
import argparse
import json
import platform
import time
from pathlib import Path
import numpy as np
from evolution import continuous as c, routing as r, scheduling as s
from evolution.common import real_ga,random_search,save_json,save_csv

ROOT=Path(__file__).resolve().parent


def run_lab(number,config):
    cfg=config[f'lab{number}']; out=ROOT/f'lab{number}'/'results';out.mkdir(parents=True,exist_ok=True)
    if number==2:
        data=r.generate_data(config['data_seed'],cfg['customers']);save_json(ROOT/'lab2/data.json',data)
        methods=['penalty_weak_inversion','penalty_strong_inversion','penalty_strong_swap','random']
    elif number==3:
        data=s.generate_data(config['data_seed'],cfg['events'],cfg['days'],cfg['slots_per_day']);save_json(ROOT/'lab3/data.json',data)
        methods=['nsga2','weighted']
    elif number==4:methods=[f'{func}_{mode}' for func in ['qing','ellipsoid'] for mode in ['fixed','adaptive']]
    else:methods=['ga_small','ga_large','random']
    rows=[];representatives={};all_history={};step_history={}
    for method in methods:
        all_history[method]=[]
        for run in range(cfg['runs']):
            seed=config['first_seed']+run
            rng=np.random.default_rng(seed);started=time.perf_counter()
            if number==1:
                kwargs=dict(budget=cfg['budget'],dimension=cfg['dimension'],lower=cfg['lower'],upper=cfg['upper'])
                if method=='random':result=random_search(c.qing,rng,**kwargs)
                else:result=real_ga(c.qing,rng,population=cfg['population'],scale=cfg['mutation_scales'][method=='ga_large'],**kwargs)
            elif number==2:
                result=r.search(data,rng,cfg['budget'],cfg['population'],penalty=.05 if 'weak' in method else 100.,mutation='swap' if 'swap' in method else 'inversion',random=method=='random')
            elif number==3:
                result=s.search(data,rng,cfg['budget'],cfg['population'],method)
                result['hypervolume']=s.hypervolume(result['front'],np.array([len(data['conflicts'])+1,97,25]))
                result['front_size']=len(np.unique(result['front'],axis=0))
            else:
                func,mode=method.split('_')
                result=c.evolution_strategy(getattr(c,func),rng,cfg['budget'],mode,cfg['population'],cfg['offspring'],cfg['dimension'])
            elapsed=time.perf_counter()-started
            row={'method':method,'run':run,'seed':seed,'value':result['value'],'evaluations':result['evaluations'],'seconds':elapsed}
            for key in ['duration','violation','feasible','hypervolume','front_size']:
                if key in result:row[key]=result[key]
            rows.append(row);all_history[method].append(result['history'])
            score=-result['hypervolume'] if number==3 else (result.get('violation',0)>1e-9,result.get('violation',0),result['value'])
            if method not in representatives or score<representatives[method][0]:representatives[method]=(score,dict(result,seed=seed))
            if 'steps' in result:step_history.setdefault(method,[]).append(result['steps'])
        print(f'Lab {number}: {method}, {cfg["runs"]} runs completed',flush=True)
    save_csv(out/'runs.csv',rows);save_json(out/'histories.json',all_history)
    save_json(out/'best.json',{k:v[1] for k,v in representatives.items()})
    if step_history:save_json(out/'steps.json',step_history)
    summary={}
    for method in methods:
        selected=[row for row in rows if row['method']==method];values=np.array([row['value'] for row in selected])
        summary[method]={'best':float(values.min()),'mean':float(values.mean()),'median':float(np.median(values)),'std':float(values.std(ddof=1)),'worst':float(values.max()),'runs':len(values),'mean_seconds':float(np.mean([row['seconds'] for row in selected]))}
        if number in [2,3]:summary[method]['feasible_fraction']=float(np.mean([row['feasible'] for row in selected]))
        if number==3:
            for key in ['hypervolume','front_size']:
                vals=np.array([row[key] for row in selected]);summary[method][key+'_mean']=float(vals.mean());summary[method][key+'_std']=float(vals.std(ddof=1))
    save_json(out/'summary.json',summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lab',choices=['all','1','2','3','4'],default='all')
    parser.add_argument('--config',type=Path,default=ROOT/'config.json')
    parser.add_argument('--seed',type=int,help='Override the first experiment seed')
    parser.add_argument('--runs',type=int,help='Override series length for a quick check')
    parser.add_argument('--plots-only',action='store_true')
    args=parser.parse_args();config=json.loads(args.config.read_text(encoding='utf-8'))
    if args.runs is not None and args.runs<1:parser.error('--runs must be positive')
    if args.seed is not None:config['first_seed']=args.seed
    if args.runs is not None:
        for i in range(1,5):config[f'lab{i}']['runs']=args.runs
    numbers=list(range(1,5)) if args.lab=='all' else [int(args.lab)]
    if not args.plots_only:
        for i in numbers:
            cfg=config[f'lab{i}']
            if cfg['population']<30 or cfg['budget']<cfg['population']:parser.error('Population >=30 and budget >= population required')
            run_lab(i,config)
            save_json(ROOT/f'lab{i}'/'results'/'config.json',config)
        save_json(ROOT/'environment.json',{'python':platform.python_version(),'numpy':np.__version__,'platform':platform.platform()})
    from reports import generate
    generate(numbers,config)


if __name__=='__main__':main()
