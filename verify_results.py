"""Check the included experiment tables and saved solutions without rerunning search."""
import csv
import json
from pathlib import Path
import numpy as np
from evolution import continuous,routing,scheduling

ROOT=Path(__file__).resolve().parent


def load(path):return json.loads(path.read_text(encoding='utf-8'))


def main():
    total=0
    for lab in range(1,5):
        folder=ROOT/f'lab{lab}';results=folder/'results'
        config=load(results/'config.json');cfg=config[f'lab{lab}']
        with (results/'runs.csv').open(encoding='utf-8',newline='') as stream:rows=list(csv.DictReader(stream))
        summary=load(results/'summary.json');best=load(results/'best.json')
        for method,stats in summary.items():
            selected=[row for row in rows if row['method']==method]
            assert len(selected)==cfg['runs'],(lab,method,'series length')
            assert len({row['seed'] for row in selected})==cfg['runs'],(lab,method,'duplicate seeds')
            assert all(int(row['evaluations'])==cfg['budget'] for row in selected),(lab,method,'budget')
            values=np.array([float(row['value']) for row in selected])
            np.testing.assert_allclose([stats[k] for k in ['best','mean','median','std','worst']],
                                       [values.min(),values.mean(),np.median(values),values.std(ddof=1),values.max()])
            item=best[method]
            if lab in [1,4]:
                fn=continuous.ellipsoid if method.startswith('ellipsoid') else continuous.qing
                np.testing.assert_allclose(fn(np.array([item['solution']])),[item['value']],rtol=1e-12,atol=1e-30)
            elif lab==2:
                value=routing.objectives([item['solution']],load(folder/'data.json'))[0]
                np.testing.assert_allclose(value,[item['value'],item['duration'],item['violation']])
            else:
                value=scheduling.objectives(item['solutions'],load(folder/'data.json'))
                np.testing.assert_array_equal(value,item['front'])
                assert len(scheduling.fronts(value))==1
                assert item['hypervolume']==scheduling.hypervolume(value,np.array([len(load(folder/'data.json')['conflicts'])+1,97,25]))
        total+=len(rows)
        print(f'Lab {lab}: {len(rows)} rows, statistics and saved solutions verified')
    print(f'Total: {total} independent runs verified')


if __name__=='__main__':main()
