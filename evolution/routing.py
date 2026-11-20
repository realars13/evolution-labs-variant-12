"""Permutation GA for a closed delivery tour with a total time limit."""
import numpy as np
from .common import Budget, tournament


def nearest_route(distance):
    remaining, route, current = set(range(1, len(distance))), [], 0
    while remaining:
        current = min(remaining, key=lambda j: (distance[current,j], j))
        route.append(current); remaining.remove(current)
    return route


def generate_data(seed=10114, customers=20):
    rng = np.random.default_rng(seed)
    coords = np.vstack([[50.,50.], rng.uniform(0,100,(customers,2))])
    distance = np.linalg.norm(coords[:,None]-coords[None,:], axis=2)
    traffic = rng.uniform(.7,2.8, distance.shape)
    time = distance * traffic / 35 * 60
    np.fill_diagonal(time, 0)
    service = np.r_[0, rng.integers(3,9,customers)]
    route = nearest_route(distance)
    cycle = [0,*route,0]
    duration = sum(time[a,b] for a,b in zip(cycle,cycle[1:])) + sum(service)
    return {'coordinates':coords.tolist(), 'distance':distance.tolist(),
            'time':time.tolist(), 'service':service.tolist(),
            'time_limit':float(duration*1.12), 'seed':seed,
            'reference_route':route}


def objectives(routes, data):
    routes = np.asarray(routes, dtype=int)
    n = len(data['coordinates'])-1
    if routes.ndim != 2 or routes.shape[1] != n or not np.all(np.sort(routes,axis=1)==np.arange(1,n+1)):
        raise ValueError('Each route must visit every customer exactly once')
    cycles = np.pad(routes, ((0,0),(1,1)))
    a,b = cycles[:,:-1], cycles[:,1:]
    distance = np.asarray(data['distance'])[a,b].sum(axis=1)
    duration = np.asarray(data['time'])[a,b].sum(axis=1)+sum(data['service'])
    violation = np.maximum(0, duration-data['time_limit'])
    return np.column_stack([distance,duration,violation])


def ox(a,b,rng):
    """Order crossover: preserve one segment and the other parent's order."""
    lo,hi = sorted(rng.choice(len(a)+1,2,replace=False))
    child = np.full(len(a),-1,dtype=int);child[lo:hi] = a[lo:hi]
    remaining = [v for v in np.r_[b[hi:],b[:hi]] if v not in child[lo:hi]]
    positions = list(range(hi,len(a)))+list(range(lo))
    child[positions] = remaining
    return child


def mutate(route,rng,kind):
    child = route.copy()
    a,b = sorted(rng.choice(len(route),2,replace=False))
    if kind == 'inversion': child[a:b+1] = child[a:b+1][::-1]
    elif kind == 'swap': child[a],child[b] = child[b],child[a]
    else: raise ValueError('Unknown mutation')
    return child


def search(data,rng,budget,population=60,penalty=100.,mutation='inversion',random=False):
    n = len(data['coordinates'])-1
    counter = Budget(lambda x: objectives(x,data),budget)
    x = np.array([rng.permutation(np.arange(1,n+1)) for _ in range(population)])
    f = counter.evaluate(x)
    best_route, best = None, None
    history=[]
    while True:
        for route, row in zip(x,f):
            key = (row[2]>1e-9, row[2] if row[2]>1e-9 else row[0])
            if best is None or key < best:
                best,best_route,best_values = key,route.copy(),row.copy()
        history.append((counter.used,float(best_values[0]),float(best_values[2])))
        if counter.used == budget: break
        count = min(population-1,budget-counter.used)
        score=f[:,0]+penalty*f[:,2]
        if random:
            children=np.array([rng.permutation(np.arange(1,n+1)) for _ in range(count)])
        else:
            parents=tournament(rng,score,2*count).reshape(count,2)
            children=[]
            for a,b in parents:
                child=ox(x[a],x[b],rng) if rng.random()<.9 else x[a].copy()
                if rng.random()<.8: child=mutate(child,rng,mutation)
                children.append(child)
            children=np.asarray(children)
        scores=counter.evaluate(children)
        elite=int(score.argmin())
        x=np.concatenate([x[elite:elite+1],children]);f=np.vstack([f[elite],scores])
    return {'value':float(best_values[0]),'duration':float(best_values[1]),
            'violation':float(best_values[2]),'feasible':bool(best_values[2]<1e-9),
            'solution':best_route.tolist(),'evaluations':counter.used,'history':history}
