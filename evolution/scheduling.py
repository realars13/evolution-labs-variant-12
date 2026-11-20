"""NSGA-II implemented from nondominated sorting and crowding distance."""
import numpy as np
from .common import Budget, tournament


def generate_data(seed=10114, events=24, days=5, slots_per_day=5):
    rng=np.random.default_rng(seed)
    # Every event belongs to one student cohort and one teacher.
    cohorts=np.arange(events)%6
    teachers=rng.integers(0,5,events)
    # Two early slots for four cohort events force a conflict/late trade-off.
    allowed=[]
    for cohort in cohorts:
        day=int(cohort)%days
        allowed.append([day*slots_per_day,day*slots_per_day+2,
                        day*slots_per_day+3,((day+1)%days)*slots_per_day+4])
    pairs=[[i,j] for i in range(events) for j in range(i+1,events)
           if cohorts[i]==cohorts[j] or teachers[i]==teachers[j]]
    return {'cohorts':cohorts.tolist(),'teachers':teachers.tolist(),
            'conflicts':pairs,'allowed_slots':allowed,'days':days,'slots_per_day':slots_per_day,'seed':seed}


def objectives(x,data):
    x=np.asarray(x,dtype=int)
    slots=data['slots_per_day'];days=data['days']
    if x.ndim!=2 or x.shape[1]!=len(data['cohorts']) or np.any((x<0)|(x>=slots*days)):
        raise ValueError('Invalid timetable encoding')
    if 'allowed_slots' in data:
        for i,allowed in enumerate(data['allowed_slots']):
            if not np.all(np.isin(x[:,i],allowed)):raise ValueError('Event outside availability')
    pairs=np.asarray(data['conflicts'])
    conflicts=(x[:,pairs[:,0]]==x[:,pairs[:,1]]).sum(axis=1)
    gaps=np.zeros(len(x),dtype=int)
    for cohort in sorted(set(data['cohorts'])):
        events=np.flatnonzero(np.asarray(data['cohorts'])==cohort)
        for day in range(days):
            occupied=np.stack([np.any(x[:,events]==day*slots+s,axis=1) for s in range(slots)],axis=1)
            count=occupied.sum(axis=1)
            first=np.argmax(occupied,axis=1)
            last=slots-1-np.argmax(occupied[:,::-1],axis=1)
            gaps+=np.where(count>0,last-first+1-count,0)
    late=(x%slots>=slots-2).sum(axis=1)
    return np.column_stack([conflicts,gaps,late]).astype(float)


def fronts(values):
    """Strict Pareto dominance, all objectives minimised, duplicates equal."""
    dominates=(values[:,None,:]<=values[None,:,:]).all(axis=2)&(values[:,None,:]<values[None,:,:]).any(axis=2)
    counts=dominates.sum(axis=0)
    current=np.flatnonzero(counts==0)
    result=[]
    while len(current):
        result.append(current)
        counts[current]=-1
        counts-=dominates[current].sum(axis=0)
        current=np.flatnonzero(counts==0)
    return result


def crowding(values):
    n=len(values);distance=np.zeros(n)
    if n<=2: return np.full(n,np.inf)
    for j in range(values.shape[1]):
        order=np.argsort(values[:,j],kind='stable')
        span=values[order[-1],j]-values[order[0],j]
        if span==0: continue
        distance[order[[0,-1]]]=np.inf
        distance[order[1:-1]]+=(values[order[2:],j]-values[order[:-2],j])/span
    return distance


def ranks(values):
    rank=np.zeros(len(values),dtype=int);distance=np.zeros(len(values))
    for r,front in enumerate(fronts(values)):
        rank[front]=r;distance[front]=crowding(values[front])
    return rank,distance


def environmental(values,size):
    chosen=[]
    for front in fronts(values):
        if len(chosen)+len(front)<=size: chosen.extend(front)
        else:
            order=np.argsort(-crowding(values[front]),kind='stable')
            chosen.extend(front[order[:size-len(chosen)]])
            break
    return np.asarray(chosen,dtype=int)


def search(data,rng,budget,population=60,method='nsga2'):
    n=len(data['cohorts']);slots=data['days']*data['slots_per_day']
    counter=Budget(lambda x:objectives(x,data),budget)
    allowed=np.asarray(data['allowed_slots'])
    def sample(count):
        alleles=rng.integers(allowed.shape[1],size=(count,n))
        return allowed[np.arange(n)[None,:],alleles]
    x=sample(population);f=counter.evaluate(x)
    archive_x=x.copy();archive_f=f.copy();history=[]
    # Fixed normalisation, independent of the population being compared.
    normalizer=np.array([len(data['conflicts']),n*(data['slots_per_day']-1),n])
    weights=np.array([.5,.3,.2])
    while True:
        history.append((counter.used,*f.min(axis=0).tolist()))
        if counter.used==budget: break
        count=min(population,budget-counter.used)
        if method=='nsga2':
            rank,dist=ranks(f)
            candidates=rng.integers(len(x),size=(2*count,2))
            parents=np.array([a if (rank[a],-dist[a])<(rank[b],-dist[b]) else b for a,b in candidates])
        elif method=='weighted':
            parents=tournament(rng,(f/normalizer)@weights,2*count)
        else: raise ValueError('Unknown multiobjective method')
        a,b=x[parents[:count]],x[parents[count:]]
        children=np.where(rng.random(a.shape)<.5,a,b)
        mask=rng.random(children.shape)<1/n
        children=np.where(mask,sample(count),children)
        cf=counter.evaluate(children)
        combined_x=np.vstack([x,children]);combined_f=np.vstack([f,cf])
        if method=='nsga2': chosen=environmental(combined_f,population)
        else: chosen=np.argsort((combined_f/normalizer)@weights,kind='stable')[:population]
        x,f=combined_x[chosen],combined_f[chosen]
        # Same external archive for BOTH algorithms; retain distinct genotypes.
        archive_x=np.vstack([archive_x,children]);archive_f=np.vstack([archive_f,cf])
        _,unique=np.unique(archive_x,axis=0,return_index=True)
        nondominated=fronts(archive_f[unique])[0]
        keep=unique[nondominated]
        archive_x,archive_f=archive_x[keep],archive_f[keep]
    return {'value':float(np.min((archive_f/normalizer)@weights)),
            'evaluations':counter.used,'history':history,
            'solutions':archive_x.tolist(),'front':archive_f.tolist(),
            'feasible':bool(np.any(archive_f[:,0]==0))}


def hypervolume(points,reference):
    """Exact 3-D dominated volume for a fixed minimisation reference point."""
    points=np.unique(np.asarray(points),axis=0)
    points=points[np.all(points<reference,axis=1)]
    if not len(points):return 0.
    volume=0.
    xs=sorted(set(points[:,0].tolist()+[float(reference[0])]))
    for left,right in zip(xs,xs[1:]):
        yz=points[points[:,0]<=left,1:]
        ys=sorted(set(yz[:,0].tolist()+[float(reference[1])]))
        area=0.
        for low,high in zip(ys,ys[1:]):
            area+=(high-low)*(reference[2]-np.min(yz[yz[:,0]<=low,1]))
        volume+=(right-left)*area
    return float(volume)
