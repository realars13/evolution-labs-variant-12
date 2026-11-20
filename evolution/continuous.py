"""Qing objective and evolution strategies with coordinate step sizes."""
import numpy as np
from .common import Budget, reflect


def qing(x):
    return np.sum((x*x - np.arange(1, x.shape[1]+1))**2, axis=1)


def ellipsoid(x):
    return np.sum(x*x * np.geomspace(1, 1e4, x.shape[1]), axis=1)


def evolution_strategy(function, rng, budget, mode='adaptive', population=40,
                       offspring=80, dimension=5, lower=-10., upper=10.):
    """(mu+lambda)-ES; adaptive mode inherits and mutates log step sizes."""
    counter = Budget(function, budget)
    x = rng.uniform(lower, upper, (population, dimension))
    sigma = np.full_like(x, .05*(upper-lower))
    f = counter.evaluate(x)
    history, steps = [(counter.used, float(f.min()))], []
    tau0, tau = 1 / np.sqrt(2*dimension), 1 / np.sqrt(2*np.sqrt(dimension))
    while counter.used < budget:
        count = min(offspring, budget-counter.used)
        a, b = rng.integers(population, size=(2, count))
        z = (x[a] + x[b]) / 2
        s = np.sqrt(sigma[a] * sigma[b])
        if mode == 'adaptive':
            s *= np.exp(tau0*rng.normal(size=(count,1)) + tau*rng.normal(size=s.shape))
            s = np.clip(s, 1e-10, upper-lower)
        elif mode != 'fixed':
            raise ValueError('Unknown strategy mode')
        z = reflect(z + s*rng.normal(size=z.shape), lower, upper)
        scores = counter.evaluate(z)
        all_f = np.r_[f, scores]
        chosen = np.argsort(all_f, kind='stable')[:population]
        x = np.concatenate([x, z])[chosen]
        sigma = np.concatenate([sigma, s])[chosen]
        f = all_f[chosen]
        history.append((counter.used, float(f.min())))
        steps.append([counter.used, *np.median(sigma, axis=0).tolist()])
    return {'value': float(f.min()), 'solution': x[f.argmin()].tolist(),
            'evaluations': counter.used, 'history': history, 'steps': steps}
