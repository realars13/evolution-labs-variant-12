"""Shared experiment accounting and real-valued operators."""
import csv
import json
from pathlib import Path
import numpy as np


class Budget:
    """Every evaluated candidate, including initialisation, consumes one unit."""
    def __init__(self, function, limit):
        self.function, self.limit, self.used = function, limit, 0

    def evaluate(self, population):
        population = np.asarray(population)
        if self.used + len(population) > self.limit:
            raise ValueError('Evaluation budget exceeded')
        self.used += len(population)
        return self.function(population)


def reflect(x, lower, upper):
    width = np.asarray(upper) - np.asarray(lower)
    return lower + width - np.abs((x - lower) % (2 * width) - width)


def tournament(rng, scores, size, k=3):
    candidates = rng.integers(len(scores), size=(size, k))
    return candidates[np.arange(size), np.argmin(scores[candidates], axis=1)]


def real_ga(function, rng, budget, population=60, dimension=5,
            lower=-500., upper=500., scale=0.01):
    counter = Budget(function, budget)
    x = rng.uniform(lower, upper, (population, dimension))
    f = counter.evaluate(x)
    history = [(counter.used, float(f.min()))]
    while counter.used < budget:
        count = min(population - 1, budget - counter.used)
        a = x[tournament(rng, f, count)]
        b = x[tournament(rng, f, count)]
        # BLX-alpha with alpha=0.3, applied with probability 0.9.
        weight = rng.uniform(-0.3, 1.3, (count, dimension))
        children = np.where(rng.random((count, 1)) < .9,
                            a + weight * (b - a), a)
        mask = rng.random(children.shape) < 1 / dimension
        children = reflect(children + mask * rng.normal(0, scale * (upper-lower), children.shape), lower, upper)
        scores = counter.evaluate(children)
        elite = int(f.argmin())
        x = np.concatenate([x[elite:elite+1], children])
        f = np.r_[f[elite], scores]
        history.append((counter.used, float(f.min())))
    return {'value': float(f.min()), 'solution': x[f.argmin()].tolist(),
            'evaluations': counter.used, 'history': history}


def random_search(function, rng, budget, dimension, lower, upper, batch=60):
    counter = Budget(function, budget)
    best, solution, history = np.inf, None, []
    while counter.used < budget:
        x = rng.uniform(lower, upper, (min(batch, budget-counter.used), dimension))
        f = counter.evaluate(x)
        if f.min() < best:
            best, solution = float(f.min()), x[f.argmin()].tolist()
        history.append((counter.used, best))
    return {'value': best, 'solution': solution, 'evaluations': counter.used, 'history': history}


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def save_csv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
