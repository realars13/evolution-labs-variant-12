import itertools
import unittest
import numpy as np
from evolution.common import Budget,reflect,real_ga
from evolution.continuous import qing,evolution_strategy
from evolution.routing import ox,mutate,generate_data,objectives,search
from evolution import scheduling as s


class AlgorithmTests(unittest.TestCase):
    def test_qing_known_optima(self):
        x=np.sqrt(np.arange(1,6))
        for signs in itertools.product([-1,1],repeat=5):
            self.assertAlmostEqual(float(qing((x*signs)[None,:])[0]),0)

    def test_reflection_multiple_widths(self):
        x=reflect(np.array([-41.,-1.,0.,11.,41.]),0.,10.)
        np.testing.assert_allclose(x,[1,1,0,9,1])

    def test_counter_rejects_overrun(self):
        counter=Budget(qing,3);counter.evaluate(np.zeros((3,5)))
        with self.assertRaises(ValueError):counter.evaluate(np.zeros((1,5)))
        self.assertEqual(counter.used,3)

    def test_permutations_preserved(self):
        rng=np.random.default_rng(9)
        for _ in range(100):
            a,b=rng.permutation(np.arange(1,21)),rng.permutation(np.arange(1,21))
            for kind in ['swap','inversion']:
                self.assertEqual(sorted(mutate(ox(a,b,rng),rng,kind)),list(range(1,21)))

    def test_route_objective_against_scalar(self):
        data=generate_data();route=data['reference_route'];cycle=[0,*route,0]
        f=objectives([route],data)[0]
        length=sum(data['distance'][a][b] for a,b in zip(cycle,cycle[1:]))
        duration=sum(data['time'][a][b] for a,b in zip(cycle,cycle[1:]))+sum(data['service'])
        np.testing.assert_allclose(f,[length,duration,max(0,duration-data['time_limit'])])
        self.assertEqual(f[2],0)
        with self.assertRaises(ValueError):objectives([[1]*20],data)

    def test_fronts_match_definition(self):
        rng=np.random.default_rng(4);f=rng.integers(0,8,(30,3)).astype(float)
        remaining=set(range(len(f)))
        for front in s.fronts(f):
            expected={i for i in remaining if not any(np.all(f[j]<=f[i]) and np.any(f[j]<f[i]) for j in remaining)}
            self.assertEqual(set(front),expected);remaining-=expected
        self.assertFalse(remaining)

    def test_duplicates_and_crowding(self):
        f=np.array([[1.,1.],[1.,1.],[0.,3.],[3.,0.],[4.,4.]])
        self.assertEqual(set(s.fronts(f)[0]),{0,1,2,3})
        self.assertTrue(np.all(np.isfinite(s.crowding(np.ones((4,3))))))

    def test_hypervolume_inclusion_exclusion(self):
        points=np.array([[1.,2.,1.],[2.,1.,1.]])
        # Two boxes of volume 4, intersecting box of volume 2.
        self.assertEqual(s.hypervolume(points,np.array([3.,3.,3.])),6.)

    def test_timetable_objectives(self):
        data={'cohorts':[0,0,1],'teachers':[0,1,0], 'conflicts':[[0,1],[0,2]],'days':2,'slots_per_day':5}
        # Cohort 0 has a gap between slots 0 and 2; two events at slot 0 conflict.
        np.testing.assert_array_equal(s.objectives([[0,2,0]],data),[[1,1,0]])
        np.testing.assert_array_equal(s.objectives([[0,3,4]],data),[[0,2,2]])

    def test_schedule_availability_and_archive(self):
        data=s.generate_data()
        result=s.search(data,np.random.default_rng(12),660,population=30)
        values=s.objectives(result['solutions'],data)
        np.testing.assert_array_equal(values,result['front'])
        self.assertEqual(len(values),len(np.unique(values,axis=0)))
        self.assertEqual(len(s.fronts(values)),1)
        with self.assertRaises(ValueError):s.objectives([[1]*24],data)

    def test_reproducibility_and_exact_budgets(self):
        for algorithm in [lambda rng:real_ga(qing,rng,137,population=30),
                          lambda rng:evolution_strategy(qing,rng,137,population=30,offspring=40),
                          lambda rng:search(generate_data(customers=15),rng,137,population=30),
                          lambda rng:s.search(s.generate_data(events=20),rng,137,population=30)]:
            a,b=algorithm(np.random.default_rng(8)),algorithm(np.random.default_rng(8))
            self.assertEqual(a,b);self.assertEqual(a['evaluations'],137)
            values=np.array(a['history'])[:,1]
            # NSGA-II individual coordinate minima need not be monotone.
            if 'violation' in a:
                history=np.array(a['history'])
                self.assertTrue(np.all(np.diff(history[:,2])<=1e-9))
                feasible=history[history[:,2]<1e-9,1]
                self.assertTrue(np.all(np.diff(feasible)<=1e-9))
            elif 'front' not in a:self.assertTrue(np.all(np.diff(values)<=1e-9))


if __name__=='__main__':unittest.main()
