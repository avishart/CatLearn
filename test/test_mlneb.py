import sys
from ase.io import read
from ase.neb import NEB
import numpy as np
from catlearn.optimize.mlneb import MLNEB
from catlearn.optimize.acquisition import Acquisition
import copy
from catlearn.optimize.functions_calc import MullerBrown
from ase import Atoms
from ase.optimize import BFGS
import unittest

np.random.seed(1)
# 1. Structural relaxation.

# Setup calculator:
ase_calculator = MullerBrown()

# # 1.1. Structures:
initial_structure = Atoms('C', positions=[(-0.55, 1.41, 0.0)])
final_structure = Atoms('C', positions=[(0.626, 0.025, 0.0)])

initial_structure.set_calculator(copy.deepcopy(ase_calculator))
final_structure.set_calculator(copy.deepcopy(ase_calculator))

# 1.2. Optimize initial and final end-points.

# Initial end-point:
initial_opt = BFGS(initial_structure)
initial_opt.run(fmax=0.01)

# Final end-point:
final_opt = BFGS(final_structure)
final_opt.run(fmax=0.01)


class TestMLNEB(unittest.TestCase):
    """ General test of the ML-NEB algorithm."""
    def test_path(self):
        """Test ML-NEB algorithm running with an interpolated path"""
        np.random.seed(1)
        n_images = 8
        images = [initial_structure]
        for i in range(1, n_images-1):
            image = initial_structure.copy()
            image.set_calculator(copy.deepcopy(ase_calculator))
            images.append(image)
        images.append(final_structure)
        
        neb = NEB(images, climb=True)
        neb.interpolate(method='linear')
        neb_catlearn = MLNEB(start=initial_structure,
                             end=final_structure,
                             interpolation=images,
                             ase_calc=MullerBrown, ase_calc_kwargs={},
                             restart=False)
        
        neb_catlearn.run(fmax=0.05, max_step=0.2)
        
        atoms_catlearn = read('evaluated_structures.traj', ':')
        n_eval_catlearn = len(atoms_catlearn) - 2
        self.assertEqual(n_eval_catlearn, 12)
        print('Checking number of function calls using 8 images...')
        np.testing.assert_array_equal(n_eval_catlearn, 12)
        max_unc = np.max(neb_catlearn.uncertainty_path)
        unc_test = 0.03245814099892351
        
        print('Checking uncertainty on the path (8 images):')
        np.testing.assert_array_almost_equal(max_unc, unc_test, decimal=4)
        
    def test_restart(self):
        """ Here we test the restart flag, the mic, and the internal
            interpolation."""
        np.random.seed(1)
        # Checking internal interpolation.
        neb_catlearn = MLNEB(start=initial_structure,
                             end=final_structure,
                             n_images=9,
                             ase_calc=MullerBrown, ase_calc_kwargs={},
                             interpolation='linear',
                             trajectory='ML-NEB.traj',
                             restart=False)
        neb_catlearn.run(fmax=0.05, max_step=0.2)
        
        print('Checking number of iterations using 9 images...')
        self.assertEqual(neb_catlearn.iter, 13)
        max_unc = np.max(neb_catlearn.uncertainty_path)
        unc_test = 0.028897920744506096
        print('Checking uncertainty on the path (9 images):')
        np.testing.assert_array_almost_equal(max_unc, unc_test, decimal=4)

        # Reducing the uncertainty and fmax, varying num. images (restart):
        print("Checking restart flag...")
        print('Using tighter convergence criteria.')
        neb_catlearn = MLNEB(start=initial_structure,
                             end=final_structure,
                             n_images=11,
                             ase_calc=MullerBrown, ase_calc_kwargs={},
                             trajectory='ML-NEB.traj',
                             restart=True)
        neb_catlearn.run(fmax=0.01, max_step=0.20,unc_convergence=0.010)
        
        print('Checking number of iterations restarting with 11 images...')
        self.assertEqual(neb_catlearn.iter, 3)
        print('Checking uncertainty on the path (11 images).')
        max_unc = np.max(neb_catlearn.uncertainty_path)
        unc_test = 0.008099052829329361
        np.testing.assert_array_almost_equal(max_unc, unc_test, decimal=4)
    
    def test_acquisition(self):
        """ Here we test the acquisition functions"""
        np.random.seed(1)
        print('Checking acquisition function 1 using 6 images...')
        acq=Acquisition(mode='ue',objective='max',kappa=2)
        neb_catlearn = MLNEB(start=initial_structure,
                             end=final_structure,
                             n_images=6,
                             ase_calc=MullerBrown, ase_calc_kwargs={},
                             trajectory='ML-NEB.traj',
                             acq=acq,
                             restart=False)
        neb_catlearn.run(fmax=0.05, max_step=0.2)

        self.assertEqual(neb_catlearn.iter, 14)
        max_unc = np.max(neb_catlearn.uncertainty_path)
        unc_test = 0.002346
        np.testing.assert_array_almost_equal(max_unc, unc_test, decimal=4)

        print('Checking acquisition function 2 using 6 images...')
        acq=Acquisition(mode='ume',objective='max',kappa=2)
        neb_catlearn = MLNEB(start=initial_structure,
                             end=final_structure,
                             n_images=6,
                             ase_calc=MullerBrown, ase_calc_kwargs={},
                             trajectory='ML-NEB.traj',
                             acq=acq,
                             restart=False)
        neb_catlearn.run(fmax=0.05, max_step=0.2)
        print(neb_catlearn.iter,neb_catlearn.uncertainty_path)
        self.assertEqual(neb_catlearn.iter, 12)
        max_unc = np.max(neb_catlearn.uncertainty_path)
        unc_test = 0.015781
        np.testing.assert_array_almost_equal(max_unc, unc_test, decimal=4)

        print('Checking acquisition function 3 using 6 images...')
        acq=Acquisition(mode='umue',objective='max',kappa=2)
        neb_catlearn = MLNEB(start=initial_structure,
                             end=final_structure,
                             n_images=6,
                             ase_calc=MullerBrown, ase_calc_kwargs={},
                             trajectory='ML-NEB.traj',
                             acq=acq,
                             restart=False)
        neb_catlearn.run(fmax=0.05, max_step=0.2)

        self.assertEqual(neb_catlearn.iter, 19)
        max_unc = np.max(neb_catlearn.uncertainty_path)
        unc_test = 0.000003
        np.testing.assert_array_almost_equal(max_unc, unc_test, decimal=4)

        print('Checking acquisition function 4 using 6 images...')
        acq=Acquisition(mode='sume',objective='max',kappa=2)
        neb_catlearn = MLNEB(start=initial_structure,
                             end=final_structure,
                             n_images=6,
                             ase_calc=MullerBrown, ase_calc_kwargs={},
                             trajectory='ML-NEB.traj',
                             acq=acq,
                             restart=False)
        neb_catlearn.run(fmax=0.05, max_step=0.2)

        self.assertEqual(neb_catlearn.iter, 14)
        max_unc = np.max(neb_catlearn.uncertainty_path)
        unc_test = 0.000828
        np.testing.assert_array_almost_equal(max_unc, unc_test, decimal=4)

        print('Checking acquisition function 5 using 6 images...')
        acq=Acquisition(mode='umucb',objective='max',kappa=2)
        neb_catlearn = MLNEB(start=initial_structure,
                             end=final_structure,
                             n_images=6,
                             ase_calc=MullerBrown, ase_calc_kwargs={},
                             trajectory='ML-NEB.traj',
                             acq=acq,
                             restart=False)
        neb_catlearn.run(fmax=0.05, max_step=0.2)

        self.assertEqual(neb_catlearn.iter, 13)
        max_unc = np.max(neb_catlearn.uncertainty_path)
        unc_test = 0.040405
        np.testing.assert_array_almost_equal(max_unc, unc_test, decimal=4)

if __name__ == '__main__':
    unittest.main()
