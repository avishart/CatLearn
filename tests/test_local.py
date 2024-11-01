import unittest
import numpy as np
from .functions import get_endstructures, check_fmax


class TestLocal(unittest.TestCase):
    """
    Test if the local active learning (AL) optimization works and
    give the right output.
    """

    def test_local_init(self):
        "Test if the local AL can be initialized."
        from catlearn.activelearning.local import LocalAL
        from ase.calculators.emt import EMT

        # Get the atoms from initial and final states
        atoms, _ = get_endstructures()
        # Move the gold atom up to prepare optimization
        pos = atoms.get_positions()
        pos[-1, 2] += 0.5
        atoms.set_positions(pos)
        atoms.get_forces()
        # Set random seed
        np.random.seed(1)
        # Initialize Local AL optimization
        LocalAL(
            atoms=atoms,
            ase_calc=EMT(),
            unc_convergence=0.02,
            use_restart=True,
            check_unc=True,
            verbose=False,
        )

    def test_local_run(self):
        "Test if the local AL can run and converge with restart of path."
        from catlearn.activelearning.local import LocalAL
        from ase.calculators.emt import EMT

        # Get the atoms from initial and final states
        atoms, _ = get_endstructures()
        # Move the gold atom up to prepare optimization
        pos = atoms.get_positions()
        pos[-1, 2] += 0.5
        atoms.set_positions(pos)
        atoms.get_forces()
        # Set random seed
        np.random.seed(1)
        # Initialize Local AL optimization
        local_al = LocalAL(
            atoms=atoms,
            ase_calc=EMT(),
            unc_convergence=0.02,
            use_restart=True,
            check_unc=True,
            verbose=False,
        )
        # Test if the Local AL optimization can be run
        local_al.run(
            fmax=0.05,
            steps=50,
            ml_steps=250,
            max_unc=0.05,
        )
        # Check that Local AL optimization converged
        self.assertTrue(local_al.converged() is True)
        # Check that Local AL optimization gives a saddle point
        atoms = local_al.get_best_structures()
        self.assertTrue(check_fmax(atoms, EMT(), fmax=0.05))


if __name__ == "__main__":
    unittest.main()
