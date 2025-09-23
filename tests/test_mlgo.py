import unittest
from .functions import get_slab_ads, check_fmax


class TestMLGO(unittest.TestCase):
    """
    Test if the MLGO works and give the right output.
    """

    def test_mlgo_init(self):
        "Test if the MLGO can be initialized."
        import numpy as np
        from catlearn.activelearning.mlgo import MLGO
        from ase.calculators.emt import EMT

        # Set random seed to give the same results every time
        seed = 1
        # Get the initial and final states
        slab, ads = get_slab_ads()
        # Make the boundary conditions for the global search
        bounds = np.array(
            [
                [0.0, 1.0],
                [0.0, 1.0],
                [0.5, 0.95],
                [0.0, 2 * np.pi],
                [0.0, 2 * np.pi],
                [0.0, 2 * np.pi],
            ]
        )
        # Initialize MLGO
        MLGO(
            slab=slab,
            adsorbate=ads,
            ase_calc=EMT(),
            unc_convergence=0.025,
            bounds=bounds,
            min_data=4,
            verbose=False,
            local_opt_kwargs=dict(logfile=None),
            seed=seed,
        )

    def test_mlgo_run(self):
        "Test if the MLGO can run and converge."
        import numpy as np
        from catlearn.activelearning.mlgo import MLGO
        from ase.calculators.emt import EMT

        # Set random seed to give the same results every time
        seed = 1
        # Get the initial and final states
        slab, ads = get_slab_ads()
        # Make the boundary conditions for the global search
        bounds = np.array(
            [
                [0.0, 0.5],
                [0.0, 0.5],
                [0.5, 0.95],
                [0.0, 2 * np.pi],
                [0.0, 2 * np.pi],
                [0.0, 2 * np.pi],
            ]
        )
        # Initialize MLGO
        mlgo = MLGO(
            slab=slab,
            adsorbate=ads,
            ase_calc=EMT(),
            unc_convergence=0.025,
            bounds=bounds,
            min_data=4,
            verbose=False,
            local_opt_kwargs=dict(logfile=None),
            seed=seed,
        )
        # Test if the MLGO can be run
        mlgo.run(
            fmax=0.05,
            steps=50,
            max_unc=0.3,
            ml_steps=4000,
            ml_steps_local=1000,
        )
        # Check that MLGO converged
        self.assertTrue(mlgo.converged() is True)
        # Check that MLGO give a minimum
        atoms = mlgo.get_best_structures()
        self.assertTrue(check_fmax(atoms, EMT(), fmax=0.05))


if __name__ == "__main__":
    unittest.main()
