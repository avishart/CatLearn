import unittest
from .functions import get_slab_ads


class TestMLGO(unittest.TestCase):
    """Test if the MLGO works and give the right output."""

    def test_mlgo_init(self):
        "Test if the MLGO can be initialized."
        import numpy as np
        from catlearn.optimize.mlgo import MLGO
        from ase.calculators.emt import EMT

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
            ads=ads,
            ase_calc=EMT(),
            bounds=bounds,
            initial_points=2,
            norelax_points=10,
            min_steps=6,
            full_output=False,
        )

    def test_mlgo_run(self):
        "Test if the MLGO can run and converge."
        import numpy as np
        from catlearn.optimize.mlgo import MLGO
        from ase.calculators.emt import EMT

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
        # Set random seed
        np.random.seed(1)
        # Initialize MLGO
        mlgo = MLGO(
            slab=slab,
            ads=ads,
            ase_calc=EMT(),
            bounds=bounds,
            initial_points=2,
            norelax_points=10,
            min_steps=6,
            full_output=False,
            local_opt_kwargs=dict(logfile=None),
            tabletxt=None,
        )
        # Test if the MLGO can be run
        mlgo.run(
            fmax=0.05,
            unc_convergence=0.025,
            steps=50,
            max_unc=0.050,
            ml_steps=500,
            ml_chains=2,
            relax=True,
            local_steps=100,
            seed=0,
        )
        # Check that MLGO converged
        self.assertTrue(mlgo.converged() is True)
        # Check that MLGO used the right number of iterations
        self.assertTrue(mlgo.step == 16)


if __name__ == "__main__":
    unittest.main()
