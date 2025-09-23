import unittest
from .functions import get_slab_ads, check_fmax


class TestRandomAdsorption(unittest.TestCase):
    """
    Test if the RandomAdsorption works and give the right output.
    """

    def test_randomadsorption_init(self):
        "Test if the RandomAdsorption can be initialized."
        import numpy as np
        from catlearn.activelearning import RandomAdsorptionAL
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
        # Initialize RandomAdsorption AL
        RandomAdsorptionAL(
            slab=slab,
            adsorbate=ads,
            ase_calc=EMT(),
            unc_convergence=0.025,
            bounds=bounds,
            min_data=4,
            verbose=False,
            seed=seed,
        )

    def test_randomadsorption_run(self):
        "Test if the RandomAdsorption can run and converge."
        import numpy as np
        from catlearn.activelearning import RandomAdsorptionAL
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
        # Initialize RandomAdsorption AL
        ads_al = RandomAdsorptionAL(
            slab=slab,
            adsorbate=ads,
            ase_calc=EMT(),
            n_random_draws=20,
            use_initial_opt=True,
            initial_fmax=0.2,
            unc_convergence=0.025,
            bounds=bounds,
            min_data=4,
            verbose=False,
            seed=seed,
        )
        # Test if the RandomAdsorption AL can be run
        ads_al.run(
            fmax=0.05,
            steps=50,
            max_unc=0.3,
            ml_steps=5000,
        )
        # Check that RandomAdsorption AL converged
        self.assertTrue(ads_al.converged() is True)
        # Check that RandomAdsorption AL give a minimum
        atoms = ads_al.get_best_structures()
        self.assertTrue(check_fmax(atoms, EMT(), fmax=0.05))

    def test_randomadsorption_run_no_initial_opt(self):
        """
        Test if the RandomAdsorption without initial optimization
        can run and converge.
        """
        import numpy as np
        from catlearn.activelearning import RandomAdsorptionAL
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
        # Initialize RandomAdsorption AL
        ads_al = RandomAdsorptionAL(
            slab=slab,
            adsorbate=ads,
            ase_calc=EMT(),
            n_random_draws=50,
            use_initial_opt=False,
            unc_convergence=0.025,
            bounds=bounds,
            min_data=4,
            verbose=False,
            seed=seed,
        )
        # Test if the RandomAdsorption AL can be run
        ads_al.run(
            fmax=0.05,
            steps=50,
            max_unc=0.3,
            ml_steps=5000,
        )
        # Check that RandomAdsorption AL converged
        self.assertTrue(ads_al.converged() is True)
        # Check that RandomAdsorption AL give a minimum
        atoms = ads_al.get_best_structures()
        self.assertTrue(check_fmax(atoms, EMT(), fmax=0.05))


if __name__ == "__main__":
    unittest.main()
