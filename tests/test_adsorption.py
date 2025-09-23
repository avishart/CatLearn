import unittest
from .functions import get_slab_ads, check_fmax


class TestAdsorption(unittest.TestCase):
    """
    Test if the Adsorption works and give the right output.
    """

    def test_adsorption_init(self):
        "Test if the Adsorption can be initialized."
        import numpy as np
        from catlearn.activelearning.adsorption import AdsorptionAL
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
        # Initialize Adsorption AL
        AdsorptionAL(
            slab=slab,
            adsorbate=ads,
            ase_calc=EMT(),
            unc_convergence=0.025,
            bounds=bounds,
            min_data=4,
            verbose=False,
            seed=seed,
        )

    def test_adsorption_run(self):
        "Test if the Adsorption can run and converge."
        import numpy as np
        from catlearn.activelearning.adsorption import AdsorptionAL
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
        # Initialize Adsorption AL
        ads_al = AdsorptionAL(
            slab=slab,
            adsorbate=ads,
            ase_calc=EMT(),
            unc_convergence=0.025,
            bounds=bounds,
            min_data=4,
            verbose=False,
            seed=seed,
        )
        # Test if the Adsorption AL can be run
        ads_al.run(
            fmax=0.05,
            steps=50,
            max_unc=0.3,
            ml_steps=4000,
        )
        # Check that Adsorption AL converged
        self.assertTrue(ads_al.converged() is True)
        # Check that Adsorption AL give a minimum
        atoms = ads_al.get_best_structures()
        self.assertTrue(check_fmax(atoms, EMT(), fmax=0.05))


if __name__ == "__main__":
    unittest.main()
