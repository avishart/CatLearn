import unittest
import numpy as np
from .functions import get_endstructures, check_image_fmax


class TestMLNEB(unittest.TestCase):
    """Test if the MLNEB works and give the right output."""

    def test_mlneb_init(self):
        "Test if the MLNEB can be initialized."
        from catlearn.optimize.mlneb import MLNEB
        from ase.calculators.emt import EMT

        # Get the initial and final states
        initial, final = get_endstructures()
        # Set random seed
        np.random.seed(1)
        # Initialize MLNEB
        MLNEB(
            start=initial,
            end=final,
            ase_calc=EMT(),
            interpolation="linear",
            n_images=11,
            use_restart_path=True,
            check_path_unc=True,
            full_output=False,
        )

    def test_mlneb_run(self):
        "Test if the MLNEB can run and converge with restart of path."
        from catlearn.optimize.mlneb import MLNEB
        from ase.calculators.emt import EMT

        # Get the initial and final states
        initial, final = get_endstructures()
        # Set random seed
        np.random.seed(1)
        # Initialize MLNEB
        mlneb = MLNEB(
            start=initial,
            end=final,
            ase_calc=EMT(),
            interpolation="linear",
            n_images=11,
            use_restart_path=True,
            check_path_unc=True,
            full_output=False,
            local_opt_kwargs=dict(logfile=None),
            tabletxt=None,
        )
        # Test if the MLNEB can be run
        mlneb.run(
            fmax=0.05,
            unc_convergence=0.05,
            steps=50,
            ml_steps=250,
            max_unc=0.05,
        )
        # Check that MLNEB converged
        self.assertTrue(mlneb.converged() is True)
        # Check that MLNEB gives a saddle point
        images = mlneb.get_images()
        self.assertTrue(check_image_fmax(images, EMT(), fmax=0.05))

    def test_mlneb_run_idpp(self):
        """
        Test if the MLNEB can run and converge with
        restart of path from IDPP.
        """
        from catlearn.optimize.mlneb import MLNEB
        from ase.calculators.emt import EMT

        # Get the initial and final states
        initial, final = get_endstructures()
        # Set random seed
        np.random.seed(1)
        # Initialize MLNEB
        mlneb = MLNEB(
            start=initial,
            end=final,
            ase_calc=EMT(),
            interpolation="idpp",
            n_images=11,
            use_restart_path=True,
            check_path_unc=True,
            full_output=False,
            local_opt_kwargs=dict(logfile=None),
            tabletxt=None,
        )
        # Test if the MLNEB can be run
        mlneb.run(
            fmax=0.05,
            unc_convergence=0.05,
            steps=50,
            ml_steps=250,
            max_unc=0.05,
        )
        # Check that MLNEB converged
        self.assertTrue(mlneb.converged() is True)
        # Check that MLNEB gives a saddle point
        images = mlneb.get_images()
        self.assertTrue(check_image_fmax(images, EMT(), fmax=0.05))

    def test_mlneb_run_path(self):
        """
        Test if the MLNEB can run and converge with
        restart of path from different initial paths.
        """
        from catlearn.optimize.mlneb import MLNEB
        from ase.calculators.emt import EMT

        # Get the initial and final states
        initial, final = get_endstructures()
        interpolations = ["idpp", "rep", "ends"]
        for interpolation in interpolations:
            with self.subTest(interpolation=interpolation):
                # Set random seed
                np.random.seed(1)
                # Initialize MLNEB
                mlneb = MLNEB(
                    start=initial,
                    end=final,
                    ase_calc=EMT(),
                    interpolation=interpolation,
                    n_images=11,
                    use_restart_path=True,
                    check_path_unc=True,
                    full_output=False,
                    local_opt_kwargs=dict(logfile=None),
                    tabletxt=None,
                )
                # Test if the MLNEB can be run
                mlneb.run(
                    fmax=0.05,
                    unc_convergence=0.05,
                    steps=50,
                    ml_steps=250,
                    max_unc=0.05,
                )
                # Check that MLNEB converged
                self.assertTrue(mlneb.converged() is True)
                # Check that MLNEB gives a saddle point
                images = mlneb.get_images()
                self.assertTrue(check_image_fmax(images, EMT(), fmax=0.05))

    def test_mlneb_run_norestart(self):
        "Test if the MLNEB can run and converge with no restart of path."
        from catlearn.optimize.mlneb import MLNEB
        from ase.calculators.emt import EMT

        # Get the initial and final states
        initial, final = get_endstructures()
        # Set random seed
        np.random.seed(1)
        # Initialize MLNEB
        mlneb = MLNEB(
            start=initial,
            end=final,
            ase_calc=EMT(),
            interpolation="linear",
            n_images=11,
            use_restart_path=False,
            full_output=False,
            local_opt_kwargs=dict(logfile=None),
            tabletxt=None,
        )
        # Test if the MLNEB can be run
        mlneb.run(
            fmax=0.05,
            unc_convergence=0.05,
            steps=50,
            ml_steps=250,
            max_unc=0.05,
        )
        # Check that MLNEB converged
        self.assertTrue(mlneb.converged() is True)
        # Check that MLNEB gives a saddle point
        images = mlneb.get_images()
        self.assertTrue(check_image_fmax(images, EMT(), fmax=0.05))

    def test_mlneb_run_savememory(self):
        "Test if the MLNEB can run and converge when it saves memory."
        from catlearn.optimize.mlneb import MLNEB
        from ase.calculators.emt import EMT

        # Get the initial and final states
        initial, final = get_endstructures()
        # Set random seed
        np.random.seed(1)
        # Initialize MLNEB
        mlneb = MLNEB(
            start=initial,
            end=final,
            ase_calc=EMT(),
            interpolation="linear",
            n_images=11,
            use_restart_path=True,
            check_path_unc=True,
            save_memory=True,
            full_output=False,
            local_opt_kwargs=dict(logfile=None),
            tabletxt=None,
        )
        # Test if the MLNEB can be run
        mlneb.run(
            fmax=0.05,
            unc_convergence=0.05,
            steps=50,
            ml_steps=250,
            max_unc=0.05,
        )
        # Check that MLNEB converged
        self.assertTrue(mlneb.converged() is True)
        # Check that MLNEB gives a saddle point
        images = mlneb.get_images()
        self.assertTrue(check_image_fmax(images, EMT(), fmax=0.05))

    def test_mlneb_run_no_maxunc(self):
        "Test if the MLNEB can run and converge when it does not use max_unc."
        from catlearn.optimize.mlneb import MLNEB
        from ase.calculators.emt import EMT

        # Get the initial and final states
        initial, final = get_endstructures()
        # Set random seed
        np.random.seed(1)
        # Initialize MLNEB
        mlneb = MLNEB(
            start=initial,
            end=final,
            ase_calc=EMT(),
            interpolation="linear",
            n_images=11,
            use_restart_path=True,
            check_path_unc=True,
            full_output=False,
            local_opt_kwargs=dict(logfile=None),
            tabletxt=None,
        )
        # Test if the MLNEB can be run
        mlneb.run(
            fmax=0.05,
            unc_convergence=0.05,
            steps=50,
            ml_steps=250,
            max_unc=False,
        )
        # Check that MLNEB converged
        self.assertTrue(mlneb.converged() is True)
        # Check that MLNEB gives a saddle point
        images = mlneb.get_images()
        self.assertTrue(check_image_fmax(images, EMT(), fmax=0.05))


if __name__ == "__main__":
    unittest.main()
