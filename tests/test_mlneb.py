import unittest
from .functions import get_endstructures


class TestMLNEB(unittest.TestCase):
    """ Test if the MLNEB works and give the right output. """

    def test_mlneb_init(self):
        " Test if the MLNEB can be initialized. "
        from catlearn.optimize.mlneb import MLNEB
        from ase.calculators.emt import EMT
        # Get the initial and final states
        initial,final=get_endstructures()
        # Initialize MLNEB
        mlneb=MLNEB(start=initial,end=final,ase_calc=EMT(),interpolation='linear',n_images=11,use_restart_path=True,check_path_unc=True,full_output=False)
        
    def test_mlneb_run(self):
        " Test if the MLNEB can run and converge with restart of path. "
        from catlearn.optimize.mlneb import MLNEB
        from ase.calculators.emt import EMT
        # Get the initial and final states
        initial,final=get_endstructures()
        # Initialize MLNEB
        mlneb=MLNEB(start=initial,end=final,ase_calc=EMT(),interpolation='linear',n_images=11,use_restart_path=True,check_path_unc=True,full_output=False,local_opt_kwargs=dict(logfile='temp.txt'))
        # Test if the MLNEB can be run
        mlneb.run(fmax=0.05,unc_convergence=0.05,steps=50,ml_steps=250,max_unc=0.05)
        # Check that MLNEB converged
        self.assertTrue(mlneb.converged()==True) 
        # Check that MLNEB used the right number of iterations
        self.assertTrue(mlneb.step==6)

    def test_mlneb_run_norestart(self):
        " Test if the MLNEB can run and converge with no restart of path. "
        from catlearn.optimize.mlneb import MLNEB
        from ase.calculators.emt import EMT
        # Get the initial and final states
        initial,final=get_endstructures()
        # Initialize MLNEB
        mlneb=MLNEB(start=initial,end=final,ase_calc=EMT(),interpolation='linear',n_images=11,use_restart_path=False,full_output=False,local_opt_kwargs=dict(logfile='temp.txt'))
        # Test if the MLNEB can be run
        mlneb.run(fmax=0.05,unc_convergence=0.05,steps=50,ml_steps=250,max_unc=0.05)
        # Check that MLNEB converged
        self.assertTrue(mlneb.converged()==True) 
        # Check that MLNEB used the right number of iterations
        self.assertTrue(mlneb.step==6)

    def test_mlneb_run_savememory(self):
        " Test if the MLNEB can run and converge when it saves memory. "
        from catlearn.optimize.mlneb import MLNEB
        from ase.calculators.emt import EMT
        # Get the initial and final states
        initial,final=get_endstructures()
        # Initialize MLNEB
        mlneb=MLNEB(start=initial,end=final,ase_calc=EMT(),interpolation='linear',n_images=11,use_restart_path=True,check_path_unc=True,save_memory=True,full_output=False,local_opt_kwargs=dict(logfile='temp.txt'))
        # Test if the MLNEB can be run
        mlneb.run(fmax=0.05,unc_convergence=0.05,steps=50,ml_steps=250,max_unc=0.05)
        # Check that MLNEB converged
        self.assertTrue(mlneb.converged()==True) 
        # Check that MLNEB used the right number of iterations
        self.assertTrue(mlneb.step==6)

if __name__ == '__main__':
    unittest.main()


