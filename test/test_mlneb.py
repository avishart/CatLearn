import numpy as np
from ase.build import fcc100,add_adsorbate
from ase.constraints import FixAtoms
from ase.optimize import BFGS
from ase.calculators.emt import EMT

def get_endstructures():
    # Make structure
    slab=fcc100('Al', size=(2, 2, 3))
    add_adsorbate(slab, 'Au', 1.7, 'hollow')
    slab.center(axis=2, vacuum=4.0)
    slab.calc=EMT()
    # Fix second and third layers:
    mask = [atom.tag > 0 for atom in slab]
    slab.set_constraint(FixAtoms(mask=mask))
    # Optimize initial structure
    initial=slab.copy()
    initial.calc=EMT()
    qn = BFGS(initial)
    qn.run(fmax=0.01)
    # Final state
    final=slab.copy()
    final[-1].x+=final.get_cell()[0,0]/2
    final.calc=EMT()
    # Optimize final structure
    qn = BFGS(final)
    qn.run(fmax=0.01)
    return initial,final


class TestMLNEB:

    def test_mlneb_init(self):
        " Test if the MLNEB can be initialized. "
        from catlearn.optimize.mlneb import MLNEB
        initial,final=get_endstructures()
        MLNEB(start=initial,
                    end=final,
                    ase_calc=EMT(),
                    interpolation='linear',
                    n_images=11,
                    full_output=False)
        
    def test_mlneb_run(self):
        " Test if the MLNEB can run and converge. "
        from catlearn.optimize.mlneb import MLNEB
        initial,final=get_endstructures()
        mlneb=MLNEB(start=initial,
                    end=final,
                    ase_calc=EMT(),
                    interpolation='linear',
                    n_images=11,
                    full_output=False)
        mlneb.run(fmax=0.05,max_unc=0.05,unc_convergence=0.05)
        assert mlneb.converged==True




