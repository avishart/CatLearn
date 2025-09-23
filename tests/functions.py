from numpy import argmax, array, concatenate, cos, linspace, sin, sqrt
from numpy.linalg import norm
from numpy.random import default_rng, Generator, RandomState
from ase import Atoms
from ase.calculators.emt import EMT


def get_rng(seed):
    "Get the random number generator."
    if isinstance(seed, int) or seed is None:
        rng = default_rng(seed)
    elif isinstance(seed, Generator):
        rng = seed
    elif isinstance(seed, RandomState):
        rng = seed
    return rng


def create_func(gridsize=200, seed=1):
    "Generate the data set from a trial function"
    rng = get_rng(seed)
    x = linspace(-40, 100, gridsize).reshape(-1, 1)
    f = 3 * (sin((x / 20) ** 2) - 3 * sin(0.6 * x / 20) + 17)
    g = 3 * (
        (2 * x / (20**2)) * cos((x / 20) ** 2)
        - 3 * (0.6 / 20) * cos(0.6 * x / 20)
    )
    i_perm = rng.permutation(list(range(len(x))))
    return x[i_perm], f[i_perm], g[i_perm]


def create_h2_atoms(gridsize=200, seed=1):
    "Generate the trial data set of H2 ASE atoms with EMT"
    rng = get_rng(seed)
    z_list = linspace(0.2, 4.0, gridsize)
    atoms_list = []
    energies, forces = [], []
    for z in z_list:
        h2 = Atoms("H2", positions=array([[0.0, 0.0, 0.0], [z, 0.0, 0.0]]))
        h2.center(vacuum=10.0)
        h2.calc = EMT()
        energies.append(h2.get_potential_energy())
        forces.append(h2.get_forces().reshape(-1))
        atoms_list.append(h2)
    i_perm = rng.permutation(list(range(len(atoms_list))))
    atoms_list = [atoms_list[i] for i in i_perm]
    return (
        atoms_list,
        array(energies).reshape(-1, 1)[i_perm],
        array(forces)[i_perm],
    )


def make_train_test_set(x, f, g, tr=20, te=20, use_derivatives=True):
    "Genterate the training and test sets"
    x_tr, f_tr, g_tr = x[:tr], f[:tr], g[:tr]
    t_all = tr + te
    x_te, f_te, g_te = x[tr:t_all], f[tr:t_all], g[tr:t_all]
    if use_derivatives:
        f_tr = concatenate([f_tr.reshape(tr, 1), g_tr.reshape(tr, -1)], axis=1)
        f_te = concatenate([f_te.reshape(te, 1), g_te.reshape(te, -1)], axis=1)
    return x_tr, f_tr, x_te, f_te


def calculate_rmse(ytest, ypred):
    "Calculate the Root-mean squarred error"
    return sqrt(((ypred - ytest) ** 2).mean())


def check_minima(
    sol,
    x_tr,
    f_tr,
    model,
    pdis=None,
    func=None,
    is_model_gp=True,
    dstep=1e-5,
    dtol=1e-5,
):
    "Check if the solution is a minimum."
    if is_model_gp:
        from catlearn.regression.gp.objectivefunctions.gp import (
            LogLikelihood,
        )
    else:
        from catlearn.regression.gp.objectivefunctions.tp import (
            LogLikelihood,
        )
    from catlearn.regression.gp.optimizers import (
        FunctionEvaluation,
    )
    from catlearn.regression.gp.hpfitter import (
        HyperparameterFitter,
    )

    # Construct optimizer
    if func is None:
        func = LogLikelihood()
    hpfitter = HyperparameterFitter(
        func=func,
        optimizer=FunctionEvaluation(jac=False),
    )
    # Get hyperparameter solution
    hp0 = sol["hp"].copy()
    is_minima = True
    # Iterate over all hyperparameters
    for para, value in hp0.items():
        hp_test = hp0.copy()
        # Get function value of minimum
        sol0 = hpfitter.fit(x_tr, f_tr, model, hp=hp_test, pdis=pdis)
        # Get function value of the larger hyperparameter value
        hp_test[para] = value + dstep
        sol1 = hpfitter.fit(x_tr, f_tr, model, hp=hp_test, pdis=pdis)
        # Get function value of the smaller hyperparameter value
        hp_test[para] = value - dstep
        sol2 = hpfitter.fit(x_tr, f_tr, model, hp=hp_test, pdis=pdis)
        # Check if it is a minimum
        if sol0["fun"] - sol1["fun"] > (
            abs(sol0["fun"]) * dtol + 1e-8
        ) or sol0["fun"] - sol2["fun"] > (abs(sol0["fun"]) * dtol + 1e-8):
            is_minima = False
    return is_minima


def get_endstructures():
    "Create the initial and final states of a NEB test."
    from ase.build import fcc100, add_adsorbate
    from ase.constraints import FixAtoms
    from ase.optimize import BFGS
    from ase.calculators.emt import EMT

    # Make structure
    slab = fcc100("Al", size=(2, 2, 3))
    add_adsorbate(slab, "Au", 1.7, "hollow")
    slab.center(vacuum=4.0, axis=2)
    slab.calc = EMT()
    # Fix Al atoms
    mask = [atom.symbol == "Al" for atom in slab]
    slab.set_constraint(FixAtoms(mask=mask))
    # Optimize initial structure
    initial = slab.copy()
    initial.calc = EMT()
    with BFGS(initial, logfile=None) as qn:
        qn.run(fmax=0.01)
    # Final state
    final = slab.copy()
    final[-1].x += final.get_cell()[0, 0] / 2
    final.calc = EMT()
    # Optimize final structure
    with BFGS(final, logfile=None) as qn:
        qn.run(fmax=0.01)
    return initial, final


def get_slab_ads():
    "Create the surface slab and the adsorbate of a adsorption search."
    from ase import Atoms
    from ase.build import fcc111
    from ase.constraints import FixAtoms
    from ase.calculators.emt import EMT
    from ase.optimize import BFGS

    # Make the surface
    slab = fcc111("Pd", size=(2, 2, 2))
    slab.center(vacuum=5.0, axis=2)
    slab.pbc = True
    slab.calc = EMT()
    with BFGS(slab, logfile=None) as qn:
        qn.run(fmax=0.01)
    # Fix Pd atoms
    mask = [atom.symbol == "Pd" for atom in slab]
    slab.set_constraint(FixAtoms(mask=mask))
    # Make the adsorbate
    ads = Atoms("O", cell=slab.cell.copy(), pbc=slab.pbc)
    ads.center()
    return slab, ads


def check_fmax(atoms, calc, fmax=0.05):
    "Check the structure is an optimum."
    atoms_c = atoms.copy()
    atoms_c.calc = calc
    forces = atoms_c.get_forces()
    return norm(forces, axis=1).max() < fmax


def check_image_fmax(images, calc, fmax=0.05):
    "Check images from NEB has a saddle point."
    energies = [image.get_potential_energy() for image in images]
    i_max = argmax(energies)
    return check_fmax(images[i_max], calc, fmax=fmax)
