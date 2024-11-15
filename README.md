# CatLearn

CatLearn utilizes machine learning in the form of the Gaussian Process or Student T process to accelerate catalysis simulations.

The local optimization of a structure is accelerated with the `LocalAL` code.
The Nudged-elastic-band method (NEB) is accelerated with `MLNEB` code.
Furthermore, a global adsorption search without local relaxation is accelerated with the `AdsorptionAL` code.
Additionally, a global adsorption search with local relaxation is accelerated with the `MLGO` code. 

CalLearn uses ASE to handle the atomic systems and the calculator interface to calculate the potential energy.

## Installation

You can install CatLearn by downloading it from GitHub as:
```shell
$ git clone --single-branch --branch activelearning https://github.com/avishart/CatLearn
$ pip install -e CatLearn/.
```

You can also install CatLearn directly from GitHub:
```shell
$ pip install git@github.com:avishart/CatLearn.git@activelearning
```

However, it is recommended to install a specific tag to ensure it is a stable version:
```shell
$ pip install git+https://github.com/avishart/CatLearn.git@v.x.x.x
```

## Usage
The active learning class is generalized to work for any defined optimizer method for ASE `Atoms` structures. The optimization method is executed iteratively with a machine-learning calculator that is retrained for each iteration. The active learning converges when the uncertainty is low (`unc_convergence`) and the energy change is within `unc_convergence` or the maximum force is within the tolerance value set.

Predefined active learning methods are created: `LocalAL`, `MLNEB`, `AdsorptionAL`, and `MLGO`.

The outputs of the active learning are `predicted.traj`, `evaluated.traj`, `converged.traj`, `initial_struc.traj`, and `ml_summary.txt`. 
The `predicted.traj` file contains the structures that the machine-learning calculator predicts after each optimization loop. The training data and ASE calculator evaluated structures are within `evaluated.traj` file. The converged structures calculated with the machine-learning calculator are saved in the `converged.traj` file. The initial structure(s) is/are saved into the `initial_struc.traj` file. The summary of the active learning is saved into a table in the `ml_summary.txt` file.

### LocalAL
The following code shows how to use `LocalAL`:
```python
from catlearn.activelearning.local import LocalAL
from ase.io import read
from ase.optimize import FIRE

# Load initial structure
atoms = read("initial.traj")

# Make the ASE calculator
calc = ...

# Initialize local optimization
dyn = LocalAL(
    atoms=atoms,
    ase_calc=calc,
    unc_convergence=0.05,
    local_opt=FIRE,    
    local_opt_kwargs={},
    save_memory=False,
    use_restart=True,
    min_data=3,
    restart=False,
    verbose=True,
)
dyn.run(
    fmax=0.05,
    max_unc=0.30,
    steps=100,
    ml_steps=1000,
)

```

The active learning minimization can be visualized by extending the Python script with the following code:
```python
import matplotlib.pyplot as plt
from catlearn.tools.plot import plot_minimize

fig, ax = plt.subplots()
plot_minimize("predicted.traj", "evaluated.traj", ax=ax)
plt.savefig('AL_minimization.png')
plt.close()
```

### MLNEB
The following code shows how to use `MLNEB`:
```python
from catlearn.activelearning.mlneb import MLNEB
from ase.io import read
from ase.optimize import FIRE

# Load endpoints
initial = read("initial.traj")
final = read("final.traj")

# Make the ASE calculator
calc = ...

# Initialize MLNEB
mlneb = MLNEB(
    start=initial,
    end=final,
    ase_calc=calc,
    unc_convergence=0.05,
    n_images=15,
    neb_method="improvedtangentneb",
    neb_kwargs={},
    neb_interpolation="linear",
    reuse_ci_path=True,
    save_memory=False,
    parallel_run=False,
    local_opt=FIRE,    
    local_opt_kwargs={},
    use_restart=True,
    min_data=3,
    restart=False,
    verbose=True,
)
mlneb.run(
    fmax=0.05,
    max_unc=0.30,
    steps=100,
    ml_steps=1000,
)

```

The obtained NEB band from the MLNEB optimization can be visualized in three ways.

The converged NEB band with uncertainties can be visualized by extending the Python code with the following code:
```python
import matplotlib.pyplot as plt
from catlearn.tools.plot import plot_neb

fig, ax = plt.subplots()
plot_neb(mlneb.get_structures(), use_uncertainty=True, ax=ax)
plt.savefig('Converged_NEB.png')
plt.close()
```

The converged NEB band can also be plotted with the predicted curve between the images by extending with the following code:
```python
import matplotlib.pyplot as plt
from catlearn.tools.plot import plot_neb_fit_mlcalc

fig, ax = plt.subplots()
plot_neb_fit_mlcalc(
    mlneb.get_structures(),
    mlcalc=mlneb.get_mlcalc(),
    use_uncertainty=True,
    ax=ax,
)
plt.savefig('Converged_NEB_fit.png')
plt.close()
```

All the obtained NEB bands from `MLNEB` can also be visualized within the same figure by using the following code:
```python
import matplotlib.pyplot as plt
from catlearn.tools.plot import plot_all_neb

fig, ax = plt.subplots()
plot_all_neb("predicted.traj", n_images=15, ax=ax)
plt.savefig('All_NEB_paths.png')
plt.close()
```

### AdsorptionAL
The following code shows how to use `AdsorptionAL`:
```python
from catlearn.activelearning.adsorption import AdsorptionAL
from ase.io import read

# Load the slab and the adsorbate
slab = read("slab.traj")
ads = read("adsorbate.traj")

# Make the ASE calculator
calc = ...

# Make the boundary conditions for the adsorbate
bounds = np.array(
    [
        [0.0, 1.0],
        [0.0, 1.0],
        [0.5, 1.0],
        [0.0, 2 * np.pi],
        [0.0, 2 * np.pi],
        [0.5, 2 * np.pi],
    ]
)

# Initialize MLGO
dyn = AdsorptionAL(
    slab=slab,
    adsorbate=ads,
    adsorbate2=None,
    ase_calc=calc,
    unc_convergence=0.02,
    bounds=bounds,
    opt_kwargs={},
    parallel_run=False,
    min_data=3,
    restart=False,
    verbose=True
)
dyn.run(
    fmax=0.05,
    max_unc=0.30,
    steps=100,
    ml_steps=4000,
)

```

The `AdsorptionAL` optimization can be visualized in the same way as the `LocalAL` optimization.

### MLGO
The following code shows how to use `MLGO`:
```python
from catlearn.activelearning.mlgo import MLGO
from ase.io import read
from ase.optimize import FIRE

# Load the slab and the adsorbate
slab = read("slab.traj")
ads = read("adsorbate.traj")

# Make the ASE calculator
calc = ...

# Make the boundary conditions for the adsorbate
bounds = np.array(
    [
        [0.0, 1.0],
        [0.0, 1.0],
        [0.5, 1.0],
        [0.0, 2 * np.pi],
        [0.0, 2 * np.pi],
        [0.5, 2 * np.pi],
    ]
)

# Initialize MLGO
mlgo = MLGO(
    slab=slab,
    adsorbate=ads,
    adsorbate2=None,
    ase_calc=calc,
    unc_convergence=0.02,
    bounds=bounds,
    opt_kwargs={},
    local_opt=FIRE,
    local_opt_kwargs={},
    reuse_data_local=True,
    parallel_run=False,
    min_data=3,
    restart=False,
    verbose=True
)
mlgo.run(
    fmax=0.05,
    max_unc=0.30,
    steps=100,
    ml_steps=4000,
)

```

The `MLGO` optimization can be visualized in the same way as the `LocalAL` optimization.
