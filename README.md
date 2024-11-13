# CatLearn

CatLearn utilities machine learning in form of Gaussian Process or Student T process to accelerate catalysis simulations. The local optimization of a structure is accelerated with the `LocalAL` code. The Nudged-elastic-band method (NEB) is accelerated with `MLNEB` code. Furthermore, a global adsorption search is accelerated with the `MLGO` code. 
CalLearn uses ASE to handle the atomic systems and the calculator interface to calculate the potential energy.

## Installation

You can simply install CatLearn by downloading it from GitHub as:
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
    ml_steps=1000,
)

```

The `MLGO` optimization can be visualized in the same way as the `LocalAL` optimization.
