# CatLearn

CatLearn utilieties machine learning in form of Gaussian Process or Student T process to accelerate catalysis simulations. The Nudged-elastic-band method (NEB) is accelerated with MLNEB code. Furthermore, a global adsorption search is accelerated with the MLGO code. 
CalLearn uses ASE for handling the atomic systems and the calculator interface for the potential energy calculations.

## Installation

You can simply install CatLearn by dowloading it from github as:
```shell
$ git clone --single-branch --branch activelearning https://github.com/avishart/CatLearn
$ pip install -e CatLearn/.
```

You can also install CatLearn directly from github:
```shell
$ pip install git@github.com:avishart/CatLearn.git@activelearning
```

However, it is recommended to install a specific tag to ensure it is a stable version:
```shell
$ pip install git+https://github.com/avishart/CatLearn.git@v.x.x.x
```

## Usage

The following code shows how to use LocalAL:
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
    verbose=True,
)
dyn.run(
    fmax=0.05,
    max_unc=0.30,
    steps=100,
    ml_steps=1000,
)

```

The following code shows how to use MLNEB:
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
    verbose=True,
)
mlneb.run(
    fmax=0.05,
    max_unc=0.30,
    steps=100,
    ml_steps=1000,
)

```

The following code shows how to use MLGO:
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
    verbose=True
)
mlgo.run(
    fmax=0.05,
    max_unc=0.30,
    steps=100,
    ml_steps=1000,
)

```

