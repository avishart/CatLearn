# CatLearn

CatLearn utilieties machine learning in form of Gaussian Process or Student T process to accelerate catalysis simulations. The Nudged-elastic-band method (NEB) is accelerated with MLNEB code. Furthermore, a global adsorption search is accelerated with the MLGO code. 
CalLearn uses ASE for handling the atomic systems and the calculator interface for the potential energy calculations.

## Installation

You can simply install CatLearn by dowloading it from github as:
```shell
$ git clone https://github.com/avishart/CatLearn/tree/uncertainty_driven
$ pip install -e CatLearn/.
```
or
```shell
$ pip install git@github.com:avishart/CatLearn.git@uncertainty_driven
```

## Usage

The following code shows how to use MLNEB:
```python
from catlearn.optimize.mlneb import MLNEB
from ase.io import read

# Load endpoints
initial = read('initial.traj')
final = read('final.traj')

# Make the ASE calculator
calc = ...

# Initialize MLNEB
mlneb = MLNEB(start=initial, end=final, ase_calc=calc, interpolation='linear', n_images=15, full_output=True)
mlneb.run(fmax=0.05, unc_convergence=0.05, max_unc=0.30, steps=100, ml_steps=1000)
```

The following code shows how to use MLGO:
```python
from catlearn.optimize.mlneb import MLGO
from ase.io import read

# Load the slab and the adsorbate
slab = read('slab.traj')
ads = read('adsorbate.traj')

# Make the ASE calculator
calc = ...

# Make the boundary conditions for the adsorbate
bounds=np.array([[0.0,1.0],[0.0,1.0],[0.5,1.0],[0.0,2*np.pi],[0.0,2*np.pi],[0.0,2*np.pi]])

# Initialize MLGO
mlgo = MLGO(slab, ads, ase_calc=calc, bounds=bounds, full_output=True)
mlgo.run(fmax=0.05, unc_convergence=0.02, max_unc=0.30, steps=100, ml_steps=1000, ml_chains=8, relax=True, local_steps=500)
```

