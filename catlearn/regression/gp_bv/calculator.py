# Made by Jose A Garrido Torres, Estefanía Garijo del Rio, and Sami Juhani Kaappa
import warnings
from copy import copy

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.singlepoint import SinglePointCalculator


def copy_image(atoms):
    """
    Copy an image, so that it is suitable as a training set point.
    It returns a copy of the atoms object with the single point
    calculator attached
    """
    # Check if the input already has the desired format
    if atoms.calc.__class__.__name__ == 'SinglePointCalculator':
        # Return a copy of the atoms object
        calc = copy(atoms.calc)
        atoms0 = atoms.copy()
    else:
        # Check if the atoms object has energy and forces calculated for this position
        # If not, compute them
        atoms.get_forces()
        # Initialize a SinglePointCalculator to store this results
        calc = SinglePointCalculator(atoms, **atoms.calc.results)
    atoms0 = atoms.copy()
    atoms0.calc = calc
    return atoms0


class GPModel:
    """
    GP model parameters
    -------------------
    train_images: list
        List of Atoms objects containing the observations which will be use
        to train the model.

    train_features: list
        List of pre-calculated fingerprints for each structure in
        train_images

    prior: Prior object or None
        Prior for the GP regression of the PES surface. See
        ase.optimize.activelearning.prior. If *Prior* is None, then it is set
        as the ConstantPrior with the constant being updated using the
        update_prior_strategy specified as a parameter.

    kerneltype: str
        One of the possible kernel types: 'sqexp', 'matern', 'rq'

    fingerprint: Fingerprint (class name)
        Fingerprint class to be used in the model

    params: dict
        Dictionary to include all the hyperparameters for the kernel and
        for the fingerprint

    use_forces: bool
        Whether to train the GP on forces
    """

    def __init__(self,gp=None,train_images=[],fingerprint=None,use_forces=True,optimize=True,**opt_kwargs):
        if gp is None:
            from catlearn.regression.gp_bv.gp import GaussianProcess
            gp=GaussianProcess(use_derivatives=use_forces)
        self.gp=gp
        if fingerprint is None:
            from catlearn.regression.gp_bv.fingerprint import Fingerprint_cartessian
            fingerprint = Fingerprint_cartessian()
        self.fp = fingerprint
        self.use_forces=use_forces
        self.optimize=optimize
        # Initialize training set
        self.train_images,self.train_features,self.train_targets=[],[],[]
        self.add_training_points(train_images)
        self.opt_kwargs=opt_kwargs
        self.gp = self.train_model(**opt_kwargs)

    def new_fingerprint(self,atoms,calc_gradients=False):
        """
        Compute a fingerprint of 'atoms' with given parameters.
        """
        return self.fp.create([atoms])[0]

    def add_training_points(self,train_images):
        '''
        Calculate fingerprints and add features and corresponding targets
        to the database of 'data'.
        '''
        for im in train_images:
            image = copy_image(im)
            # Calculate fingerprint, energies, and forces
            fp = self.new_fingerprint(image)
            if self.use_forces:
                force=image.get_forces(apply_constraint=False).flatten()
            energy=image.get_potential_energy(apply_constraint=False)
            # Store the data
            self.train_images.append(image)
            self.train_features.append(fp)
            y=np.concatenate([[energy],-force]) if self.use_forces else np.array([energy])
            self.train_targets.append(y)
        return self

    def train_model(self):
        '''
        Train a Gaussian process with given data.

        Parameters
        ----------
        gp : GaussianProcess instance
        data : Database instance

        Return a trained GaussianProcess instance
        '''
        if self.optimize:
            self.optimize_model(**self.opt_kwargs)
        else:
            self.gp.train(np.array(self.train_features),np.array(self.train_targets))
        return self.gp

    def optimize_model(self,retrain=True,hp=None,maxiter=None,prior=None,verbose=False):
        self.gp.optimize(np.array(self.train_features),np.array(self.train_targets),retrain=retrain,\
                hp=hp,maxiter=maxiter,prior=prior,verbose=verbose)
        return self.gp

    def calculate(self,fp,get_variance=True):
        '''
        Calculate energy, forces and uncertainty for the given
        fingerprint. If get_variance==False, variance is returned
        as None.
        '''
        fp=np.array([fp])
        if get_variance:
            y,unc=self.gp.predict(fp,get_variance=get_variance,get_derivatives=True)
            return y[0],unc[0]
        return self.gp.predict(fp,get_variance=get_variance,get_derivatives=True)[0],None


class GPCalculator(Calculator):

    implemented_properties = ['energy', 'forces', 'uncertainty']
    nolabel = True

    def __init__(self,model=None,calculate_uncertainty=True):
        " A Gaussian process calculator object applicable in ASE"
        Calculator.__init__(self)
        if model is None:
            model=GPModel(gp=None)
        self.model=model
        self.calculate_uncertainty=calculate_uncertainty

    def calculate(self, atoms=None,
                  properties=['energy', 'forces', 'uncertainty'],
                  system_changes=all_changes):
        '''
        Calculate the energy, forces and uncertainty on the energies for a
        given Atoms structure. Predicted energies can be obtained by
        *atoms.get_potential_energy()*, predicted forces using
        *atoms.get_forces()* and uncertainties using
        *atoms.get_calculator().results['uncertainty'].
        '''
        # Atoms object.
        Calculator.calculate(self, atoms, properties, system_changes)
        # Calculate fingerprint:
        x=self.model.new_fingerprint(self.atoms)
        # Get predictions:
        f,V=self.model.calculate(x,get_variance=self.calculate_uncertainty)
        # Obtain energy and forces for the given geometry.
        energy=f[0]
        forces=-f[1:].reshape(-1, 3)
        # Get uncertainty for the given geometry.
        if self.calculate_uncertainty:
            uncertainty=V.item(0)
            if uncertainty < 0.0:
                uncertainty = 0.0
                warning = ('Imaginary uncertainty has been set to zero')
                warnings.warn(warning)
            uncertainty = np.sqrt(uncertainty)
        else:
            uncertainty = None

        # Results:
        self.results['energy']=energy
        self.results['forces']=forces
        self.results['uncertainty']=uncertainty





