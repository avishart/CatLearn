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
    gp=None,train_images=[],fingerprint=None,use_forces=True,optimize=True,index_mask=None,**opt_kwargs
    -------------------
    gp: Gaussian process class
        The Gaussian proccess used for the training and predictions
    train_images: list
        List of Atoms objects containing the observations which will be use
        to train the model.
    fingerprint: Fingerprint (class name)
        Fingerprint class to be used in the model
    use_forces: bool
        Whether to train the GP on forces
    optimize: bool
        Whether to optimize the GP
    index_mask: None or list
        The index of the atoms that are constrained
    opt_kwargs: dict
        The optimization arguments for the hyperparameter tuning of the GP
    """

    def __init__(self,gp=None,train_images=[],fingerprint=None,use_forces=True,optimize=True,index_mask=None,**opt_kwargs):
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
        self.opt_kwargs=opt_kwargs
        self.index_mask=index_mask
        # Initialize training set
        self.train_images,self.train_features,self.train_targets=[],[],[]
        if len(train_images):
            self.add_training_points(train_images)
            self.train_model(**opt_kwargs)

    def new_fingerprint(self,atoms,calc_gradients=False):
        """ Compute a fingerprint of 'atoms' with given parameters. """
        return self.fp.create([atoms[self.not_masked]])[0]

    def add_training_points(self,train_images):
        """ Calculate fingerprints and add features and corresponding targets. """
        # Check train_images has the right format
        if not isinstance(train_images,(list,np.ndarray)):
            train_images=[train_images]
        # Constrained atoms 
        self.num_atoms=len(train_images[0])
        self.not_masked=list(range(self.num_atoms))
        if self.index_mask is not None:
            self.not_masked=[i for i in self.not_masked if i not in self.index_mask]
        for im in train_images:
            image = copy_image(im)
            # Calculate fingerprint, energies, and forces
            fp = self.new_fingerprint(image)
            if self.use_forces:
                force=image.get_forces(apply_constraint=False)[self.not_masked].flatten()
            energy=image.get_potential_energy(apply_constraint=False)
            # Store the data
            self.train_images.append(image)
            self.train_features.append(fp)
            y=np.concatenate([[energy],-force]) if self.use_forces else np.array([energy])
            self.train_targets.append(y)
        return self

    def train_model(self):
        """ Train a Gaussian process with given data and return it trained. """
        if self.optimize:
            self.optimize_model(**self.opt_kwargs)
        else:
            self.gp.train(np.array(self.train_features),np.array(self.train_targets))
        return self.gp

    def optimize_model(self,retrain=True,hp=None,maxiter=None,prior=None,verbose=False):
        " Optimize the hyperparameters of the GP "
        self.gp.optimize(np.array(self.train_features),np.array(self.train_targets),retrain=retrain,\
                hp=hp,maxiter=maxiter,prior=prior,verbose=verbose)
        return self.gp

    def calculate(self,atoms,get_variance=True):
        """
        Calculate energy, forces and uncertainty for the given atoms. 
        If get_variance==False, variance is returned
        as None. Constrained can also be used, which gives 0 forces
        """
        # Calculate fingerprint:
        fp=np.array([self.new_fingerprint(atoms)])
        # Calculate energy, forces, and uncertainty
        if get_variance:
            y,unc=self.gp.predict(fp,get_variance=get_variance,get_derivatives=True)
            energy,forces,uncertainty=y[0][0],-y[0][1:].reshape(-1,3),np.sqrt(unc.item(0))
        else:
            y=self.gp.predict(fp,get_variance=get_variance,get_derivatives=True)[0]
            energy,forces,uncertainty=y[0],-y[1:].reshape(-1,3),None
        forces=[forces[self.not_masked.index(i)] if i in self.not_masked else [0]*3 for i in range(self.num_atoms)]
        return energy,np.array(forces),uncertainty


class GPCalculator(Calculator):

    implemented_properties = ['energy', 'forces', 'uncertainty']
    nolabel = True

    def __init__(self,model=None,calculate_uncertainty=True,index_mask=None):
        " A Gaussian process calculator object applicable in ASE"
        Calculator.__init__(self)
        if model is None:
            model=GPModel(gp=None)
        self.model=model
        self.calculate_uncertainty=calculate_uncertainty
        self.index_mask=index_mask

    def calculate(self,atoms=None,properties=['energy', 'forces', 'uncertainty'],system_changes=all_changes):
        """
        Calculate the energy, forces and uncertainty on the energies for a
        given Atoms structure. Predicted energies can be obtained by
        *atoms.get_potential_energy()*, predicted forces using
        *atoms.get_forces()* and uncertainties using
        *atoms.get_calculator().results['uncertainty'].
        """
        # Atoms object.
        Calculator.calculate(self, atoms, properties, system_changes)
        # Obtain energy and forces for the given geometry from predictions:
        energy,forces,uncertainty=self.model.calculate(atoms,get_variance=self.calculate_uncertainty)
        # Results:
        self.results['energy']=energy
        self.results['forces']=forces
        self.results['uncertainty']=uncertainty


