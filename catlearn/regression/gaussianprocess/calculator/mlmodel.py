import numpy as np

class MLModel:
    def __init__(self,model=None,database=None,baseline=None,optimize=True,optimize_kwargs={},**kwargs):
        " ML model used for a ASE calculator. "
        self.set_model_database(model=model,database=database)
        if baseline is None:
            self.baseline=None
            self.use_baseline=False
        else:
            self.baseline=baseline.copy()
            self.use_baseline=True
        self.optimize=optimize
        self.optimize_kwargs=optimize_kwargs.copy()

    def add_training(self,atoms_list):
        " Add training ase Atoms data to the database. "
        if isinstance(atoms_list,(list,np.ndarray)):
            self.database.add_set(atoms_list)
        else:
            self.database.add(atoms_list)
        return self

    def train_model(self,verbose=False,**kwargs):
        " Train the ML model. "
        features=self.database.get_features()
        targets=self.database.get_targets()
        if self.use_baseline:
            targets=self.baseline_correction(targets,None,add=False,use_derivatives=self.database.use_derivatives,negative_forces=self.database.negative_forces)
        if self.optimize:
            return self.optimize_model(features,targets,verbose=verbose,**self.optimize_kwargs)
        self.model.train(features,targets)
        return self.model

    def optimize_model(self,features,targets,retrain=True,hp=None,pdis=None,verbose=False,update_pdis=False,**kwargs):
        " Optimize the ML model with the arguments set in optimize_kwargs. "
        # Update the prior distribution
        if update_pdis and pdis is not None:
            from ..pdistributions.pdistributions import make_prior
            pdis=make_prior(self.model,list(pdis.keys()),features,targets,prior_dis=pdis,scale=1)
        return self.model.optimize(features,targets,retrain=retrain,hp=hp,pdis=pdis,verbose=verbose)

    def baseline_correction(self,targets,atoms=None,add=False,use_derivatives=True,negative_forces=True,**kwargs):
        " Baseline correction if a baseline is used. Either add the correction to an atom object or subtract it from training data. "
        # Whether the baseline is used to training or prediction
        if not add:
            atoms_list=self.database.get_atoms()
        else:
            atoms_list=[atoms]
        # Calculate the baseline for each ASE atoms object
        y_base=[]
        for atoms in atoms_list:
            atoms_base=atoms.copy()
            atoms_base.calc=self.baseline
            y_base.append(self.database.get_target(atoms_base,use_derivatives=use_derivatives,negative_forces=negative_forces))
        # Either add or subtract it from targets
        if add:
            return targets+np.array(y_base)[0]
        return targets-np.array(y_base)

    def calculate(self,atoms,get_variance=True,get_forces=True,**kwargs):
        """ Calculate the energy and also the uncertainties and forces if selected.
        If get_variance=False, variance is returned as None. """
        # Calculate fingerprint:
        fp=self.database.get_atoms_feature(atoms)
        # Calculate energy, forces, and uncertainty
        y,var=self.model.predict(np.array([fp]),get_variance=get_variance,get_derivatives=get_forces)
        # Get the uncertainty if it requested
        uncertainty,uncertainty_forces=None,None
        if get_variance:
            var=np.sqrt(var)
            uncertainty=var[0][0]
            if get_forces:
                uncertainty_forces=var[0][1:].copy()
        # Correct with the baseline if it is used
        if self.use_baseline:
            y=self.baseline_correction(y,atoms=atoms,add=True,use_derivatives=get_forces,negative_forces=self.database.negative_forces)
        # Get energy
        energy=y[0,0]
        # Get the forces if they are requested
        if get_forces:
            natoms=len(atoms)
            not_masked=self.database.get_constrains(atoms)
            forces=-y[0,1:].reshape(-1,3)
            forces=self.not_masked_reshape(forces,not_masked,natoms)
            if uncertainty_forces is not None:
                uncertainty_forces=uncertainty_forces.reshape(-1,3)
                uncertainty_forces=self.not_masked_reshape(uncertainty_forces,not_masked,natoms)
            return energy,forces,uncertainty,uncertainty_forces
        return energy,None,uncertainty,uncertainty_forces

    def not_masked_reshape(self,array,not_masked,natoms):
        " Reshape an array so that it works for all atom coordinates and set constrained indicies to 0. "
        full_array=np.zeros((natoms,3))
        full_array[not_masked]=array
        return full_array

    def set_model_database(self,model=None,database=None,**kwargs):
        " Set the ML model and the database and make sure it uses the same attributes. "
        # Get attributes from either the model or the database if one of them are defined
        if model is not None:
            use_derivatives=model.use_derivatives
            use_fingerprint=model.kernel.use_fingerprint
        elif database is not None:
            use_derivatives=database.use_derivatives
            use_fingerprint=database.use_fingerprint
        else:
            use_derivatives=True
            use_fingerprint=False
        # If the model and/or the database are not defined then get a default one
        if model is None:
            model=self.get_default_model(use_derivatives=use_derivatives,use_fingerprint=use_fingerprint)
        if database is None:
            database=self.get_default_database(use_derivatives=use_derivatives,use_fingerprint=use_fingerprint)
        # Save the model and database
        self.model=model.copy()
        self.database=database.copy()
        # Check the model and database have the same attributes
        self.check_attributes()
        return self

    def get_default_model(self,use_derivatives=True,use_fingerprint=False,**kwargs):
        " Get the ML model as a default GP model. "
        from ..models.gp import GaussianProcess
        from ..kernel.se import SE
        from ..means.mean import Prior_mean
        from ..hpfitter import HyperparameterFitter
        from ..objectivefunctions.gp.factorized_likelihood import FactorizedLogLikelihood
        from ..optimizers import run_golden,line_search_scale
        local_kwargs=dict(tol=1e-5,optimize=True,multiple_max=True)
        kwargs_optimize=dict(local_run=run_golden,maxiter=5000,jac=False,bounds=None,ngrid=80,use_bounds=True,local_kwargs=local_kwargs)
        hpfitter=HyperparameterFitter(FactorizedLogLikelihood(),optimization_method=line_search_scale,opt_kwargs=kwargs_optimize,distance_matrix=True)
        kernel=SE(use_derivatives=use_derivatives,use_fingerprint=use_fingerprint)
        model=GaussianProcess(prior=Prior_mean(),kernel=kernel,use_derivatives=use_derivatives,hpfitter=hpfitter)
        return model

    def get_default_database(self,use_derivatives=True,use_fingerprint=False,**kwargs):
        " Get a default database used to keep track of the training systems. "
        from .database import Database
        from ..fingerprint.cartesian import Cartesian
        fp=Cartesian(reduce_dimensions=True,use_derivatives=use_derivatives)
        database=Database(fingerprint=fp,reduce_dimensions=True,use_derivatives=use_derivatives,negative_forces=True,use_fingerprint=use_fingerprint,**kwargs)
        return database
    
    def save_data(self,trajectory='data.traj',**kwarg):
        " Save the ASE atoms data to a trajectory. "
        self.database.save_data(trajectory=trajectory,**kwarg)
        return self
    
    def get_training_set_size(self):
        " Get the number of atoms objects in the database. "
        return len(self.database)

    def copy(self):
        " Copy the MLModel. "
        clone=self.__class__(model=self.model,
                             database=self.database,
                             baseline=self.baseline,
                             optimize=self.optimize,
                             optimize_kwargs=self.optimize_kwargs)
        return clone

    def check_attributes(self):
        " Check if all attributes agree between the class and subclasses. "
        if self.model.kernel.use_fingerprint!=self.database.use_fingerprint:
            raise Exception('Model and Database do not agree whether to use fingerprints!')
        if self.model.use_derivatives!=self.database.use_derivatives:
            raise Exception('Model and Database do not agree whether to use derivatives/forces!')
        return

