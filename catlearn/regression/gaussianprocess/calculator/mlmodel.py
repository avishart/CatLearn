import numpy as np
from copy import deepcopy

class MLModel:
    def __init__(self,model=None,database=None,baseline=None,optimize=True,optimize_kwargs={},**kwargs):
        " ML model used for a ASE calculator. "
        self.set_model_database(model=model,database=database)
        self.baseline=deepcopy(baseline)
        self.optimize=optimize
        self.optimize_kwargs=optimize_kwargs.copy()
        self.use_baseline=False if self.baseline is None else True

    def add_training(self,atoms_list):
        " Add training ase Atoms data to the database. "
        if isinstance(atoms_list,(list,np.ndarray)):
            self.database.add_set(atoms_list)
        else:
            self.database.add(atoms_list)
        return self

    def train_model(self):
        " Train the ML model. "
        features=self.database.get_features()
        targets=self.database.get_targets()
        if self.use_baseline:
            targets=self.baseline_correction(targets,None,add=False,use_forces=self.database.use_forces,negative_forces=self.database.negative_forces)
        if self.optimize:
            return self.optimize_model(features,targets,**self.optimize_kwargs)
        self.model.train(features,targets)
        return self.model

    def optimize_model(self,features,targets,retrain=True,hp=None,prior=None,verbose=False,update_prior_dist=False):
        " Optimize the ML model with the arguments set in optimize_kwargs. "
        # Update the prior distribution
        if update_prior_dist and prior is not None:
            from ..pdistributions.pdistributions import make_prior
            prior=make_prior(self.model,list(prior.keys()),features,targets,prior_dis=prior,scale=1)
        return self.model.optimize(features,targets,retrain=retrain,hp=hp,prior=prior,verbose=verbose)

    def baseline_correction(self,targets,atoms=None,add=False,use_forces=True,negative_forces=True):
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
            y_base.append(self.database.get_target(atoms_base,use_forces=use_forces,negative_forces=negative_forces))
        # Either add or subtract it from targets
        if add:
            return targets+np.array(y_base)
        return targets-np.array(y_base)

    def calculate(self,atoms,get_variance=True,get_forces=True):
        """ Calculate the energy and also the uncertainties and forces if selected.
        If get_variance=False, variance is returned as None. """
        # Calculate fingerprint:
        fp=self.database.get_atoms_feature(atoms)
        # Calculate energy, forces, and uncertainty
        pred=self.model.predict(np.array([fp]),get_variance=get_variance,get_derivatives=get_forces)
        # Get the uncertainty if it requested
        if get_variance:
            y,uncertainty=pred[0][0].copy(),np.sqrt(pred[1].item(0))
        else:
            y,uncertainty=pred[0].copy(),None
        # Correct with the baseline if it is used
        if self.use_baseline:
            y=self.baseline_correction(y,atoms=atoms,add=True)
        energy=y[0]
        # Get the forces if they are requested
        if get_forces:
            not_masked=self.database.get_constrains(atoms)
            if self.model.use_derivatives:
                forces=-y[1:].reshape(-1,3)
            else:
                forces=-self.derivatives_fd(atoms,not_masked=not_masked).reshape(-1,3)
            forces=np.array([forces[not_masked.index(i)] if i in not_masked else [0.0]*3 for i in range(len(atoms))])
            return energy,forces,uncertainty
        return energy,uncertainty

    def derivatives_fd(self,atoms,d_step=1e-5,not_masked=[]):
        " Calculate the derivatives of the energy (-forces) from finite difference "
        # Copy atoms and get positions
        atoms_c=atoms.copy()
        pos=atoms_c.get_positions().reshape(-1)
        pos_m=np.array([pos]*len(pos))
        # Make the first finite difference
        pred1=self.fd_part(pos_m.copy(),atoms_c,sign=1,d_step=d_step,not_masked=not_masked)
        # Make the second finite difference
        pred2=self.fd_part(pos_m.copy(),atoms_c,sign=-1,d_step=d_step,not_masked=not_masked)
        # Calculate derivatives
        return ((pred1-pred2)/(2*d_step)).reshape(-1,3)

    def fd_part(self,pos_fd,atoms_c,sign=1,d_step=1e-5,not_masked=[]):
        " Calculate the finite difference part "
        # Make the first finite difference
        pos_fd[range(len(pos_fd)),range(len(pos_fd))]+=sign*d_step
        atoms_list=[atoms_c.set_positions(pos.reshape(-1,3)) for p,pos in enumerate(pos_fd) if int(np.ceil(p/3)) not in not_masked]
        fps=np.array([self.database.get_atoms_feature(atoms_c2) for atoms_c2 in atoms_list])
        pred_fd=self.model.predict(fps,get_variance=False,get_derivatives=False)
        if self.use_baseline:
            pred_fd=np.array([self.baseline_correction(pred_fd[i:i+1],atoms=atoms_list[i],add=True,use_forces=False) for i in range(len(atoms_list))])
        return pred_fd.reshape(-1)

    def set_model_database(self,model=None,database=None):
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
        self.model=deepcopy(model)
        self.database=deepcopy(database)
        # Check the model and database have the same attributes
        self.check_attributes()
        return self

    def get_default_model(self,use_derivatives=True,use_fingerprint=False):
        " Get the ML model as a default GP model. "
        from ..gp.gp import GaussianProcess
        from ..kernel.se import SE,SE_Derivative
        from ..means.median import Prior_median
        from ..hpfitter import HyperparameterFitter
        from ..objectfunctions.factorized_likelihood import FactorizedLogLikelihood
        from ..optimizers import run_golden,line_search_scale
        local_kwargs=dict(tol=1e-5,optimize=True,multiple_max=True)
        kwargs_optimize=dict(local_run=run_golden,maxiter=5000,jac=False,bounds=None,ngrid=80,use_bounds=True,local_kwargs=local_kwargs)
        hpfitter=HyperparameterFitter(FactorizedLogLikelihood(),optimization_method=line_search_scale,opt_kwargs=kwargs_optimize,distance_matrix=True)
        kernel=SE_Derivative(use_fingerprint=use_fingerprint) if use_derivatives else SE(use_fingerprint=use_fingerprint)
        model=GaussianProcess(prior=Prior_median(),kernel=kernel,use_derivatives=use_derivatives,hpfitter=hpfitter)
        return model

    def get_default_database(self,use_derivatives=True,use_fingerprint=False):
        " Get a default database used to keep track of the training systems. "
        from .database import Database
        from ..fingerprint.cartesian import Cartesian
        fp=Cartesian(reduce_dimensions=True,use_derivatives=use_derivatives)
        database=Database(fingerprint=fp,reduce_dimensions=True,use_derivatives=use_derivatives,negative_forces=True,use_fingerprint=use_fingerprint)
        return database

    def check_attributes(self):
        " Check if all attributes agree between the class and subclasses. "
        if self.model.kernel.use_fingerprint!=self.database.use_fingerprint:
            raise Exception('Model and Database do not agree whether to use fingerprints!')
        if self.model.use_derivatives!=self.database.use_derivatives:
            raise Exception('Model and Database do not agree whether to use derivatives/forces!')
        return





