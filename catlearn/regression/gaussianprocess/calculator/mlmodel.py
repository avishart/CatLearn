import numpy as np

class MLModel:
    def __init__(self,model=None,database=None,baseline=None,optimize=True,hp=None,pdis=None,verbose=False,**kwargs):
        """
        Machine Learning model used for ASE Atoms and calculator.

        Parameters:
            model : Model
                The Machine Learning Model with kernel and prior that are optimized.
            database : Database object
                The Database object with ASE atoms.
            baseline : Baseline object
                The Baseline object calculator that calculates energy and forces. 
            optimize : bool
                Whether to optimize the hyperparameters when the model is trained.
            hp : dict
                Use a set of hyperparameters to optimize from else the current set is used.
            pdis : dict
                A dict of prior distributions for each hyperparameter type.
            verbose : bool
                Whether to print statements in the optimization.
        """
        # Make default model if it is not given
        if model is None:
            model=get_default_model()
        # Make default database if it is not given
        if database is None:
            database=get_default_database()
        # Use default baseline if it is not given
        if baseline is None:
            self.baseline=None
        # Use default pdis if it is not given
        if pdis is None:
            self.pdis=None
        # Make default hyperparameters if it is not given
        self.hp=None
        # Set the arguments
        self.update_arguments(model=model,
                              database=database,
                              baseline=baseline,
                              optimize=optimize,
                              hp=hp,
                              pdis=pdis,
                              verbose=verbose,
                              **kwargs)

    def add_training(self,atoms_list,**kwargs):
        """
        Add training data in form of the ASE Atoms to the database.

        Parameters:
            atoms_list : list or ASE Atoms
                A list of or a single ASE Atoms with calculated energies and forces.

        Returns:
            self: The updated object itself.
        """
        if isinstance(atoms_list,(list,np.ndarray)):
            self.database.add_set(atoms_list)
        else:
            self.database.add(atoms_list)
        return self

    def train_model(self,**kwargs):
        """ 
        Train the ML model and optimize its hyperparameters if it is chosen. 

        Returns:
            self: The updated object itself.
        """
        # Get data from the data base
        features,targets=self.get_data()
        # Correct targets with a baselin
        if self.use_baseline:
            targets=self.baseline_corrected_targets(targets)
        # Train model
        if self.optimize:
            # Optimize the hyperparameters and train the ML model
            self.model_optimization(features,targets,**kwargs)
        else:
            # Train the ML model
            self.model_training(features,targets,**kwargs)
        return self
    
    def calculate(self,atoms,get_uncertainty=True,get_forces=True,get_force_uncertainties=False,get_unc_derivatives=False,**kwargs):
        """ 
        Calculate the energy and also the uncertainties and forces if selected.
        If get_variance=False, variance is returned as None. 

        Parameters:
            atoms : ASE Atoms
                The ASE Atoms object that the properties (incl. energy) are calculated for.
            get_uncertainty : bool
                Whether to calculate the uncertainty. 
                The uncertainty is None if get_uncertainty=False.
            get_forces : bool
                Whether to calculate the forces.
            get_force_uncertainties : bool
                Whether to calculate the uncertainties of the predicted forces. 
            get_unc_derivatives : bool
                Whether to calculate the derivatives of the uncertainty of the predicted energy.

        Returns:
            energy : float
                The predicted energy of the ASE Atoms.
            forces : (Nat,3) array or None
                The predicted forces if get_forces=True.
            uncertainty : float or None
                The predicted uncertainty of the energy if get_uncertainty=True.
            uncertainty_forces : (Nat,3) array or None
                The predicted uncertainties of the forces if get_uncertainty=True and get_forces=True.
        """
        # Calculate energy, forces, and uncertainty
        energy,forces,unc,unc_forces,unc_deriv=self.model_prediction(atoms,
                                                                     get_uncertainty=get_uncertainty,
                                                                     get_forces=get_forces,
                                                                     get_force_uncertainties=get_force_uncertainties,
                                                                     get_unc_derivatives=get_unc_derivatives)
        # Store the predictions
        results=self.store_results(atoms,energy=energy,forces=forces,unc=unc,unc_forces=unc_forces,unc_deriv=unc_deriv)
        return results
    
    def save_data(self,trajectory='data.traj',**kwarg):
        """
        Save the ASE Atoms data to a trajectory.

        Parameters:
            trajectory : str
                The name of the trajectory file where the data is saved.

        Returns:
            self: The updated object itself.
        """
        " Save the ASE atoms data to a trajectory. "
        self.database.save_data(trajectory=trajectory,**kwarg)
        return self

    def get_training_set_size(self,**kwargs):
        """
        Get the number of atoms objects in the database.

        Returns:
            int: The number of atoms objects in the database.
        """
        return len(self.database)
    
    def is_in_database(self,atoms,**kwargs):
        """ 
        Check if the ASE Atoms is in the database.

        Parameters:
            atoms : ASE Atoms
                The ASE Atoms object with a calculator.

        Returns:
            bool: Whether the ASE Atoms object is within the database.
        """
        return self.database.is_in_database(atoms=atoms,**kwargs)
    
    def copy_atoms(self,atoms,**kwargs):
        """
        Copy the atoms object together with the calculated properties.

        Parameters:
            atoms : ASE Atoms
                The ASE Atoms object with a calculator that is copied.

        Returns:
            ASE Atoms: The copy of the Atoms object with saved data in the calculator.
        """
        return self.database.copy_atoms(atoms,**kwargs)
    
    def update_database_arguments(self,point_interest=None,**kwargs):
        """ 
        Update the arguments in the database.

        Parameters:
            point_interest : list
                A list of the points of interest as ASE Atoms instances. 

        Returns:
            self: The updated object itself.
        """
        self.database.update_arguments(point_interest=point_interest,**kwargs)
        return self
    
    def update_arguments(self,model=None,database=None,baseline=None,optimize=None,hp=None,pdis=None,verbose=None,**kwargs):
        """
        Update the class with its arguments. The existing arguments are used if they are not given.

        Parameters:
            model : Model
                The Machine Learning Model with kernel and prior that are optimized.
            database : Database object
                The Database object with ASE atoms.
            baseline : Baseline object
                The Baseline object calculator that calculates energy and forces. 
            optimize : bool
                Whether to optimize the hyperparameters when the model is trained.
            hp : dict
                Use a set of hyperparameters to optimize from else the current set is used.
            pdis : dict
                A dict of prior distributions for each hyperparameter type.
            verbose : bool
                Whether to print statements in optimization.

        Returns:
            self: The updated object itself.
        """
        if model is not None:
            self.model=model.copy()
        if database is not None:
            self.database=database.copy()
        if baseline is not None:
            self.baseline=baseline.copy()
        if optimize is not None:
            self.optimize=optimize
        if hp is not None:
            self.hp=hp.copy()
        if pdis is not None:
            self.pdis=pdis.copy()
        if verbose is not None:
            self.verbose=verbose
        # Check if the baseline is used
        if self.baseline is None:
            self.use_baseline=False
        else:
            self.use_baseline=True
        # Check that the model and database have the same attributes
        self.check_attributes()
        return self

    def model_optimization(self,features,targets,**kwargs):
        " Optimize the ML model with the arguments set in optimize_kwargs. "
        sol=self.model.optimize(features,targets,retrain=True,hp=self.hp,pdis=self.pdis,verbose=False,**kwargs)
        if self.verbose:
            from ase.parallel import parprint
            parprint(sol)
        return self.model
    
    def model_training(self,features,targets,**kwargs):
        " Train the model without optimizing the hyperparameters. "
        self.model.train(features,targets,**kwargs)
        return self.model
    
    def model_prediction(self,atoms,get_uncertainty=True,get_forces=True,get_force_uncertainties=False,get_unc_derivatives=False,**kwargs):
        " Predict the targets and uncertainties. "
        # Calculate fingerprint
        fp=self.database.make_atoms_feature(atoms)
        # Calculate energy, forces, and uncertainty
        y,var,var_deriv=self.model.predict(np.array([fp]),
                                 get_derivatives=get_forces,
                                 get_variance=get_uncertainty,
                                 include_noise=False,
                                 get_derivtives_var=get_force_uncertainties,
                                 get_var_derivatives=get_unc_derivatives)
        # Correct the predicted targets with the baseline if it is used
        if self.use_baseline:
            y=self.add_baseline_correction(y,atoms=atoms,use_derivatives=get_forces)
        # Extract the energy
        energy=y[0][0]
        # Extract the forces if they are requested
        if get_forces:
            forces=-y[0][1:]
        else:
            forces=None
        # Get the uncertainties if they are requested
        if get_uncertainty:
            unc=np.sqrt(var[0][0])
            # Get the uncertainty of the forces if they are requested
            if get_force_uncertainties and get_forces:
                unc_forces=np.sqrt(unc[0][1:])
            else:
                unc_forces=None
            # Get the derivatives of the predicted uncertainty
            if get_unc_derivatives:
                unc_deriv=(0.5/unc)*var_deriv
            else:
                unc_deriv=None
        else:
            unc=None
            unc_forces=None
            unc_deriv=None
        return energy,forces,unc,unc_forces,unc_deriv
    
    def store_results(self,atoms,energy=None,forces=None,unc=None,unc_forces=None,unc_deriv=None,**kwargs):
        " Store the predicted results in a dictionary. "
        results={}
        # Save the energy
        if energy is not None:
            results['energy']=energy
        # Save the uncertainty
        if unc is not None:
            results['uncertainty']=unc
        # Get constraints
        if (forces is not None) or (unc_forces is not None) or (unc_deriv is not None):
            natoms,not_masked=self.get_constraints(atoms)
        # Make the full matrix of forces and save it
        if forces is not None:
            results['forces']=self.not_masked_reshape(forces,not_masked,natoms)
        # Make the full matrix of force uncertainties and save it
        if unc_forces is not None:
            results['force uncertainties']=self.not_masked_reshape(unc_forces,not_masked,natoms)
        # Make the full matrix of derivatives of uncertainty and save it
        if unc_deriv is not None:
            results['uncertainty derivatives']=self.not_masked_reshape(unc_deriv,not_masked,natoms)
        return results
    
    def add_baseline_correction(self,targets,atoms,use_derivatives=True,**kwargs):
        " Baseline correction if a baseline is used. Add the baseline correction to an atom object. "
        # Calculate the baseline for the ASE atoms object
        y_base=self.calculate_baseline([atoms],use_derivatives=use_derivatives,**kwargs)
        # Add baseline correction to the targets
        return targets+np.array(y_base)[0]
    
    def baseline_corrected_targets(self,targets,**kwargs):
        " Baseline correction if a baseline is used. Subtract the baseline correction from training targets. "
        # Get the ASE atoms from the database
        atoms_list=self.database.get_atoms()
        # Calculate the baseline for each ASE atoms objects
        y_base=self.calculate_baseline(atoms_list,use_derivatives=self.database.use_derivatives,**kwargs)
        # Subtract baseline correction from the targets
        return targets-np.array(y_base)
    
    def calculate_baseline(self,atoms_list,use_derivatives=True,**kwargs):
        " Calculate the baseline for each ASE atoms object. "
        y_base=[]
        for atoms in atoms_list:
            atoms_base=atoms.copy()
            atoms_base.calc=self.baseline
            y_base.append(self.make_targets(atoms_base,use_derivatives=use_derivatives))
        return y_base

    def not_masked_reshape(self,array,not_masked,natoms,**kwargs):
        " Reshape an array so that it works for all atom coordinates and set constrained indicies to 0. "
        full_array=np.zeros((natoms,3))
        full_array[not_masked]=array.reshape(-1,3)
        return full_array
    
    def get_data(self,**kwargs):
        " Get data from the data base. "
        features=self.database.get_features()
        targets=self.database.get_targets()
        return features,targets
    
    def make_targets(self,atoms,use_derivatives=True,**kwargs):
        " Make the target in the data base. "
        return self.database.make_target(atoms,use_derivatives=use_derivatives,use_negative_forces=True)
    
    def get_constraints(self,atoms,**kwargs):
        " Get the number of atoms and the indicies of the atoms without constraints. "
        natoms=len(atoms)
        not_masked=self.database.get_constraints(atoms,**kwargs)
        return natoms,not_masked
    
    def check_attributes(self):
        " Check if all attributes agree between the class and subclasses. "
        if self.model.kernel.use_fingerprint!=self.database.use_fingerprint:
            raise Exception('Model and Database do not agree whether to use fingerprints!')
        if self.model.use_derivatives!=self.database.use_derivatives:
            raise Exception('Model and Database do not agree whether to use derivatives/forces!')
        return True

    def get_arguments(self):
        " Get the arguments of the class itself. "
        # Get the arguments given to the class in the initialization
        arg_kwargs=dict(model=self.model,
                        database=self.database,
                        baseline=self.baseline,
                        optimize=self.optimize,
                        hp=self.hp,
                        pdis=self.pdis,
                        verbose=self.verbose)
        # Get the constants made within the class
        constant_kwargs=dict()
        # Get the objects made within the class
        object_kwargs=dict()
        return arg_kwargs,constant_kwargs,object_kwargs

    def copy(self):
        " Copy the object. "
        # Get all arguments
        arg_kwargs,constant_kwargs,object_kwargs=self.get_arguments()
        # Make a clone
        clone=self.__class__(**arg_kwargs)
        # Check if constants have to be saved
        if len(constant_kwargs.keys()):
            for key,value in constant_kwargs.items():
                clone.__dict__[key]=value
        # Check if objects have to be saved
        if len(object_kwargs.keys()):
            for key,value in object_kwargs.items():
                clone.__dict__[key]=value.copy()
        return clone
    
    def __repr__(self):
        arg_kwargs=self.get_arguments()[0]
        str_kwargs=",".join([f"{key}={value}" for key,value in arg_kwargs.items()])
        return "{}({})".format(self.__class__.__name__,str_kwargs)
    


def get_default_model(model='tp',prior='median',use_derivatives=True,use_fingerprint=False,parallel=False,**kwargs):
    """
    Get the default ML model from the simple given arguments.

    Parameters:
        model : str
            Either the tp that gives the Studen T process or gp that gives the Gaussian process.
        prior : str
            Specify what prior mean should be used.
        use_derivatives : bool
            Whether to use derivatives of the targets.
        use_fingerprint : bool
            Whether to use fingerprints for the features.
            This has to be the same as for the database!
        parallel : bool
            Whether to optimize the hyperparameters in parallel.

    Returns:
        model : Model
            The Machine Learning Model with kernel and prior that are optimized.
    """
    # Make the prior mean from given string
    if isinstance(prior,str):
        if prior.lower()=='median':
            from ..means.median import Prior_median
            prior=Prior_median()
        elif prior.lower()=='mean':
            from ..means.mean import Prior_mean
            prior=Prior_mean()
        elif prior.lower()=='min':
            from ..means.min import Prior_min
            prior=Prior_min()
        elif prior.lower()=='max':
            from ..means.max import Prior_max
            prior=Prior_max()
    # Construct the kernel class object
    from ..kernel.se import SE
    kernel=SE(use_fingerprint=use_fingerprint,use_derivatives=use_derivatives)
    # Set the hyperparameter optimization method with or without parallelization
    from ..optimizers.globaloptimizer import FactorizedOptimizer
    if parallel:
        from ..optimizers.linesearcher import FineGridSearch
        line_optimizer=FineGridSearch(optimize=True,multiple_min=False,ngrid=80,loops=3,parallel=True)
    else:
        from ..optimizers.linesearcher import GoldenSearch
        line_optimizer=GoldenSearch(optimize=True,multiple_min=False,parallel=False)
    optimizer=FactorizedOptimizer(line_optimizer=line_optimizer,ngrid=80,calculate_init=False,parallel=parallel)
    # Use either the Student t process or the Gaussian process
    if isinstance(model,str):
        from ..hpfitter import HyperparameterFitter    
        if model.lower()=='tp':
            from ..models.tp import TProcess
            from ..objectivefunctions.tp.factorized_likelihood import FactorizedLogLikelihood
            hpfitter=HyperparameterFitter(func=FactorizedLogLikelihood(),optimizer=optimizer)
            model=TProcess(prior=prior,kernel=kernel,use_derivatives=use_derivatives,hpfitter=hpfitter,a=1e-3,b=1e-4)
        else:
            from ..models.gp import GaussianProcess
            from ..objectivefunctions.gp.factorized_likelihood import FactorizedLogLikelihood
            hpfitter=HyperparameterFitter(func=FactorizedLogLikelihood(),optimizer=optimizer)
            model=GaussianProcess(prior=prior,kernel=kernel,use_derivatives=use_derivatives,hpfitter=hpfitter)
    return model


def get_default_database(fp=None,use_derivatives=True,database_reduction=False,database_reduction_kwargs={},**kwargs):
    """
    Get the default Database from the simple given arguments.

    Parameters:
        fp : Fingerprint class object or None
            The fingerprint object used to generate the fingerprints.
            Cartesian coordinates are used if it is None.
        use_derivatives : bool
            Whether to use derivatives of the targets.
        database_reduction : bool
            Whether to used a reduced database after a number of training points.
        database_reduction_kwargs : dict
            A dictionary with the arguments for the reduced database if it is used.

    Returns:
        database : Database object
            The Database object with ASE atoms.
    """
    # Set a fingerprint
    if fp is None:
        from ..fingerprint.cartesian import Cartesian
        # Use cartesian coordinates as the fingerprint
        fp=Cartesian(reduce_dimensions=True,use_derivatives=use_derivatives,mic=True)
        use_fingerprint=False
    else:
        use_fingerprint=True
    # Make the data base ready
    if isinstance(database_reduction,str):
        data_kwargs=dict(reduce_dimensions=True,use_derivatives=use_derivatives,use_fingerprint=use_fingerprint,npoints=50,initial_indicies=[0,1],include_last=True)
        data_kwargs.update(database_reduction_kwargs)
        if database_reduction.lower()=='distance':
            from .database_reduction import DatabaseDistance
            database=DatabaseDistance(fingerprint=fp,**data_kwargs)
        elif database_reduction.lower()=='random':
            from .database_reduction import DatabaseRandom
            database=DatabaseRandom(fingerprint=fp,**data_kwargs)
        elif database_reduction.lower()=='hybrid':
            from .database_reduction import DatabaseHybrid
            database=DatabaseHybrid(fingerprint=fp,**data_kwargs)
        elif database_reduction.lower()=='min':
            from .database_reduction import DatabaseMin
            database=DatabaseMin(fingerprint=fp,**data_kwargs)
        elif database_reduction.lower()=='last':
            from .database_reduction import DatabaseLast
            database=DatabaseLast(fingerprint=fp,**data_kwargs)
        elif database_reduction.lower()=='restart':
            from .database_reduction import DatabaseRestart
            database=DatabaseRestart(fingerprint=fp,**data_kwargs)
        elif database_reduction.lower()=='interest':
            from .database_reduction import DatabasePointsInterest
            database=DatabasePointsInterest(fingerprint=fp,**data_kwargs)
        elif database_reduction.lower()=='each_interest':
            from .database_reduction import DatabasePointsInterestEach
            database=DatabasePointsInterestEach(fingerprint=fp,**data_kwargs)
    else:
        from .database import Database
        database=Database(fingerprint=fp,reduce_dimensions=True,use_derivatives=use_derivatives,use_fingerprint=use_fingerprint)
    return database


def get_default_mlmodel(model='tp',fp=None,baseline=None,prior='median',use_derivatives=True,parallel=False,use_pdis=True,database_reduction=False,database_reduction_kwargs={},verbose=False,**kwargs):
    """
    Get the default ML model with a database for the ASE Atoms from the simple given arguments.

    Parameters:
        model : str
            Either the tp that gives the Studen T process or gp that gives the Gaussian process.
        fp : Fingerprint class object or None
            The fingerprint object used to generate the fingerprints.
            Cartesian coordinates are used if it is None.
        baseline : Baseline object
            The Baseline object calculator that calculates energy and forces. 
        prior : str
            Specify what prior mean should be used.
        use_derivatives : bool
            Whether to use derivatives of the targets.
        parallel : bool
            Whether to optimize the hyperparameters in parallel.
        use_pdis : bool
            Whether to make prior distributions for the hyperparameters.
        database_reduction : bool
            Whether to used a reduced database after a number of training points.
        database_reduction_kwargs : dict
            A dictionary with the arguments for the reduced database if it is used.
        verbose : bool
            Whether to print statements in the optimization.
            
    Returns:
        mlmodel : MLModel class object
            Machine Learning model used for ASE Atoms and calculator.
    """
    # Check if fingerprints are used
    if fp is None:
        use_fingerprint=False
    else:
        use_fingerprint=True
    # Make the model
    if isinstance(model,str):
        model=get_default_model(model=model,prior=prior,use_derivatives=use_derivatives,use_fingerprint=use_fingerprint,parallel=parallel)
    # Make the database
    database=get_default_database(fp=fp,use_derivatives=use_derivatives,database_reduction=database_reduction,database_reduction_kwargs=database_reduction_kwargs)
    # Make prior distributions for the hyperparameters if specified
    if use_pdis:
        from ..pdistributions.normal import Normal_prior
        pdis=dict(length=Normal_prior(mu=[-0.5],std=[1.0]),noise=Normal_prior(mu=[-9.0],std=[1.0]))
    else:
        pdis=None
    # Make the ML model with database
    return MLModel(model=model,database=database,baseline=baseline,optimize=True,pdis=pdis,verbose=verbose)
