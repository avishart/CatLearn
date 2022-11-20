import numpy as np
from ase.neb import NEB
from ase.io import read,write
from ase.io.trajectory import TrajectoryWriter
from catlearn import __version__
from copy import deepcopy
import datetime
from mpi4py import MPI


class MLNEB(object):

    def __init__(self,start,end,mlcalc=None,ase_calc=None,acq=None,interpolation='idpp',interpolation_kwargs={},
                climb=True,neb_kwargs=dict(k=0.1,method='improvedtangent',remove_rotation_and_translation=False), 
                n_images=15,mic=False,prev_calculations=None,
                force_consistent=None,local_opt=None,local_opt_kwargs={},
                trainingset='evaluated_structures.traj',trajectory='MLNEB.traj',full_output=False):
        """ Nudged elastic band (NEB) with Machine Learning as active learning.
            Parameters:
                start: Atoms object with calculated energy or ASE Trajectory file.
                    Initial end-point of the NEB path.
                end: Atoms object with calculated energy or ASE Trajectory file.
                    Final end-point of the NEB path.
                mlcalc: ML-calculator Object.
                    The ML-calculator object used as surrogate surface. A default ML-model is used
                    if mlcalc is None.
                ase_calc: ASE calculator Object.
                    ASE calculator as implemented in ASE.
                    See https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html
                acq: Acquisition Object.
                    The Acquisition object used for calculating the acq. function and choose a candidate
                    to calculate next. A default Acquisition object is used if acq is None.
                interpolation: string or list of ASE Atoms or ASE Trajectory file.
                    Automatic interpolation can be done ('idpp' and 'linear' as
                    implemented in ASE).
                    See https://wiki.fysik.dtu.dk/ase/ase/neb.html.
                    Manual: Trajectory file (in ASE format) or list of Atoms.
                interpolation_kwargs: dict.
                    A dictionary with the arguments used in the interpolation.
                    See https://wiki.fysik.dtu.dk/ase/ase/neb.html. 
                climb : bool
                    Whether to use climbing image in the ML-NEB. It is strongly recommended to have climb=True. 
                    It is only activated when the uncertainty is low and a NEB without climbing image can converge.
                neb_kwargs: dict.
                    A dictionary with the arguments used in the NEB method. climb can not be included.
                    See https://wiki.fysik.dtu.dk/ase/ase/neb.html. 
                n_images: int.
                    Number of images of the path (if not included a path before).
                    The number of images include the 2 end-points of the NEB path.
                mic: boolean.
                    Use mic=True to use the Minimum Image Convention and calculate the
                    interpolation considering periodic boundary conditions.
                prev_calculations: Atoms list or ASE Trajectory file.
                    (optional) The user can feed previously calculated data for the
                    same hypersurface. The previous calculations must be fed as an
                    Atoms list or Trajectory file.
                force_consistent: boolean or None.
                    Use force-consistent energy calls (as opposed to the energy
                    extrapolated to 0 K). By default (force_consistent=None) uses
                    force-consistent energies if available in the calculator, but
                    falls back to force_consistent=False if not.
                local_opt: ASE local optimizer Object. 
                    A local optimizer object from ASE. If None is given then MDMin is used.
                local_opt_kwargs: dict
                    Arguments used for the ASE local optimizer.
                trainingset: string.
                    Trajectory filename to store the evaluated training data.
                trajectory: string
                    Trajectory filename to store the predicted NEB path.
                full_output: boolean
                    Whether to print on screen the full output (True) or not (False).
        """
        # Setup parallelization
        self.parallel_setup()
        # NEB parameters
        self.interpolation=interpolation
        self.interpolation_kwargs=interpolation_kwargs.copy()
        self.n_images=n_images
        self.mic=mic
        self.climb=climb
        self.neb_kwargs=neb_kwargs.copy()
        # Whether to have the full output
        self.full_output=full_output  
        # Setup the ML calculator
        if mlcalc is None:
            mlcalc=self.get_default_mlcalc()
        self.mlcalc=deepcopy(mlcalc)
        # Select an acquisition function 
        if acq is None:
            from .acquisition import Acquisition
            acq=Acquisition(mode='ume',objective='max',kappa=2,unc_convergence=0.05)
        self.acq=deepcopy(acq)
        # Save initial and final state
        self.set_up_endpoints(start,end)
        # Save the ASE calculator
        self.ase_calc=ase_calc
        self.force_consistent=force_consistent
        ## Save local optimizer
        if local_opt is None:
            from ase.optimize import MDMin
            local_opt=MDMin
            local_opt_kwargs=dict(dt=0.025,trajectory='surrogate_neb.traj')
        self.local_opt=local_opt
        self.local_opt_kwargs=local_opt_kwargs
        # Trajectories
        self.trainingset=trainingset
        self.trajectory=trajectory
        # Load previous calculations to the ML model
        self.use_prev_calculations(prev_calculations)
              

    def run(self,fmax=0.05,unc_convergence=0.025,steps=500,ml_steps=750,max_unc=0.25):
        """ Run the active learning NEB process. 
            Parameters:
                fmax : float
                    Convergence criteria (in eV/Angs).
                unc_convergence: float
                    Maximum uncertainty for convergence (in eV).
                steps : int
                    Maximum number of evaluations.
                ml_steps: int
                    Maximum number of steps for the NEB optimization on the
                    predicted landscape.
                max_unc: float
                    Early stopping criteria. Maximum uncertainty before stopping the
                    optimization on the surrogate surface.
        """
        # Active learning parameters
        candidate=None
        self.acq.unc_convergence=unc_convergence
        self.steps=0
        self.trajectory_neb=TrajectoryWriter(self.trajectory,mode='a')
        # Calculate a extra data point if only start and end is given
        self.extra_initial_data()
        # Run the active learning
        while True:
            self.steps+=1
            # Train and optimize ML model
            self.ml_optimize()
            # Perform NEB on ML surrogate surface
            candidate=self.run_mlneb(fmax=fmax*0.8,ml_steps=ml_steps,max_unc=max_unc)
            # Evaluate candidate
            self.evaluate(candidate)
            # Print the results for this iteration
            self.print_neb()
            # Check convergence
            converged=self.check_convergence(fmax,unc_convergence)
            if converged:
                break
            if self.steps>=steps:
                self.message_system('MLNEB did not converge!')
                break
        self.trajectory_neb.close()
        return self

    def set_up_endpoints(self,start,end):
        " Load and calculate the intial and final states"
        # Load initial and final states
        if isinstance(start, str):
            start=read(start)
        if isinstance(end, str):
            end=read(end)
        # Add initial and final states to ML model
        self.add_training([start,end])
        # Store the initial and final energy
        self.start_energy=start.get_potential_energy()
        self.end_energy=end.get_potential_energy()
        self.start=start.copy()
        self.end=end.copy()
        return 

    def use_prev_calculations(self,prev_calculations):
        " Use previous calculations to restart ML calculator."
        if prev_calculations is None:
            return
        if isinstance(prev_calculations,str):
            prev_calculations=read(prev_calculations,':')
        # Add calculations to the ML model
        self.add_training(prev_calculations)
        return

    def make_interpolation(self,interpolation='idpp'):
        " Make the NEB interpolation path "
        # Use a premade interpolation path
        if isinstance(interpolation,(list,np.ndarray)):
            images=interpolation.copy()
        else:
            if interpolation in ['linear','idpp']:
                # Make path by the NEB methods interpolation
                images=[self.start.copy() for i in range(self.n_images-1)]+[self.end.copy()]
                neb=NEB(images,**self.neb_kwargs)
                if interpolation=='linear':
                    neb.interpolate(mic=self.mic,**self.interpolation_kwargs)
                elif interpolation=='idpp':
                    neb.interpolate(method='idpp',mic=self.mic,**self.interpolation_kwargs)
            else:
                images=read(interpolation,':')
        # Attach the ML calculator to all images
        images=self.attach_mlcalc(images)
        return images

    def attach_mlcalc(self,imgs):
        " Attach the ML calculator to the given images. "
        images=[]
        for img in imgs:
            image=img.copy()
            image.calc=deepcopy(self.mlcalc)
            images.append(image)
        return images

    def parallel_setup(self):
        " Setup the parallelization. "
        self.comm = MPI.COMM_WORLD
        self.rank,self.size=self.comm.Get_rank(),self.comm.Get_size()
        return

    def evaluate(self,candidate):
        " Evaluate the ASE atoms with the ASE calculator. "
        self.message_system('Performing evaluation.',end='\r')
        # Broadcast the system to all cpus
        if self.rank==0:
            candidate=candidate.copy()
        candidate=self.comm.bcast(candidate,root=0)
        self.comm.barrier()
        # Calculate the energies and forces
        candidate.calc=self.ase_calc
        forces=candidate.get_forces()
        self.energy=candidate.get_potential_energy(force_consistent=self.force_consistent)
        self.max_abs_forces=np.max(np.linalg.norm(forces,axis=1))
        self.message_system('Single-point calculation finished.')
        # Store the data
        self.add_training([candidate])
        self.mlcalc.mlmodel.database.save_data()
        return

    def add_training(self,atoms_list):
        " Add atoms_list data to ML model on rank=0. "
        if self.rank==0:
            self.mlcalc.mlmodel.add_training(atoms_list)
        return

    def ml_optimize(self):
        " Train the ML model "
        if self.rank==0:
            self.mlcalc.mlmodel.train_model(verbose=self.full_output)
        return

    def extra_initial_data(self):
        " If only initial and final state is given then a third data point is calculated. "
        candidate=None
        if self.rank==0:
            if len(self.mlcalc.mlmodel.database)==2:
                images=self.make_interpolation(interpolation=self.interpolation)
                candidate=images[1+int((self.n_images-2)/3.0)].copy()
        candidate=self.comm.bcast(candidate,root=0)
        if candidate is not None:
            self.evaluate(candidate)
        return candidate

    def run_mlneb(self,fmax=0.05,ml_steps=750,max_unc=0.25):
        " Run the NEB on the ML surrogate surface"
        if self.rank==0:
            # Make the interpolation from the initial points
            images=self.make_interpolation(interpolation=self.interpolation)
            if self.get_fmax_predictions(images)<1e-14:
                self.message_system('Too low forces on initial path!')
                candidate=self.choose_candidate(images)
                return candidate
            # Run the NEB on the surrogate surface
            self.message_system('Starting NEB without climbing image on surrogate surface.')
            images=self.mlneb_opt(images,fmax=fmax,ml_steps=ml_steps,max_unc=max_unc,climb=False)
            self.save_mlneb(images)
            # Get the candidate
            candidate=self.choose_candidate(images)
            return candidate
        return None

    def get_predictions(self,images):
        " Calculate the energies and uncertainties with the ML calculator "
        energies=[image.get_potential_energy() for image in images]
        uncertainties=[image.calc.results['uncertainty'] for image in images]
        return np.array(energies),np.array(uncertainties)

    def get_fmax_predictions(self,images):
        " Calculate the maximum force with the ML calculator "
        forces=np.array([image.get_forces() for image in images])
        return np.nanmax(np.linalg.norm(forces,axis=1))

    def choose_candidate(self,images):
        " Use acquisition functions to chose the next training point "
        # Get the energies and uncertainties
        energy_path,unc_path=self.get_predictions(images)
        self.emax_ml=np.nanmax(energy_path)
        self.umax_ml=np.nanmax(unc_path)
        self.umean_ml=np.mean(unc_path)
        # Calculate the acquisition function for each image
        acq_values=self.acq.calculate(energy_path[1:-1],unc_path[1:-1])
        # Chose the maximum value given by the Acq. class
        i_sort=self.acq.choose(acq_values)
        # The next training point
        image=images[1+i_sort[0]]
        self.energy_pred=image.get_potential_energy()
        return image.copy()

    def mlneb_opt(self,images,fmax=0.05,ml_steps=750,max_unc=0.25,climb=False):
        " Run the ML NEB with checking uncertainties if selected. "
        neb=NEB(images,climb=climb,**self.neb_kwargs)
        neb_opt=self.local_opt(neb,**self.local_opt_kwargs)
        # Run the ML NEB fully without consider the uncertainty
        if max_unc==False:
            neb_opt.run(fmax=fmax*0.8,steps=ml_steps)
            return images
        # Stop the ML NEB if the uncertainty becomes too large
        for i in range(1,ml_steps+1):
            # Run the NEB on the surrogate surface
            neb_opt.run(fmax=fmax,steps=i)
            #neb_opt.nsteps = 0
            energy_path,unc_path=self.get_predictions(images)
            if np.max(unc_path)>=max_unc:
                self.message_system('NEB on surrogate surface stopped due to high uncertainty!')
                break
            if np.isnan(energy_path).any():
                images=self.make_interpolation(interpolation=self.interpolation)
                self.message_system('Stopped due to NaN value in prediction!')
                break
            if neb_opt.converged():
                self.message_system('NEB on surrogate surface converged!',end='\r')
                break
        # Activate climbing when the model has low uncertainty and it is converged
        if neb_opt.converged():
            if climb==False and self.climb==True:
                self.message_system('Starting NEB with climbing image on surrogate surface.')
                return self.mlneb_opt(images,fmax=fmax,ml_steps=ml_steps-neb_opt.nsteps,max_unc=max_unc,climb=True)
            self.message_system('NEB on surrogate surface converged!')
        return images

    def save_mlneb(self,images):
        " Save the ML NEB result in the trajectory. "
        for image in images:
            self.trajectory_neb.write(self.mlcalc.mlmodel.database.copy_atoms(image))
        self.images=deepcopy(images)
        return 

    def message_system(self,message,obj=None,end='\n'):
        " Print output on rank=0. "
        if self.full_output is True:
            if self.rank==0:
                if obj is None:
                    print(message,end=end)
                else:
                    print(message,obj,end=end)
        return

    def print_neb(self):
        " Print the NEB process as a table "
        if self.rank==0:
            now=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            try:
                len(self.print_neb_list)
            except:
                self.print_neb_list=['| Step |        Time         | Pred. barrier (-->) | Pred. barrier (<--) | Max. uncert. | Avg. uncert. |   fmax   |']
            msg='|{0:6d}| '.format(self.steps)+'{} |'.format(now)
            msg+='{0:21f}|'.format(self.emax_ml-self.start_energy)+'{0:21f}|'.format(self.emax_ml-self.end_energy)
            msg+='{0:14f}|'.format(self.umax_ml)
            msg+='{0:14f}|'.format(np.mean(self.umean_ml))+'{0:10f}|'.format(self.max_abs_forces)
            self.print_neb_list.append(msg)
            msg='\n'.join(self.print_neb_list)
            self.message_system(msg)
        return

    def check_convergence(self,fmax,unc_convergence):
        " Check if the ML-NEB is converged to the final path with low uncertainty "
        converged=False
        if self.rank==0:
            # Check the force criterion is met
            if self.max_abs_forces<=fmax and self.umax_ml<=unc_convergence:
                if np.abs(self.energy_pred-self.energy)<=unc_convergence:
                    self.message_system("Congratulations! Your ML NEB is converged.") 
                    self.print_cite()
                    converged=True
        converged=self.comm.bcast(converged,root=0)
        return converged

    def print_cite(self):
        msg = "\n" + "-" * 79 + "\n"
        msg += "You are using AIDNEB. Please cite: \n"
        msg += "[1] J. A. Garrido Torres, M. H. Hansen, P. C. Jennings, "
        msg += "J. R. Boes and T. Bligaard. Phys. Rev. Lett. 122, 156001. "
        msg += "https://doi.org/10.1103/PhysRevLett.122.156001 \n"
        msg += "[2] O. Koistinen, F. B. Dagbjartsdottir, V. Asgeirsson, A. Vehtari"
        msg += " and H. Jonsson. J. Chem. Phys. 147, 152720. "
        msg += "https://doi.org/10.1063/1.4986787 \n"
        msg += "[3] E. Garijo del Rio, J. J. Mortensen and K. W. Jacobsen. "
        msg += "Phys. Rev. B 100, 104103."
        msg += "https://doi.org/10.1103/PhysRevB.100.104103. \n"
        msg += "-" * 79 + '\n'
        self.message_system(msg)
        return 

    def get_default_mlcalc(self):
        " Get a default ML calculator if a calculator is not given. This is a recommended ML calculator."
        from ..regression.gaussianprocess.calculator.mlcalc import MLCalculator
        from ..regression.gaussianprocess.calculator.mlmodel import MLModel
        from ..regression.gaussianprocess.gp.gp import GaussianProcess
        from ..regression.gaussianprocess.kernel.se import SE,SE_Derivative
        from ..regression.gaussianprocess.means.median import Prior_median
        from ..regression.gaussianprocess.hpfitter import HyperparameterFitter
        from ..regression.gaussianprocess.objectfunctions.factorized_likelihood import FactorizedLogLikelihood
        from ..regression.gaussianprocess.optimizers import run_golden,line_search_scale
        from ..regression.gaussianprocess.calculator.database import Database
        from ..regression.gaussianprocess.fingerprint.cartesian import Cartesian
        from ..regression.gaussianprocess.pdistributions import Normal_prior
        use_derivatives=True
        use_fingerprint=False
        # Use a GP as the model 
        local_kwargs=dict(tol=1e-5,optimize=True,multiple_max=True)
        kwargs_optimize=dict(local_run=run_golden,maxiter=5000,jac=False,bounds=None,ngrid=80,use_bounds=True,local_kwargs=local_kwargs)
        hpfitter=HyperparameterFitter(FactorizedLogLikelihood(),optimization_method=line_search_scale,opt_kwargs=kwargs_optimize,distance_matrix=True)
        kernel=SE_Derivative(use_fingerprint=use_fingerprint) if use_derivatives else SE(use_fingerprint=use_fingerprint)
        model=GaussianProcess(prior=Prior_median(),kernel=kernel,use_derivatives=use_derivatives,hpfitter=hpfitter)
        # Use cartesian coordinates and make the database ready
        fp=Cartesian(reduce_dimensions=True,use_derivatives=use_derivatives,mic=self.mic)
        database=Database(fingerprint=fp,reduce_dimensions=True,use_derivatives=use_derivatives,negative_forces=True,use_fingerprint=use_fingerprint)
        # Make prior distributions for hyperparameters
        prior=dict(length=np.array([Normal_prior(0.0,8.0)]),noise=np.array([Normal_prior(-14.0,14.0)]))
        # Make the ML model with model and database
        ml_opt_kwargs=dict(retrain=True,prior=prior)
        mlmodel=MLModel(model=model,database=database,baseline=None,optimize=True,optimize_kwargs=ml_opt_kwargs)
        # Finally make the calculator
        mlcalc=MLCalculator(mlmodel=mlmodel,calculate_uncertainty=True)
        return mlcalc
