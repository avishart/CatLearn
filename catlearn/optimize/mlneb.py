import numpy as np
from ase.neb import NEB
from ase.io import read
from ase.io.trajectory import TrajectoryWriter
from ase.parallel import parprint
from catlearn import __version__
from copy import deepcopy
import datetime
from mpi4py import MPI



class MLNEB(object):

    def __init__(self,start,end,mlcalc=None,ase_calc=None,acq=None,interpolation='idpp',interpolation_kwargs={},
                neb_kwargs=dict(k=0.1,climb=False,method='improvedtangent',remove_rotation_and_translation=False), 
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
                neb_kwargs: dict.
                    A dictionary with the arguments used in the NEB method.
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
        # Setup the ML calculator
        if mlcalc is None:
            from ..regression.gaussianprocess.calculator.mlcalc import MLCalculator
            mlcalc=MLCalculator(model=None,calculate_uncertainty=True)
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
        # NEB parameters
        self.interpolation=interpolation
        self.interpolation_kwargs=interpolation_kwargs.copy()
        self.n_images=n_images
        self.mic=mic
        self.neb_kwargs=neb_kwargs.copy()
        ## Save local optimizer
        if local_opt is None:
            from ase.optimize import MDMin
            local_opt=MDMin
            local_opt_kwargs=dict(dt=0.025,trajectory='surrogate_neb.traj')
        self.local_opt=local_opt
        self.local_opt_kwargs=local_opt_kwargs
        # Trajectories
        self.trainingset=trainingset
        self.trajectory=TrajectoryWriter(trajectory)
        # Load previous calculations to the ML model
        self.use_prev_calculations(prev_calculations)
        # Whether to have the full output
        self.full_output=full_output        

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
        self.step=0
        # Calculate a extra data point if only start and end is given
        self.extra_initial_data()
        # Run the active learning
        while True:
            # Train and optimize ML model
            self.ml_optimize()
            # Perform NEB on ML surrogate surface
            candidate=self.run_mlneb(fmax=fmax*0.8,ml_steps=ml_steps,max_unc=max_unc)
            # Evaluate candidate
            self.evaluate(candidate)
            # Print the results for this iteration
            self.print_neb()
            # Check convergence
            self.check_convergence(self,fmax,unc_convergence)
            if self.steps>=steps:
                self.message_system('MLNEB did not converge!')
                break
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
        self.mlcalc.model.add_training(prev_calculations)
        return

    def make_interpolation(self,interpolation='idpp'):
        " Make the NEB interpolation path "
        # Use a premade interpolation path
        if isinstance(interpolation,(list,np.ndarray)):
            images=interpolation.copy()
        else:
            if interpolation in ['linear','idpp']:
                # Make path by the NEB methods interpolation
                images=[self.start]*(self.n_images-1)+[self.end]
                neb=NEB(images,**self.neb_kwargs)
                if interpolation=='linear':
                    neb.interpolate(mic=self.mic,**self.interpolation_kwargs)
                elif interpolation=='idpp':
                    neb.idpp_interpolate(mic=self.mic,**self.interpolation_kwargs)
            else:
                images=read(interpolation,':')
        # Attach the ML calculator to all images
        for image in images:
            image.calc=deepcopy(self.mlcalc)
        return images

    def parallel_setup(self):
        " Setup the parallelization. "
        self.comm = MPI.COMM_WORLD
        self.rank,self.size=self.comm.Get_rank(),self.comm.Get_size()
        return

    def message_system(self,message,obj=None):
        " Print output on rank=0. "
        if self.full_output is True:
            if self.rank==0:
                if obj is None:
                    print(message)
                else:
                    print(message,obj)
        return

    def evaluate(self,candidate):
        " Evaluate the ASE atoms with the ASE calculator. "
        self.message_system('Performing evaluation.')
        # Broadcast the system to all cpus
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
        self.mlcalc.model.database.save_data()
        self.step+=1
        return

    def add_training(self,atoms_list):
        " Add atoms_list data to ML model on rank=0. "
        if self.rank==0:
            self.mlcalc.model.add_training(atoms_list)
        return

    def ml_optimize(self):
        " Train the ML model "
        if self.rank==0:
            self.mlcalc.model.train_model()
        return

    def extra_initial_data(self):
        " If only initial and final state is given then a third data point is calculated. "
        candidate=None
        if self.rank==0:
            if len(self.mlcalc.model.database)==2:
                images=self.make_interpolation(interpolation=self.interpolation)
                candidate=images[1+int((self.n_images-2)/3.0)].copy()
        candidate=self.comm.bcast(candidate,root=0)
        print('extra point',self.rank)
        if candidate is not None:
            self.evaluate(candidate)
        return candidate

    def run_mlneb(self,fmax=0.05,ml_steps=750,max_unc=0.25):
        " Run the NEB on the ML surrogate surface"
        if self.rank==0:
            # Make the interpolation from the initial points
            images=self.make_interpolation(interpolation=self.interpolation)
            # Run the NEB on the surrogate surface
            images=self.mlneb_opt(images,fmax=fmax,ml_steps=ml_steps,max_unc=max_unc)
            self.save_mlneb(images)
            # Get the energies and uncertainties
            energy_path,unc_path=self.get_predictions(images)
            self.emax_ml=np.nanmax(energy_path)
            self.umax_ml=np.nanmax(unc_path)
            self.umean_ml=np.mean(unc_path)
            # Get the candidate
            candidate=self.choose_candidate(images,energy_path,unc_path)
            return candidate
        return None

    def get_predictions(self,images):
        " Calculate the energies and uncertainties with the ML calculator "
        energies=[image.get_potential_energy() for image in images]
        uncertainties=[image.calc.results['uncertainty'] for image in images]
        return np.array(energies),np.array(uncertainties)

    def choose_candidate(self,images,energy_path,unc_path):
        " Use acquisition functions to chose the next training point "
        # Calculate the acquisition function for each image
        acq_values=self.acq.calculate(energy_path[1:-1],unc_path[1:-1])
        # Chose the maximum value given by the Acq. class
        i_sort=self.acq.choose(acq_values)
        # The next training point
        image=images[1+i_sort[0]]
        self.energy_pred=image.get_potential_energy()
        return image.copy()

    def mlneb_opt(self,images,fmax=0.05,ml_steps=750,max_unc=0.25):
        " Run the ML NEB with checking uncertainties if selected. "
        neb=NEB(images,**self.neb_kwargs)
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
                break
            if neb_opt.converged():
                break
        return images

    def save_mlneb(self,images):
        " Save the ML NEB result in the trajectory. "
        for image in images:
            self.trajectory.write(self.mlcalc.model.database.copy_atoms(image))
        return 

    def print_neb(self):
        " Print the NEB process as a table "
        if self.rank==0:
            now=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            try:
                len(self.print_neb_list)
            except:
                self.print_neb_list=['| Step |        Time         | Pred. barrier (-->) | Pred. barrier (<--) | Max. uncert. | Avg. uncert. |   fmax   |']
            msg='|{0:6d}| '.format(self.step)+'{} |'.format(now)
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

