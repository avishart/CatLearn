import numpy as np
from ase.neb import NEB
from ase.io import read
from ase.io.trajectory import TrajectoryWriter
from copy import deepcopy
from ase.parallel import world,broadcast
import datetime

class MLNEB(object):

    def __init__(self,start,end,ase_calc,mlcalc=None,acq=None,interpolation='idpp',interpolation_kwargs=dict(),
                 climb=True,neb_kwargs=dict(),n_images=15,prev_calculations=None,
                 use_restart_path=True,check_path_unc=True,save_memory=False,
                 force_consistent=None,local_opt=None,local_opt_kwargs=dict(),
                 trainingset='evaluated_structures.traj',trajectory='MLNEB.traj',tabletxt=None,full_output=False,**kwargs):
        """ 
        Nudged elastic band (NEB) with Machine Learning as active learning.

        Parameters:
            start: Atoms object with calculated energy or ASE Trajectory file.
                Initial end-point of the NEB path.
            end: Atoms object with calculated energy or ASE Trajectory file.
                Final end-point of the NEB path.
            ase_calc: ASE calculator Object.
                ASE calculator as implemented in ASE.
                See https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html
            mlcalc: ML-calculator Object.
                The ML-calculator object used as surrogate surface. A default ML-model is used
                if mlcalc is None.
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
            prev_calculations: Atoms list or ASE Trajectory file.
                (optional) The user can feed previously calculated data for the
                same hypersurface. The previous calculations must be fed as an
                Atoms list or Trajectory file.
            use_restart_path: bool
                Use the path from last robust iteration (low uncertainty).
            check_path_unc: bool
                Check if the uncertainty is large for the restarted path and
                if it is then use the initial interpolation.
            save_memory: bool
                Whether to only train the ML calculator and store all objects on one CPU. 
                If save_memory==True then parallel optimization of the hyperparameters can not be achived.
                If save_memory==False no MPI object is used.  
            force_consistent: boolean or None.
                Use force-consistent energy calls (as opposed to the energy
                extrapolated to 0 K). By default (force_consistent=None) uses
                force-consistent energies if available in the calculator, but
                falls back to force_consistent=False if not.
            local_opt: ASE local optimizer Object. 
                A local optimizer object from ASE. If None is given then FIRE is used.
            local_opt_kwargs: dict
                Arguments used for the ASE local optimizer.
            trainingset: string.
                Trajectory filename to store the evaluated training data.
            trajectory: string
                Trajectory filename to store the predicted NEB path.
            tabletxt: string
                Name of the .txt file where the summary table is printed. 
                It is not saved to the file if tabletxt=None.
            full_output: boolean
                Whether to print on screen the full output (True) or not (False).
        """
        # Setup parallelization
        self.parallel_setup(save_memory)
        # NEB parameters
        self.interpolation=interpolation
        self.interpolation_kwargs=dict(mic=True)
        self.interpolation_kwargs.update(interpolation_kwargs)
        self.n_images=n_images
        self.climb=climb
        self.neb_kwargs=dict(k=3.0,method='improvedtangent',remove_rotation_and_translation=False)
        self.neb_kwargs.update(neb_kwargs)
        # General parameter settings
        self.use_restart_path=use_restart_path
        self.check_path_unc=check_path_unc
        # Set initial parameters
        self.step=0
        self.converging=False
        # Whether to have the full output
        self.full_output=full_output  
        # Setup the ML calculator
        if mlcalc is None:
            from ..regression.gaussianprocess.calculator.mlmodel import get_default_mlmodel
            from ..regression.gaussianprocess.calculator.mlcalc import MLCalculator
            mlmodel=get_default_mlmodel(model='tp',prior='max',baseline=None,use_derivatives=True,parallel=(not save_memory),database_reduction=False)
            self.mlcalc=MLCalculator(mlmodel=mlmodel)
        else:
            self.mlcalc=mlcalc.copy()
        self.set_verbose(verbose=full_output)
        # Select an acquisition function 
        if acq is None:
            from .acquisition import AcqUME
            self.acq=AcqUME(objective='max',unc_convergence=0.05)
        else:
            self.acq=acq.copy()
        # Save initial and final state
        self.set_up_endpoints(start,end)
        # Save the ASE calculator
        self.ase_calc=ase_calc
        self.force_consistent=force_consistent
        # Save local optimizer
        local_opt_kwargs_default=dict(trajectory='surrogate_neb.traj')
        if local_opt is None:
            from ase.optimize import MDMin
            local_opt=MDMin
            local_opt_kwargs_default.update(dict(dt=0.01))
        self.local_opt=local_opt
        local_opt_kwargs_default.update(local_opt_kwargs)
        self.local_opt_kwargs=local_opt_kwargs_default.copy()
        # Trajectories
        self.trainingset=trainingset
        self.trajectory=trajectory
        # Summary table file name
        self.tabletxt=tabletxt
        # Load previous calculations to the ML model
        self.use_prev_calculations(prev_calculations)
              

    def run(self,fmax=0.05,unc_convergence=0.05,steps=500,ml_steps=750,max_unc=0.05,**kwargs):
        """ 
        Run the active learning NEB process. 

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
        self.acq.set_parameters(unc_convergence=unc_convergence)
        self.trajectory_neb=TrajectoryWriter(self.trajectory,mode='w',properties=['energy','forces'])
        # Calculate a extra data point if only start and end is given
        self.extra_initial_data()
        # Define the last images that can be used to restart the interpolation
        self.last_images=self.make_interpolation(interpolation=self.interpolation)
        self.last_images_tmp=None
        # Run the active learning
        for step in range(1,steps+1):
            # Train and optimize ML model
            self.train_mlmodel()
            # Perform NEB on ML surrogate surface
            #max_u=((max_unc*(self.steps-1))+unc_convergence)/self.steps
            candidate,neb_converged=self.run_mlneb(fmax=fmax*0.8,ml_steps=ml_steps,max_unc=max_unc)
            # Evaluate candidate
            self.evaluate(candidate)
            # Print the results for this iteration
            self.print_statement(step)
            # Check convergence
            self.converging=self.check_convergence(fmax,unc_convergence,neb_converged)
            if self.converging:
                break
        if self.converging==False:
            self.message_system('MLNEB did not converge!')
        self.trajectory_neb.close()
        return self

    def set_up_endpoints(self,start,end,**kwargs):
        " Load and calculate the intial and final states"
        # Load initial and final states
        if isinstance(start,str):
            start=read(start)
        if isinstance(end,str):
            end=read(end)
        # Add initial and final states to ML model
        self.add_training([start,end])
        # Store the initial and final energy
        self.start_energy=start.get_potential_energy()
        self.end_energy=end.get_potential_energy()
        self.start=start.copy()
        self.end=end.copy()
        return 

    def use_prev_calculations(self,prev_calculations,**kwargs):
        " Use previous calculations to restart ML calculator."
        if prev_calculations is None:
            return
        if isinstance(prev_calculations,str):
            prev_calculations=read(prev_calculations,':')
        # Add calculations to the ML model
        self.add_training(prev_calculations)
        return

    def make_interpolation(self,interpolation='idpp',**kwargs):
        " Make the NEB interpolation path "
        # Use a premade interpolation path
        if isinstance(interpolation,(list,np.ndarray)):
            images=interpolation.copy()
        else:
            if interpolation in ['linear','idpp']:
                # Make path by the NEB methods interpolation
                images=[self.start.copy() for i in range(self.n_images-1)]+[self.end.copy()]
                neb=NEB(images,**self.neb_kwargs)
                if interpolation.lower()=='linear':
                    neb.interpolate(**self.interpolation_kwargs)
                elif interpolation.lower()=='idpp':
                    neb.interpolate(method='idpp',**self.interpolation_kwargs)
            else:
                # Import interpolation from a trajectory file
                images=read(interpolation,'-{}:'.format(self.n_images))
        # Attach the ML calculator to all images
        images=self.attach_mlcalc(images)
        return images
    
    def make_reused_interpolation(self,max_unc,**kwargs):
        " Make the NEB interpolation path or use the previous path if it has low uncertainty. "
        # Make the interpolation from the initial points
        if not self.use_restart_path or self.last_images_tmp is None:
            self.message_system('The initial interpolation is used as the initial path!')
            return self.make_interpolation(interpolation=self.interpolation)
        else:
            # Reuse the previous path 
            images=self.make_interpolation(interpolation=self.last_images_tmp)
            if self.check_path_unc:
                # Check if the uncertainty is too large
                if np.nanmax(self.get_predictions(images)[1])>=max_unc:
                    self.last_images_tmp=None
                    self.message_system('The previous last path is used as the initial path due to uncertainty!')
                    return self.make_interpolation(interpolation=self.last_images)
                else:
                    # Check if the perpendicular forces are less for the new path
                    images_last=self.make_interpolation(interpolation=self.last_images)
                    if self.get_fmax_predictions(images)<=self.get_fmax_predictions(images_last):
                        # The last path is used as a stable path in the future
                        self.message_system('The last path is used as the initial path!')
                        self.last_images=self.last_images_tmp.copy()
                    else:
                        self.message_system('The previous last path is used as the initial path due to fmax!')
                        self.last_images_tmp=None
                        return images_last
            else:
                self.message_system('The last path is used as the initial path!')
        return images

    def attach_mlcalc(self,imgs,**kwargs):
        " Attach the ML calculator to the given images. "
        images=[]
        for img in imgs:
            image=img.copy()
            image.calc=self.mlcalc.copy()
            images.append(image)
        return images

    def parallel_setup(self,save_memory=False,**kwargs):
        " Setup the parallelization. "
        self.save_memory=save_memory
        self.rank=world.rank
        self.size=world.size
        return self

    def evaluate(self,candidate,**kwargs):
        " Evaluate the ASE atoms with the ASE calculator. "
        self.message_system('Performing evaluation.',end='\r')
        # Reset calculator results
        self.ase_calc.reset()
        # Broadcast the system to all cpus
        if self.save_memory:
            if self.rank==0:
                candidate=candidate.copy()
            candidate=broadcast(candidate,root=0)
        # Calculate the energies and forces
        candidate.calc=self.ase_calc
        candidate.calc.reset()
        forces=candidate.get_forces()
        self.energy_true=candidate.get_potential_energy(force_consistent=self.force_consistent)
        self.step+=1
        self.message_system('Single-point calculation finished.')
        # Store the data
        self.max_abs_forces=np.nanmax(np.linalg.norm(forces,axis=1))
        self.add_training([candidate])
        self.mlcalc.save_data(trajectory=self.trainingset)
        return

    def add_training(self,atoms_list,**kwargs):
        " Add atoms_list data to ML model on rank=0. "
        if self.save_memory:
            if self.rank!=0:
                return self.mlcalc
        self.mlcalc.add_training(atoms_list)
        return self.mlcalc

    def train_mlmodel(self,**kwargs):
        " Train the ML model "
        if self.save_memory:
            if self.rank!=0:
                return self.mlcalc
        self.mlcalc.train_model()
        return self.mlcalc

    def set_verbose(self,verbose,**kwargs):
        " Set verbose of MLModel. "
        self.mlcalc.mlmodel.verbose=verbose
        return 

    def extra_initial_data(self,**kwargs):
        " If only initial and final state is given then a third data point is calculated. "
        candidate=None
        if self.mlcalc.get_training_set_size()==2:
            images=self.make_interpolation(interpolation=self.interpolation)
            middle=int((self.n_images-2)/3.0) if self.start_energy>=self.end_energy else int((self.n_images-2)*2.0/3.0)
            candidate=images[1+middle].copy()
        if candidate is not None:
            self.evaluate(candidate)
        return candidate

    def run_mlneb(self,fmax=0.05,ml_steps=750,max_unc=0.25,**kwargs):
        " Run the NEB on the ML surrogate surface"
        # Convergence of the NEB
        neb_converged=False
        # If memeory is saved NEB is only performed on one CPU
        if self.save_memory:
            if self.rank!=0:
                return None,neb_converged
        # Make the interpolation from initial path or the previous path
        images=self.make_reused_interpolation(max_unc)
        # Check whether the predicted fmax for each image are lower than the NEB convergence fmax
        if self.get_fmax_predictions(images)<fmax:
            self.message_system('Too low forces on initial path!')
        else:
            # Run the NEB on the surrogate surface
            self.message_system('Starting NEB without climbing image on surrogate surface.')
            images,neb_converged=self.mlneb_opt(images,fmax=fmax,ml_steps=ml_steps,max_unc=max_unc,climb=False)
            self.save_mlneb(images)
        # Get the candidate
        candidate=self.choose_candidate(images)
        return candidate,neb_converged

    def get_predictions(self,images,**kwargs):
        " Calculate the energies and uncertainties with the ML calculator "
        energies=[image.get_potential_energy() for image in images]
        uncertainties=[image.calc.get_uncertainty() for image in images]
        return np.array(energies),np.array(uncertainties)

    def get_fmax_predictions(self,images,**kwargs):
        " Calculate the maximum perpendicular force with the ML calculator "
        forces=np.array([image.get_forces() for image in images]).reshape(-1,3)
        return np.nanmax(np.linalg.norm(forces,axis=1))
    
    def choose_candidate(self,images,**kwargs):
        " Use acquisition functions to chose the next training point "
        # Get the energies and uncertainties
        energy_path,unc_path=self.get_predictions(images)
        # Store the maximum predictions
        self.emax_ml=np.nanmax(energy_path)
        self.umax_ml=np.nanmax(unc_path)
        self.umean_ml=np.mean(unc_path)
        # Calculate the acquisition function for each image
        acq_values=self.acq.calculate(energy_path[1:-1],unc_path[1:-1])
        # Chose the maximum value given by the Acq. class
        i_min=self.acq.choose(acq_values)[0]
        # The next training point
        image=images[int(1+i_min)].copy()
        self.energy_pred=energy_path[int(1+i_min)]
        return image

    def mlneb_opt(self,images,fmax=0.05,ml_steps=750,max_unc=0.25,climb=False,**kwargs):
        " Run the ML NEB with checking uncertainties if selected. "
        neb=NEB(images,climb=climb,**self.neb_kwargs)
        neb_opt=self.local_opt(neb,**self.local_opt_kwargs)
        # Run the ML NEB fully without consider the uncertainty
        if max_unc==False:
            neb_opt.run(fmax=fmax*0.8,steps=ml_steps)
            self.message_system('NEB on surrogate surface converged!')
            return images,neb_opt.converged()
        # Stop the ML NEB if the uncertainty becomes too large
        for i in range(1,ml_steps+1):
            # Make backup of images before NEB step that can be used as a restart interpolation
            self.last_images_tmp=[image.copy() for image in images]
            # Run the NEB on the surrogate surface
            neb_opt.run(fmax=fmax,steps=i)
            # Calculate energy and uncertainty
            energy_path,unc_path=self.get_predictions(images)
            # Check if the uncertainty is too large
            if np.max(unc_path)>=max_unc:
                self.message_system('NEB on surrogate surface stopped due to high uncertainty!')
                break
            # Check if there is a problem with prediction
            if np.isnan(energy_path).any():
                images=self.make_interpolation(interpolation=self.last_images_tmp)
                for image in images:
                    image.get_forces()
                self.message_system('Stopped due to NaN value in prediction!')
                break
            # Check if the NEB is converged on the predicted surface
            if neb_opt.converged():
                self.message_system('NEB on surrogate surface converged!')
                break
        # Activate climbing when the model has low uncertainty and it is converged
        if neb_opt.converged():
            if climb==False and self.climb==True:
                self.message_system('Starting NEB with climbing image on surrogate surface.')
                return self.mlneb_opt(images,fmax=fmax,ml_steps=ml_steps-neb_opt.nsteps,max_unc=max_unc,climb=True)
        return images,neb_opt.converged()

    def save_mlneb(self,images,**kwargs):
        " Save the ML NEB result in the trajectory. "
        for image in images:
            self.trajectory_neb.write(self.mlcalc.mlmodel.database.copy_atoms(image))
        self.images=deepcopy(images)
        return self.images
    
    def get_barrier(self,forward=True,**kwargs):
        " Get the forward or backward predicted potential energy barrier. "
        if forward:
            return self.emax_ml-self.start_energy
        return self.emax_ml-self.end_energy

    def message_system(self,message,obj=None,end='\n'):
        " Print output once. "
        if self.full_output is True:
            if self.rank==0:
                if obj is None:
                    print(message,end=end)
                else:
                    print(message,obj,end=end)
        return

    def check_convergence(self,fmax,unc_convergence,neb_converged,**kwargs):
        " Check if the ML-NEB is converged to the final path with low uncertainty "
        converged=False
        if not self.save_memory or self.rank==0:
            # Check if NEB on the predicted potential energy surface is converged
            if neb_converged:
                # Check the force and uncertainty criteria are met
                if self.max_abs_forces<=fmax and self.umax_ml<=unc_convergence:
                    # Check the true energy deviation match the uncertainty prediction
                    if np.abs(self.energy_pred-self.energy_true)<=2.0*unc_convergence:
                        self.message_system("MLNEB is converged.") 
                        self.print_cite()
                        converged=True
        # Broadcast convergence statement if MPI is used
        if self.save_memory:
            converged=broadcast(converged,root=0)
        return converged

    def converged(self):
        " Whether MLNEB is converged. "
        return self.converging

    def print_cite(self):
        msg= "\n" + "-" * 79 + "\n"
        msg+="You are using AIDNEB. Please cite: \n"
        msg+="[1] J. A. Garrido Torres, M. H. Hansen, P. C. Jennings, "
        msg+="J. R. Boes and T. Bligaard. Phys. Rev. Lett. 122, 156001. "
        msg+="https://doi.org/10.1103/PhysRevLett.122.156001 \n"
        msg+="[2] O. Koistinen, F. B. Dagbjartsdottir, V. Asgeirsson, A. Vehtari"
        msg+=" and H. Jonsson. J. Chem. Phys. 147, 152720. "
        msg+="https://doi.org/10.1063/1.4986787 \n"
        msg+="[3] E. Garijo del Rio, J. J. Mortensen and K. W. Jacobsen. "
        msg+="Phys. Rev. B 100, 104103."
        msg+="https://doi.org/10.1103/PhysRevB.100.104103. \n"
        msg+="-" * 79 + '\n'
        self.message_system(msg)
        return 
    
    def make_summary_table(self,step,**kwargs):
        " Make the summary of the NEB process as table. "
        now=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            len(self.print_neb_list)
        except:
            self.print_neb_list=['| Step |        Time         | Pred. barrier (-->) | Pred. barrier (<--) | Max. uncert. | Avg. uncert. |   fmax   |']
        msg='|{0:6d}| '.format(step)
        msg+='{} |'.format(now)
        msg+='{0:21f}|'.format(self.get_barrier(forward=True))
        msg+='{0:21f}|'.format(self.get_barrier(forward=False))
        msg+='{0:14f}|'.format(self.umax_ml)
        msg+='{0:14f}|'.format(np.mean(self.umean_ml))
        msg+='{0:10f}|'.format(self.max_abs_forces)
        self.print_neb_list.append(msg)
        msg='\n'.join(self.print_neb_list)
        return msg
    
    def save_summary_table(self,**kwargs):
        " Save the summary table in the .txt file. "
        if self.tabletxt is not None:
            with open(self.tabletxt,'w') as thefile:
                msg='\n'.join(self.print_neb_list)
                thefile.writelines(msg)
        return
    
    def print_statement(self,step,**kwargs):
        " Print the NEB process as a table "
        msg=''
        if not self.save_memory or self.rank==0:
            msg=self.make_summary_table(step,**kwargs)
            self.save_summary_table()
            self.message_system(msg)
        return msg

