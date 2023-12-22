import numpy as np
from ase.neb import BaseNEB
from ase.io import read
from ase.io.trajectory import TrajectoryWriter
from ase.parallel import world,broadcast
import datetime

class MLNEB:
    def __init__(self,start,end,ase_calc,mlcalc=None,acq=None,
                 interpolation='idpp',interpolation_kwargs=dict(),
                 climb=True,neb_method=BaseNEB,neb_kwargs=dict(),n_images=15,
                 prev_calculations=None,use_database_check=True,
                 use_restart_path=True,check_path_unc=True,check_path_fmax=True,
                 use_low_unc_ci=True,save_memory=False,
                 apply_constraint=True,force_consistent=None,scale_fmax=0.8,
                 local_opt=None,local_opt_kwargs=dict(),
                 trainingset='evaluated_structures.traj',trajectory='MLNEB.traj',
                 tabletxt=None,full_output=False,**kwargs):
        """ 
        Nudged elastic band (NEB) with Machine Learning as active learning.

        Parameters:
            start: Atoms object with calculated energy or ASE Trajectory file.
                Initial end-point of the NEB path.
            end: Atoms object with calculated energy or ASE Trajectory file.
                Final end-point of the NEB path.
            ase_calc: ASE calculator instance.
                ASE calculator as implemented in ASE.
                See https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html
            mlcalc: ML-calculator instance.
                The ML-calculator instance used as surrogate surface. A default ML-model is used
                if mlcalc is None.
            acq: Acquisition class instance.
                The Acquisition instance used for calculating the acq. function and choose a candidate
                to calculate next. A default Acquisition instance is used if acq is None.
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
            neb_method : class object.
                The NEB implemented class object used for the ML-NEB. 
            neb_kwargs: dict.
                A dictionary with the arguments used in the NEB object to create the instance. 
                Climb must not be included.
                See https://wiki.fysik.dtu.dk/ase/ase/neb.html. 
            n_images: int.
                Number of images of the path (if not included a path before).
                The number of images include the 2 end-points of the NEB path.
            prev_calculations: Atoms list or ASE Trajectory file.
                (optional) The user can feed previously calculated data for the
                same hypersurface. The previous calculations must be fed as an
                Atoms list or Trajectory file.
            use_database_check : bool
                Whether to check if the new structure is within the database.
                If it is in the database, the structure is rattled. 
            use_restart_path: bool
                Use the path from last robust iteration (low uncertainty).
            check_path_unc: bool
                Check if the uncertainty is large for the restarted path and
                if it is then use the initial interpolation.
            check_path_fmax: bool
                Check if the maximum perpendicular force is larger for the restarted path than
                the initial interpolation and if so then replace it.
            use_low_unc_ci: bool
                Whether to only activative climbing image NEB when the uncertainties of all images are below unc_convergence.
                If use_low_unc_ci=False, the climbing image is activated without checking the uncertainties.
            save_memory: bool
                Whether to only train the ML calculator and store all objects on one CPU. 
                If save_memory==True then parallel optimization of the hyperparameters can not be achived.
                If save_memory==False no MPI object is used.  
            apply_constraint : boolean
                Whether to apply the constrains of the ASE Atoms instance to the calculated forces. 
                By default (apply_constraint=True) forces are 0 for constrained atoms and directions.
            force_consistent: boolean or None.
                Use force-consistent energy calls (as opposed to the energy
                extrapolated to 0 K). By default (force_consistent=None) uses
                force-consistent energies if available in the calculator, but
                falls back to force_consistent=False if not.
            scale_fmax : float
                The scaling of the fmax for the ML-NEB runs. 
                It makes the path converge tighter on surrogate surface. 
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
        self.interpolation_kwargs=dict(mic=True,remove_rotation_and_translation=False)
        self.interpolation_kwargs.update(interpolation_kwargs)
        self.n_images=n_images
        self.climb=climb
        self.neb_method=neb_method
        self.neb_kwargs=dict(k=3.0,method='improvedtangent',remove_rotation_and_translation=False)
        self.neb_kwargs.update(neb_kwargs)
        # General parameter settings
        self.use_database_check=use_database_check
        self.use_restart_path=use_restart_path
        self.check_path_unc=check_path_unc
        self.check_path_fmax=check_path_fmax
        self.use_low_unc_ci=use_low_unc_ci
        # Set initial parameters
        self.step=0
        self.converging=False
        # Setup the ML calculator
        if mlcalc is None:
            from ..regression.gaussianprocess.calculator.mlmodel import get_default_mlmodel
            from ..regression.gaussianprocess.calculator.mlcalc import MLCalculator
            from ..regression.gaussianprocess.fingerprint.invdistances import Inv_distances
            from ..regression.gaussianprocess.means.max import Prior_max
            if len(start)>1:
                fp=Inv_distances(reduce_dimensions=True,use_derivatives=True,mic=False,sorting=False)
            else:
                fp=None
            prior=Prior_max(add=1.0)
            mlmodel=get_default_mlmodel(model='tp',prior=prior,fp=fp,baseline=None,use_derivatives=True,parallel=(not save_memory),database_reduction=False)
            self.mlcalc=MLCalculator(mlmodel=mlmodel)
        else:
            self.mlcalc=mlcalc.copy()
        # Whether to have the full output
        self.full_output=full_output  
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
        self.apply_constraint=apply_constraint
        self.force_consistent=force_consistent
        # Scale the fmax on the surrogate surface
        self.scale_fmax=scale_fmax
        # Save local optimizer
        local_opt_kwargs_default=dict(trajectory='surrogate_neb.traj')
        if local_opt is None:
            from ase.optimize import MDMin
            local_opt=MDMin
            local_opt_kwargs_default.update(dict(dt=0.05))
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
              

    def run(self,fmax=0.05,unc_convergence=0.05,steps=500,ml_steps=1500,max_unc=0.25,**kwargs):
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
            max_unc: float (optional)
                Early stopping criteria. Maximum uncertainty before stopping the
                optimization on the surrogate surface.
                If it is None or False, it will run to convergence.
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
            candidate,neb_converged=self.run_mlneb(fmax=fmax*self.scale_fmax,ml_steps=ml_steps,max_unc=max_unc,unc_convergence=unc_convergence)
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
        from .interpolate_band import make_interpolation
        # Make the interpolation path
        images=make_interpolation(self.start.copy(),self.end.copy(),n_images=self.n_images,method=interpolation,**self.interpolation_kwargs)
        # Attach the ML calculator to all images
        images=self.attach_mlcalc(images)
        return images
    
    def make_reused_interpolation(self,unc_convergence,**kwargs):
        " Make the NEB interpolation path or use the previous path if it has low uncertainty. "
        # Make the interpolation from the initial points
        if not self.use_restart_path or self.last_images_tmp is None:
            self.message_system('The initial interpolation is used as the initial path!')
            return self.make_interpolation(interpolation=self.interpolation)
        else:
            # Reuse the previous path
            if self.check_path_unc and self.check_path_fmax:
                uncmax_tmp,fmax_tmp=self.get_path_unc_fmax(interpolation=self.last_images_tmp)
                # Check if the uncertainty is too large
                if uncmax_tmp<=unc_convergence:
                    # Check if the perpendicular forces are less for the new path
                    fmax_last=self.get_path_unc_fmax(interpolation=self.last_images)[1]
                    if fmax_tmp<=fmax_last:
                        self.message_system('The last path is used as the initial path!')
                        self.last_images=[image.copy() for image in self.last_images_tmp]
                        return self.make_interpolation(interpolation=self.last_images_tmp)
                    else:
                        self.last_images_tmp=None
                        self.message_system('The previous last path is used as the initial path due to fmax!')
                else:
                    self.last_images_tmp=None
                    self.message_system('The previous last path is used as the initial path due to uncertainty!')
            elif self.check_path_unc:
                uncmax_tmp=self.get_path_unc_fmax(interpolation=self.last_images_tmp)[0]
                # Check if the uncertainty is too large
                if uncmax_tmp<=unc_convergence:
                    self.message_system('The last path is used as the initial path!')
                    self.last_images=[image.copy() for image in self.last_images_tmp]
                    return self.make_interpolation(interpolation=self.last_images_tmp)
                else:
                    self.last_images_tmp=None
                    self.message_system('The previous last path is used as the initial path due to uncertainty!')
            elif self.check_path_fmax:
                # Check if the perpendicular forces are less for the new path
                fmax_tmp=self.get_path_unc_fmax(interpolation=self.last_images_tmp)[1]
                fmax_last=self.get_path_unc_fmax(interpolation=self.last_images)[1]
                if fmax_tmp<=fmax_last:
                    self.message_system('The last path is used as the initial path!')
                    self.last_images=[image.copy() for image in self.last_images_tmp]
                    return self.make_interpolation(interpolation=self.last_images_tmp)
                else:
                    self.last_images_tmp=None
                    self.message_system('The previous last path is used as the initial path due to fmax!')
            else:
                self.message_system('The last path is used as the initial path!')
        return self.make_interpolation(interpolation=self.last_images)

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
        # Reset calculator results
        self.ase_calc.reset()
        # Ensure that the candidate is not already in the database
        if self.use_database_check:
            candidate=self.ensure_not_in_database(candidate)
        # Broadcast the system to all cpus
        if self.rank==0:
            candidate=candidate.copy()
        candidate=broadcast(candidate,root=0)
        # Calculate the energies and forces
        self.message_system('Performing evaluation.',end='\r')
        candidate.calc=self.ase_calc
        candidate.calc.reset()
        forces=candidate.get_forces(apply_constraint=self.apply_constraint)
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
        self.mlcalc.add_training(atoms_list)
        return self.mlcalc

    def train_mlmodel(self,**kwargs):
        " Train the ML model "
        if self.save_memory:
            if self.rank!=0:
                return self.mlcalc
        # Update database with the points of interest
        self.update_database_arguments(point_interest=self.last_images)
        # Train the ML model
        self.mlcalc.train_model()
        return self.mlcalc

    def set_verbose(self,verbose,**kwargs):
        " Set verbose of MLModel. "
        self.mlcalc.mlmodel.update_arguments(verbose=verbose)
        return 
    
    def is_in_database(self,atoms,**kwargs):
        " Check if the ASE Atoms is in the database. "
        return self.mlcalc.is_in_database(atoms,**kwargs)
    
    def update_database_arguments(self,point_interest=None,**kwargs):
        " Update the arguments in the database. "
        self.mlcalc.update_database_arguments(point_interest=point_interest,**kwargs)
        return self
    
    def ensure_not_in_database(self,atoms,perturb=0.01,**kwargs):
        " Ensure the ASE Atoms object is not in database by perturb it if it is. "
        # Return atoms if it does not exist
        if atoms is None:
            return atoms
        # Check if atoms object is in the database
        if self.is_in_database(atoms,**kwargs):
            # Get positions
            pos=atoms.get_positions()
            # Rattle the positions
            pos=pos+np.random.uniform(low=-perturb,high=perturb,size=pos.shape)
            atoms.set_positions(pos)
            self.message_system('The system is rattled, since it is already in the database.')
        return atoms

    def extra_initial_data(self,**kwargs):
        " If only initial and final state is given then a third data point is calculated. "
        candidate=None
        if self.get_training_set_size()<=2:
            images=self.make_interpolation(interpolation=self.interpolation)
            middle=int((self.n_images-2)/3.0) if self.start_energy>=self.end_energy else int((self.n_images-2)*2.0/3.0)
            candidate=images[1+middle].copy()
        if candidate is not None:
            self.evaluate(candidate)
        return candidate

    def run_mlneb(self,fmax=0.05,ml_steps=750,max_unc=0.25,unc_convergence=0.05,**kwargs):
        " Run the NEB on the ML surrogate surface"
        # Convergence of the NEB
        neb_converged=False
        # If memeory is saved NEB is only performed on one CPU
        if self.rank!=0:
            return None,neb_converged
        # Make the interpolation from initial path or the previous path
        images=self.make_reused_interpolation(unc_convergence)
        # Run the NEB on the surrogate surface
        self.message_system('Starting NEB without climbing image on surrogate surface.')
        images,neb_converged=self.mlneb_opt(images,fmax=fmax,ml_steps=ml_steps,max_unc=max_unc,unc_convergence=unc_convergence,climb=False)
        self.save_mlneb(images)
        # Get the candidate
        candidate=self.choose_candidate(images)
        return candidate,neb_converged
    
    def get_training_set_size(self):
        " Get the size of the training set "
        return self.mlcalc.get_training_set_size()

    def get_predictions(self,images,**kwargs):
        " Calculate the energies and uncertainties with the ML calculator "
        energies=[]
        uncertainties=[]
        for image in images:
            uncertainties.append(image.calc.get_uncertainty(image))
            energies.append(image.get_potential_energy())
        return np.array(energies),np.array(uncertainties)

    def get_path_unc_fmax(self,interpolation,**kwargs):
        " Get the maximum uncertainty and fmax prediction from the NEB interpolation. "
        uncmax=None
        fmax=None
        images=self.make_interpolation(interpolation=interpolation)
        if self.check_path_unc:
            uncmax=np.nanmax(self.get_predictions(images)[1])
        if self.check_path_fmax:
            fmax=self.get_fmax_predictions(images)
        return uncmax,fmax

    def get_fmax_predictions(self,images,**kwargs):
        " Calculate the maximum perpendicular force with the ML calculator "
        neb=self.neb_method(images,climb=False,**self.neb_kwargs)
        forces=neb.get_forces()
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

    def mlneb_opt(self,images,fmax=0.05,ml_steps=750,max_unc=0.25,unc_convergence=0.05,climb=False,**kwargs):
        " Run the ML NEB with checking uncertainties if selected. "
        # Construct the NEB
        neb=self.neb_method(images,climb=climb,**self.neb_kwargs)
        with self.local_opt(neb,**self.local_opt_kwargs) as neb_opt:
            # Run the MLNEB fully without consider the uncertainty
            if max_unc==False or max_unc is None:
                neb_opt,images=self.mlneb_opt_no_max_unc(neb_opt,images,fmax=fmax,ml_steps=ml_steps,climb=climb,**kwargs)
            else:
                # Stop the MLNEB if the uncertainty becomes too large
                neb_opt,images=self.mlneb_opt_max_unc(neb_opt,images,fmax=fmax,ml_steps=ml_steps,max_unc=max_unc,climb=climb,**kwargs)
            # Check if the MLNEB is converged
            converged=neb_opt.converged()
            # Check the number of iterations used
            nsteps=neb_opt.nsteps
        # Activate climbing when the NEB is converged
        if converged:
            self.message_system('NEB on surrogate surface converged!')
            if not climb and nsteps<ml_steps and self.climb:
                # Check that the uncertainty is low enough to do CI-NEB if requested
                if not self.use_low_unc_ci or np.max(self.get_predictions(images)[1])<=unc_convergence:
                    self.message_system('Starting NEB with climbing image on surrogate surface.')
                    return self.mlneb_opt(images,fmax=fmax,ml_steps=ml_steps-nsteps,max_unc=max_unc,unc_convergence=unc_convergence,climb=True)
        return images,converged
    
    def mlneb_opt_no_max_unc(self,neb_opt,images,fmax=0.05,ml_steps=750,climb=False,**kwargs):
        " Run the MLNEB fully without consider the uncertainty. "
        neb_opt.run(fmax=fmax,steps=ml_steps)
        if not climb:
            self.last_images_tmp=[image.copy() for image in images]
        return neb_opt,images
    
    def mlneb_opt_max_unc(self,neb_opt,images,fmax=0.05,ml_steps=750,max_unc=0.25,climb=False,**kwargs):
        " Run the MLNEB, but stop it if the uncertainty becomes too large. "
        for i in range(1,ml_steps+1):
            # Make backup of images before NEB step that can be used as a restart interpolation
            if not climb:
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
                if not climb:
                    self.last_images_tmp=[image.copy() for image in images]
                break
        return neb_opt,images

    def save_mlneb(self,images,**kwargs):
        " Save the ML NEB result in the trajectory. "
        self.images=[]
        for image in images:
            image=self.mlcalc.copy_atoms(image)
            self.images.append(image)
            self.trajectory_neb.write(image)
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
        if self.rank==0:
            # Check if NEB on the predicted potential energy surface is converged
            if neb_converged:
                # Check the force and uncertainty criteria are met
                if self.max_abs_forces<=fmax and self.umax_ml<=unc_convergence:
                    # Check the true energy deviation match the uncertainty prediction
                    if np.abs(self.energy_pred-self.energy_true)<=2.0*unc_convergence:
                        self.message_system("MLNEB is converged.") 
                        self.print_cite()
                        converged=True
        # Broadcast convergence statement
        converged=broadcast(converged,root=0)
        return converged

    def converged(self):
        " Whether MLNEB is converged. "
        return self.converging

    def print_cite(self):
        msg= "\n" + "-" * 79 + "\n"
        msg+="You are using MLNEB. Please cite: \n"
        msg+="[1] J. A. Garrido Torres, M. H. Hansen, P. C. Jennings, "
        msg+="J. R. Boes and T. Bligaard. Phys. Rev. Lett. 122, 156001. "
        msg+="https://doi.org/10.1103/PhysRevLett.122.156001 \n"
        msg+="[2] O. Koistinen, F. B. Dagbjartsdottir, V. Asgeirsson, A. Vehtari"
        msg+=" and H. Jonsson. J. Chem. Phys. 147, 152720. "
        msg+="https://doi.org/10.1063/1.4986787 \n"
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
        if self.rank==0:
            msg=self.make_summary_table(step,**kwargs)
            self.save_summary_table()
            self.message_system(msg)
        return msg

