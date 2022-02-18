import numpy as np
from catlearn.optimize.io import print_cite_mlneb
from ase.neb import NEB
from ase.io import read, write
from ase.optimize import MDMin
from ase.calculators.singlepoint import SinglePointCalculator
from ase.parallel import parprint
import os
from catlearn import __version__
import copy
import datetime
from ase.parallel import parallel_function
from mpi4py import MPI


class MLNEB(object):

    def __init__(self, start, end, prev_calculations=None,
                 n_images=0.25, k=None, interpolation='linear', mic=False,
                 neb_method='improvedtangent', ase_calc=None, ase_calc_kwargs={}, restart=True,
                 force_consistent=None, mlmodel=None, mlcalc=None, acq=None,trainingset='evaluated_structures.traj',trajectory='all_predicted_paths.traj'):

        """ Nudged elastic band (NEB) setup.

        Parameters
        ----------
        start: Trajectory file (in ASE format) or Atoms object.
            Initial end-point of the NEB path or Atoms object.
        end: Trajectory file (in ASE format) or Atoms object.
            Final end-point of the NEB path.
        prev_calculations: Atoms list or Trajectory file (in ASE format).
            (optional) The user can feed previously calculated data for the
            same hypersurface. The previous calculations must be fed as an
            Atoms list or Trajectory file.
        n_images: int or float
            Number of images of the path (if not included a path before).
            The number of images include the 2 end-points of the NEB path.
        k: float or list
            Spring constant(s) in eV/Ang.
        interpolation: string or Atoms list or Trajectory
            Automatic interpolation can be done ('idpp' and 'linear' as
            implemented in ASE).
            See https://wiki.fysik.dtu.dk/ase/ase/neb.html.
            Manual: Trajectory file (in ASE format) or list of Atoms.
            Atoms trajectory or list of Atoms containing the images along the
            path.
        mic: boolean
            Use mic=True to use the Minimum Image Convention and calculate the
            interpolation considering periodic boundary conditions.
        neb_method: string
            NEB method as implemented in ASE. ('aseneb', 'improvedtangent'
            or 'eb').
            See https://wiki.fysik.dtu.dk/ase/ase/neb.html.
        ase_calc: ASE calculator Object.
            ASE calculator as implemented in ASE.
            See https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html
        restart: boolean
            Only useful if you want to continue your ML-NEB in the same
            directory. The file "evaluated_structures.traj" from the
            previous run, must be located in the same run directory.
        force_consistent: boolean or None
            Use force-consistent energy calls (as opposed to the energy
            extrapolated to 0 K). By default (force_consistent=None) uses
            force-consistent energies if available in the calculator, but
            falls back to force_consistent=False if not.

        """
        # General setup.
        self.trainingset=trainingset
        self.trajectory=trajectory
        self.n_images = n_images
        self.interpolation=interpolation
        self.feval = 0
        self.fc = force_consistent
        self.iter = 0
        self.ase_calc = ase_calc
        assert self.ase_calc, 'ASE calculator not provided (see "ase_calc" flag).'
        self.ase_calc_kwargs = ase_calc_kwargs
        self.mic=mic
        self.restart=restart
        self.version='ML-NEB ' + __version__
        self.interesting_point=None
        self.train_images=[]
        # Settings for the NEB.
        self.neb_method=neb_method
        # Load previous data
        self.save_prev_calculations(prev_calculations=prev_calculations)
        # Set up the machine learning part
        if mlmodel is None:
            from catlearn.regression.gp_bv.calculator import GPModel
            mlmodel=GPModel()
        if mlcalc is None:
            from catlearn.regression.gp_bv.calculator import GPCalculator
            mlcalc=GPCalculator()
        if acq is None:
            from catlearn.optimize.acquisition import Acquisition
            acq=Acquisition(mode='umucb',objective='max',kappa=2)
        self.mlcalc=copy.deepcopy(mlcalc)
        self.mlcalc.model=copy.deepcopy(mlmodel)
        self.acq=copy.deepcopy(acq)
        # Set up initial and final states
        self.set_up_endpoints(start,end)
        # Make initial path and add training data to ML-Model
        self.initial_interpolation(interpolation=interpolation,k=k)
        


    def run(self, fmax=0.05, unc_convergence=0.050, steps=500,
            ml_steps=750, max_step=0.25, sequential=False,
            full_output=False, local_opt=None, local_opt_kwargs={}):

        """Executing run will start the NEB optimization process.

        Parameters
        ----------
        fmax : float
            Convergence criteria (in eV/Angs).
        unc_convergence: float
            Maximum uncertainty for convergence (in eV).
        steps : int
            Maximum number of iterations in the surrogate model.
        trajectory: string
            Filename to store the output.
        acquisition : string
            Acquisition function.
        dt : float
            dt parameter for MDMin.
        ml_steps: int
            Maximum number of steps for the NEB optimization on the
            predicted landscape.
        max_step: float
            Early stopping criteria. Maximum uncertainty before stopping the
            optimization in the predicted landscape.
        sequential: boolean
            When sequential is set to True, the ML-NEB algorithm starts
            with only one moving image. After finding a saddle point
            the algorithm adds all the images selected in the MLNEB class
            (the total number of NEB images is defined in the 'n_images' flag).
        full_output: boolean
            Whether to print on screen the full output (True) or not (False).

        Returns
        -------
        Minimum Energy Path from the initial to the final states.

        """
        # Wheter to print everything
        self.fullout = full_output
        # Define local optimizer
        if local_opt is None:
            local_opt=MDMin
            local_opt_kwargs={'dt':0.025}
        # General setup
        stationary_point_found = False
        org_n_images = self.n_images
        self.acq.unc_convergence=unc_convergence
        converged=False
        # Set up parallel objects
        self.comm = MPI.COMM_WORLD
        self.rank,self.size=self.comm.Get_rank(),self.comm.Get_size()
        # Calculate a third point if only initial and final structures are known.
        self.extra_data_point()
        # Use only one moving image
        if sequential is True:
            self.n_images = 3
        while True:
            # 1. Perform the ML-NEB inclusive finding the next point
            self.mlneb_part(fmax,max_step,ml_steps,stationary_point_found,org_n_images,local_opt,local_opt_kwargs)

            # 2. Calculate the next point 
            self.evaluate_ase()            

            # 3. Store results.
            self.store_results_mlneb()

            # 4. Check convergence:
            stationary_point_found,converged=self.check_convergence(fmax,unc_convergence,org_n_images,stationary_point_found)
            converged=self.broadcast_converged(converged=converged)

            # Check convergence criteria
            if converged:
                break
            # Break if reaches the max number of iterations set by the user.
            if steps <= self.iter:
                parprint('Maximum number iterations reached. Not converged.')
                break

        parprint('Number of steps performed in total:',self.iter)
        print_cite_mlneb()
        return self

    def set_up_endpoints(self,start,end):
        " Load and calculate the intial and final stats"
        # Load and calculate initial state
        if isinstance(start, str):
            start=read(start, '-1:')
        try:
            self.start=self.copy_image(start)
        except:
            raise Exception('Initial structure for the NEB was not provided')
        self.eval_and_append(self.start)
        self.energy_is=self.energy
        # Number of atoms and the constraint used
        self.constraints=self.start.constraints
        self.index_mask=None
        if len(self.constraints)>0:
            from ase.constraints import FixAtoms
            self.index_mask=np.array([c.get_indices() for c in self.constraints if isinstance(c,FixAtoms)]).flatten()
            self.index_mask=sorted(list(set(self.index_mask)))
            self.mlcalc.model.index_mask=copy.deepcopy(self.index_mask)
        
        # Load and calculate final state
        if isinstance(end, str):
            end=read(end, '-1:')
        try:
            self.end=self.copy_image(end)
        except:
            raise Exception('Final structure for the NEB was not provided')
        self.eval_and_append(self.end)
        self.energy_fs=self.energy
        # Calculate the direct distance between the initial and final states
        self.path_distance=np.linalg.norm(self.start.get_positions().flatten()-self.end.get_positions().flatten())
        pass

    def save_prev_calculations(self,prev_calculations=None):
        " Store previous calculated data "
        # Store previous calculated data given in prev_calculations
        if prev_calculations is not None:
            if isinstance(prev_calculations,str):
                if os.path.exists(prev_calculations):
                    prev_calculations=read(prev_calculations,':')
                    for atoms in prev_calculations:
                        self.eval_and_append(atoms)
        # Store previous calculated data given from trajectory file
        if self.restart and prev_calculations is None:
            if os.path.exists(self.trainingset):
                prev_calculations=read(self.trainingset,':')
                for atoms in prev_calculations:
                    self.eval_and_append(atoms)
        pass

    @parallel_function
    def initial_interpolation(self,interpolation='linear',k=None):
        " Add training data to machine learning model and make the first path used "
        # Add training data to ML-Model
        self.mlcalc.model.add_training_points(self.train_images)
        self.mlcalc.model.train_model()
        # Make initial path
        path=None
        # If the path is given then use it
        if interpolation not in ['idpp','linear']:
            if isinstance(interpolation,str):
                if os.path.exists(interpolation):
                    path=read(interpolation,':')
            elif isinstance(interpolation,list):
                path=[self.copy_image(atoms) for atoms in interpolation]
        # Calculate the number of images if a float is given
        if path is None:
            if isinstance(self.n_images,float):
                self.n_images=int(self.path_distance/self.n_images)
                if self.n_images<3:
                    self.n_images=3
        else:
            self.n_images=len(path)
        # Calculate the spring constant if it is not given
        self.spring = k if k is not None else np.sqrt((self.n_images-1) / self.path_distance)
        # Set up NEB path
        self.images=self.make_interpolation(interpolation=interpolation,path=path)
        # Save files with all the paths that have been predicted:
        self.e_path,self.uncertainty_path=self.energy_and_uncertainty()
        write(self.trajectory, self.images)
        pass

    def make_interpolation(self,interpolation='linear',path=None):
        " Make the NEB interpolation path "
        images=[self.copy_image(self.start)]
        for i in range(1,self.n_images-1):
            image=self.copy_image(self.start)
            image.set_calculator(copy.deepcopy(self.mlcalc))
            if path is not None:
                image.set_positions(path[i].get_positions())
            image.set_constraint(self.constraints)
            images.append(image)
        images.append(self.copy_image(self.end))
        if path is None:
            neb_interpolation=NEB(images,k=self.spring)
            neb_interpolation.interpolate(method=interpolation,mic=self.mic)
        return images  

    def eval_and_append(self,atoms):
        " Recalculate the energy and forces with the ASE calculator and store it as training data "
        self.energy=atoms.get_potential_energy(force_consistent=self.fc)
        self.forces=atoms.get_forces()
        self.train_images.append(atoms)
        write(self.trainingset,self.train_images)
        self.max_abs_forces=np.max(np.linalg.norm(self.forces,axis=1))
        pass

    def evaluate_ase(self):
        " Set the ASE calculator evaluate the point of interest in parallel and add it as a new training point"
        # Broadcast the system to other cpus
        if self.rank==0:
            self.message_system('Performing evaluation on the real landscape...')
            self.interesting_point.set_calculator(None)
            for r in range(1,self.size):
                self.comm.send(self.interesting_point,dest=r,tag=1)
        else:
            self.interesting_point=self.comm.recv(source=0,tag=1)
        self.comm.barrier()
        self.interesting_point.set_calculator(self.ase_calc(**self.ase_calc_kwargs))
        # Evaluate the energy and forces
        self.energy=self.interesting_point.get_potential_energy(force_consistent=self.fc)
        self.forces=self.interesting_point.get_forces()
        # Add the structure as training data
        if self.rank==0:
            self.interesting_point=self.copy_image(self.interesting_point)
            self.message_system('Single-point calculation finished.')
            self.eval_and_append(self.interesting_point)
            self.mlcalc.model.add_training_points([self.interesting_point])
        self.iter+=1
        pass

    def copy_image(self,atoms):
        """
        Copy an image. It returns a copy of the atoms object with the single point
        calculator attached
        """
        # Check if the atoms object has energy and forces calculated for this position
        # If not, compute them
        atoms.get_forces()
        # Initialize a SinglePointCalculator to store this results
        calc = SinglePointCalculator(atoms, **atoms.calc.results)
        atoms0 = atoms.copy()
        atoms0.calc = calc
        return atoms0

    def broadcast_converged(self,converged=False):
        " Broadcast the convergence statement to all CPUs"
        if self.rank==0:
            for r in range(1,self.size):
                self.comm.send(converged,dest=r,tag=2)
        else:
            converged=self.comm.recv(source=0,tag=2)
        self.comm.barrier()
        return converged

    def energy_and_uncertainty(self):
        " Calculate the energies and uncertainties with the ML calculator "
        energies=[img.get_potential_energy() for img in self.images]
        uncertainties=[0]+[img.calc.results['uncertainty'] for img in self.images[1:-1]]+[0]
        return np.array(energies),np.array(uncertainties)

    def message_system(self,message,obj=None):
        " Print output "
        if self.fullout is True:
            if obj is None:
                parprint(message)
            else:
                parprint(message,obj)
        pass

    
    def extra_data_point(self):
        " Calculate a third point if only initial and final structures are known. "
        if len(self.train_images) == 2:
            self.middle_extra_data()
            self.evaluate_ase()
            self.print_neb()
        pass

    @parallel_function
    def middle_extra_data(self):
        " What data point to calculate extra if only initial and final structures are known "
        middle = int(self.n_images * (1./3.)) if self.energy_is>=self.energy_fs else int(self.n_images * (2./3.)) 
        self.interesting_point=copy.deepcopy(self.images[middle])
        pass

    @parallel_function
    def mlneb_part(self,fmax,max_step,ml_steps,stationary_point_found,org_n_images,local_opt,local_opt_kwargs):
        " Run the ML-NEB part with training the ML, run ML-NEB, get results, and get next point "
        # 1. Train Machine Learning process.
        self.mlcalc.model.train_model()
        
        # 2. Setup and run ML NEB:
        self.mlneb_opt(fmax,max_step,ml_steps,stationary_point_found,org_n_images,local_opt,local_opt_kwargs)

        # 3. Get results from ML NEB using ASE NEB Tools (https://wiki.fysik.dtu.dk/ase/ase/neb.html)
        self.interesting_point = []
        # Get fit of the discrete path.
        self.e_path,self.uncertainty_path=self.energy_and_uncertainty()

        # 4. Select next point to train (acquisition function):
        self.acq.stationary_point_found=stationary_point_found
        acq_values=self.acq.calculate(self.e_path[1:-1],self.uncertainty_path[1:-1])
        argmax=self.acq.choose(acq_values)[0]
        self.interesting_point=copy.deepcopy(self.images[1+argmax])
        pass

    def mlneb_opt(self,fmax,max_step,ml_steps,stationary_point_found,org_n_images,local_opt,local_opt_kwargs):
        " Setup and run the ML-NEB: "
        ml_cycles = 0
        while True:
            if stationary_point_found is True:
                self.n_images = org_n_images
            # Start from the last path
            starting_path = self.images 
            # Use the initial path
            if ml_cycles == 0:
                self.message_system('Using initial path.')
                starting_path=read(self.trajectory,'0:'+str(self.n_images))
            # Use the last predicted path for the previous run
            if ml_cycles == 1:
                self.message_system('Using last predicted path.')
                starting_path = read(self.trajectory,str(-self.n_images)+':')
            # Make the path
            self.images=self.make_interpolation(interpolation=self.interpolation,path=starting_path)
            # Check energy and uncertainty before optimization:
            self.e_path,self.uncertainty_path=self.energy_and_uncertainty()
            unc_ml=np.max(self.uncertainty_path)
            self.max_target=np.max(self.e_path)
            if unc_ml >= max_step:
                self.message_system('Maximum uncertainty reach in initial path. Early stop.')
                break
            # Perform NEB in the predicted landscape.
            ml_neb=NEB(self.images, climb=True,method=self.neb_method,k=self.spring)
            neb_opt=local_opt(ml_neb,**local_opt_kwargs) if self.fullout else local_opt(ml_neb,logfile=None,**local_opt_kwargs)
            # Run the NEB optimization
            ml_converged=self.mlneb_opt_run(ml_neb,neb_opt,fmax,max_step,ml_steps)
            # Check if it is converged
            if ml_converged:
                self.message_system('Converged opt. in the predicted landscape.')
                break
            ml_cycles += 1
            self.message_system('ML cycles performed:', ml_cycles)
            if ml_cycles == 2:
                self.message_system('ML process not optimized...not safe... \nChange interpolation or numb. of images.')
                break
        return ml_cycles

    def mlneb_opt_run(self,ml_neb,neb_opt,fmax,max_step,ml_steps):
        " Run the NEB on the predicted surface one step at the time. It is stopped if the energy or uncertainty is too large"
        ml_converged = False
        for n_steps_performed in range(1,ml_steps):
            # Save prev. positions:
            prev_save_positions = [img.get_positions() for img in self.images]
            # Run the NEB optimization one step
            neb_opt.run(fmax=(fmax * 0.85), steps=1)
            neb_opt.nsteps = 0
            # Calculate the maximum energies and uncertainties
            self.e_path,self.uncertainty_path=self.energy_and_uncertainty()
            e_ml=np.max(self.e_path[1:-1])
            unc_ml=np.max(self.uncertainty_path)
            # If the energy barrier is too large
            if e_ml >= self.max_target + 0.2:
                for i in range(1, self.n_images-1):
                    self.images[i].positions = prev_save_positions[i]
                self.message_system('Pred. energy above max. energy. Early stop.')
                break
            # If the uncertainty is too large
            if unc_ml >= max_step:
                for i in range(1, self.n_images-1):
                    self.images[i].positions = prev_save_positions[i]
                self.message_system('Maximum uncertainty reach. Early stop.')
                break
            # If the energy is a nan value (error)
            if np.isnan(ml_neb.emax):
                self.images = read(self.trajectory, str(-self.n_images) + ':')
                self.message_system('Not converged')
                break
            # The NEB is converged 
            if neb_opt.converged():
                ml_converged = True
                break
            # Not converged within the steps given
            if n_steps_performed > ml_steps-1:
                self.message_system('Not converged yet...')
                break
        return ml_converged

    @parallel_function
    def store_results_mlneb(self):
        " Store the forward and backwards energy and the neb path "
        parprint('\n')
        self.energy_forward = np.max(self.e_path) - self.e_path[0]
        self.energy_backward = np.max(self.e_path) - self.e_path[-1]
        self.print_neb()
        write(self.trajectory, self.images)
        pass

    @parallel_function
    def check_convergence(self,fmax,unc_convergence,org_n_images,stationary_point_found):
        " Check if the ML-NEB is converged to the final path with low uncertainty "
        converged=False
        # Check whether the evaluated point is a stationary point.
        if self.max_abs_forces <= fmax:
            stationary_point_found = True
            # Check all images are used
            if self.n_images == org_n_images:
                # Check the maximum uncertainty is lower than the convergence criteria
                if np.max(self.uncertainty_path[1:-1])<unc_convergence:
                    # Write the Last path.
                    write(self.trajectory, self.images)
                    parprint("Congratulations! Your ML NEB is converged. See the final path in file {}".format(self.trajectory))
                    converged=True
        return stationary_point_found,converged

    def print_neb(self):
        " Print the NEB process as a table "
        if self.rank==0:
            now=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            try:
                len(self.print_neb_list)
            except:
                self.print_neb_list=['| Step |        Time         | Pred. barrier (-->) | Pred. barrier (<--) | Max. uncert. | Avg. uncert. |   fmax   |']
                self.energy_backward,self.energy_forward=0,0

            msg='|{0:6d}| '.format(self.iter)+'{} |'.format(now)
            msg+='{0:21f}|'.format(self.energy_forward)+'{0:21f}|'.format(self.energy_backward)
            msg+='{0:14f}|'.format(np.max(self.uncertainty_path[1:-1]))
            msg+='{0:14f}|'.format(np.mean(self.uncertainty_path[1:-1]))+'{0:10f}|'.format(self.max_abs_forces)
            self.print_neb_list.append(msg)
            msg='\n'.join(self.print_neb_list)
            parprint(msg)
        pass

