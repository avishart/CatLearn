import numpy as np
from catlearn.optimize.io import ase_to_catlearn, store_results_neb, \
                                 print_version, store_trajectory_neb, \
                                 print_info_neb, array_to_ase, print_cite_mlneb
from catlearn.optimize.constraints import create_mask, apply_mask
from ase.neb import NEB
from ase.neb import NEBTools
from ase.io import read, write
from ase.optimize import MDMin
from ase.parallel import parprint, rank, parallel_function
from scipy.spatial import distance
import os
from catlearn.regression.gp_bv.calculator import GPModel,GPCalculator
from catlearn.optimize.acquisition import Acquisition
from ase.calculators.calculator import Calculator, all_changes
from ase.atoms import Atoms
from catlearn import __version__
import copy
import datetime


class MLNEB(object):

    def __init__(self, start, end, prev_calculations=None,
                 n_images=0.25, k=None, interpolation='linear', mic=False,
                 neb_method='improvedtangent', ase_calc=None, restart=True,
                 force_consistent=None, mlmodel=None, mlcalc=None, acq=None,trainingset='evaluated_structures.traj',trajectory='all_predicted_paths.traj'):

        """ Nudged elastic band (NEB) setup.

        Parameters
        ----------
        start: Trajectory file (in ASE format) or Atoms object.
            Initial end-point of the NEB path or Atoms object.
        end: Trajectory file (in ASE format).
            Final end-point of the NEB path.
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
        prev_calculations: Atoms list or Trajectory file (in ASE format).
            (optional) The user can feed previously calculated data for the
            same hypersurface. The previous calculations must be fed as an
            Atoms list or Trajectory file.
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
        self.mic=mic
        self.restart=restart
        self.version='ML-NEB ' + __version__
        self.interesting_point=None
        self.train_images=[]
        self.list_gradients=[]
        # Settings for the NEB.
        self.neb_method=neb_method

        # Start end-point and final end-point
        self.save_prev_calculations(prev_calculations=None)
        self.set_up_endpoints(start,end,prev_calculations)

        # Set up the machine learning part
        if mlmodel is None:
            mlmodel=GPModel(train_images=self.train_images)
        if mlcalc is None:
            mlcalc=GPCalculator(mlmodel)
        if acq is None:
            acq=Acquisition(mode='umucb',objective='max',kappa=2)
        self.mlmodel=copy.deepcopy(mlmodel)
        self.mlcalc=copy.deepcopy(mlcalc)
        self.acq=copy.deepcopy(acq)

        self.initial_interpolation(interpolation=interpolation,k=k)

        # Save files with all the paths that have been predicted:
        self.e_path,self.uncertainty_path=self.energy_and_uncertainty()
        write(self.trajectory, self.images)


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
        self.fullout = full_output
        if local_opt is None:
            local_opt=MDMin
            local_opt_kwargs={'dt':0.025}
        stationary_point_found = False
        org_n_images = self.n_images
        self.acq.unc_convergence=unc_convergence

        # Calculate a third point if only known initial & final structures.
        if len(self.train_images) == 2:
            middle = int(self.n_images * (1./3.)) if self.energy_is >= self.energy_fs else int(self.n_images * (2./3.)) 
            self.interesting_point = copy.deepcopy(self.images[middle])
            self.evaluate_ase(self.interesting_point)
            self.mlcalc.model.add_training_points([self.interesting_point])
            self.print_neb()

        if sequential is True:
            self.n_images = 3

        while True:
            # 1. Train Machine Learning process.
            self.mlcalc.model.train_model()
            
            # 2. Setup and run ML NEB:
            self.mlneb_opt(fmax,max_step,ml_steps,stationary_point_found,org_n_images,local_opt,local_opt_kwargs)

            # 3. Get results from ML NEB using ASE NEB Tools:
            # See https://wiki.fysik.dtu.dk/ase/ase/neb.html
            self.interesting_point = []
            # Get fit of the discrete path.
            self.e_path,self.uncertainty_path=self.energy_and_uncertainty()

            # 4. Select next point to train (acquisition function):
            self.acq.stationary_point_found=stationary_point_found
            acq_values=self.acq.calculate(self.e_path[1:-1],self.uncertainty_path[1:-1])
            argmax=self.acq.choose(acq_values)[0]
            self.interesting_point=copy.deepcopy(self.images[1+argmax])

            # 5. Add a new training point and evaluate it.
            self.message_system('Performing evaluation on the real landscape...')
            self.evaluate_ase(self.interesting_point)
            self.message_system('Single-point calculation finished.')
            self.mlcalc.model.add_training_points([self.interesting_point])

            # 6. Store results.
            parprint('\n')
            self.energy_forward = np.max(self.e_path) - self.e_path[0]
            self.energy_backward = np.max(self.e_path) - self.e_path[-1]
            #self.max_abs_forces = self.list_gradients[-1]
            self.print_neb()
            write(self.trajectory, self.images)

            # 7. Check convergence:
            stationary_point_found,converged=self.check_convergence(fmax,unc_convergence,org_n_images,stationary_point_found)
            if converged:
                break
            # Break if reaches the max number of iterations set by the user.
            if steps <= self.iter:
                parprint('Maximum number iterations reached. Not converged.')
                break

        parprint('Number of steps performed in total:',self.iter)
        print_cite_mlneb()
        return self

    #def run_neb(self,)

    def set_up_endpoints(self,start,end,prev_calculations=None):
        # Initial state
        if isinstance(start, str):
            start=read(start, '-1:')
        try:
            self.start=copy.deepcopy(start)
        except:
            raise Exception('Initial structure for the NEB was not provided')
        self.eval_and_append(self.start)
        self.energy_is=self.energy

        self.num_atoms = len(self.start)
        self.constraints=self.start.constraints
        if len(self.constraints) < 0:
            self.constraints=None
        if self.constraints is not None:
            self.index_mask=create_mask(self.start, self.constraints)
        
        # Final state
        if isinstance(end, str):
            end=read(end, '-1:')
        try:
            self.end=copy.deepcopy(end)
        except:
            raise Exception('Final structure for the NEB was not provided')
        self.eval_and_append(self.end)
        self.energy_fs=self.energy

        self.path_distance=np.linalg.norm(self.start.get_positions().flatten()-self.end.get_positions().flatten())
        pass

    def save_prev_calculations(self,prev_calculations=None):
        # Store previous calculated data
        if prev_calculations is not None:
            if isinstance(prev_calculations,str):
                if os.path.exists(prev_calculations):
                    prev_calculations=read(prev_calculations,':')
                    for atoms in prev_calculations:
                        self.eval_and_append(atoms)
        if self.restart and prev_calculations is None:
            if os.path.exists(self.trainingset):
                prev_calculations=read(self.trainingset,':')
                for atoms in prev_calculations:
                    self.eval_and_append(atoms)
        pass

    def initial_interpolation(self,interpolation='linear',k=None):
        path=None
        if interpolation not in ['idpp','linear']:
            if isinstance(interpolation,str):
                if os.path.exists(interpolation):
                    path=read(interpolation,':')
            elif isinstance(interpolation,list):
                path=[copy.deepcopy(atoms) for atoms in interpolation]
        
        if path is None:
            if isinstance(self.n_images,float):
                self.n_images=int(self.path_distance/self.n_images)
                if self.n_images<3:
                    self.n_images=3
        else:
            self.n_images=len(path)
        # Set up NEB path
        self.spring = k if k is not None else np.sqrt((self.n_images-1) / self.path_distance)
        self.images=self.make_interpolation(interpolation=interpolation,path=path)
        pass

    def make_interpolation(self,interpolation='linear',path=None):
        images=[copy.deepcopy(self.start)]
        for i in range(1,self.n_images-1):
            image=copy.deepcopy(self.start)
            image.set_calculator(copy.deepcopy(self.mlcalc))
            if path is not None:
                image.set_positions(path[i].get_positions())
            image.set_constraint(self.constraints)
            images.append(image)
        images.append(copy.deepcopy(self.end))
        if path is None:
            neb_interpolation=NEB(images,k=self.spring)
            neb_interpolation.interpolate(method=interpolation,mic=self.mic)
        return images

    def eval_and_append(self,atoms):
        self.energy=atoms.get_potential_energy(force_consistent=self.fc)
        self.forces=atoms.get_forces()
        self.train_images.append(atoms)
        write(self.trainingset,self.train_images)
        self.max_abs_forces=np.max(np.linalg.norm(self.forces,axis=1))
        self.list_gradients.append(self.max_abs_forces)
        pass

    def energy_and_uncertainty(self):
        energies=[img.get_potential_energy() for img in self.images]
        uncertainties=[0]+[img.calc.results['uncertainty'] for img in self.images[1:-1]]+[0]
        return np.array(energies),np.array(uncertainties)

    def evaluate_ase(self,atoms):
        atoms.set_calculator(self.ase_calc)
        self.eval_and_append(atoms)
        self.iter+=1
        pass

    def message_system(self,message,obj=None):
        if self.fullout is True:
            if obj is None:
                parprint(message)
            else:
                parprint(message,obj)
        pass

    def mlneb_opt(self,fmax,max_step,ml_steps,stationary_point_found,org_n_images,local_opt,local_opt_kwargs):
        ml_cycles = 0
        while True:
            if stationary_point_found is True:
                self.n_images = org_n_images

            # Start from last path.
            starting_path = self.images  

            if ml_cycles == 0:
                self.message_system('Using initial path.')
                starting_path=read(self.trajectory,'0:'+str(self.n_images))

            if ml_cycles == 1:
                self.message_system('Using last predicted path.')
                starting_path = read('./all_predicted_paths.traj',str(-self.n_images)+':')

            self.images=self.make_interpolation(interpolation=self.interpolation,path=starting_path)
            
            # Test before optimization:
            self.e_path,self.uncertainty_path=self.energy_and_uncertainty()
            unc_ml=np.max(self.uncertainty_path)
            self.max_target=np.max(self.e_path)

            if unc_ml >= max_step:
                self.message_system('Maximum uncertainty reach in initial path. Early stop.')
                break

            # Perform NEB in the predicted landscape.
            ml_neb = NEB(self.images, climb=True,method=self.neb_method,k=self.spring)
            neb_opt=local_opt(ml_neb,**local_opt_kwargs) if self.fullout else local_opt(ml_neb,logfile=None,**local_opt_kwargs)

            #run
            ml_neb,neb_opt,n_steps_performed=self.mlneb_opt_run(ml_neb,neb_opt,fmax,max_step,ml_steps)

            if n_steps_performed <= ml_steps-1:
                self.message_system('Converged opt. in the predicted landscape.')
                break

            ml_cycles += 1
            self.message_system('ML cycles performed:', ml_cycles)

            if ml_cycles == 2:
                self.message_system('ML process not optimized...not safe... \nChange interpolation or numb. of images.')
                break
        return ml_cycles

    def mlneb_opt_run(self,ml_neb,neb_opt,fmax,max_step,ml_steps):
        ml_converged = False
        n_steps_performed = 0
        while ml_converged is False:
            # Save prev. positions:
            prev_save_positions = [img.get_positions() for img in self.images]

            neb_opt.run(fmax=(fmax * 0.85), steps=1)
            neb_opt.nsteps = 0

            n_steps_performed += 1
            self.e_path,self.uncertainty_path=self.energy_and_uncertainty()
            e_ml=np.max(self.e_path[1:-1])
            unc_ml=np.max(self.uncertainty_path)

            if e_ml >= self.max_target + 0.2:
                for i in range(1, self.n_images-1):
                    self.images[i].positions = prev_save_positions[i]
                self.message_system('Pred. energy above max. energy. \nEarly stop.')
                ml_converged = True

            if unc_ml >= max_step:
                for i in range(1, self.n_images-1):
                    self.images[i].positions = prev_save_positions[i]
                self.message_system('Maximum uncertainty reach. Early stop.')
                ml_converged = True
            if neb_opt.converged():
                ml_converged = True

            if np.isnan(ml_neb.emax):
                self.images = read(self.trajectory, str(-self.n_images) + ':')
                self.message_system('Not converged')
                #n_steps_performed = 10000
                break

            if n_steps_performed > ml_steps-1:
                self.message_system('Not converged yet...')
                ml_converged = True
        return ml_neb,neb_opt,n_steps_performed

    def check_convergence(self,fmax,unc_convergence,org_n_images,stationary_point_found):
        converged=False
        if self.max_abs_forces <= fmax:
            stationary_point_found = True
        # Check whether the evaluated point is a stationary point.
        if self.max_abs_forces <= fmax and self.n_images == org_n_images:
            if np.max(self.uncertainty_path[1:-1]) < unc_convergence:
                # Last path.
                write(self.trajectory, self.images)
                parprint("Congratulations! Your ML NEB is converged. See the final path in file {}".format(self.trajectory))
                converged=True
        return stationary_point_found,converged

    def print_neb(self):
        now=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if self.iter<2:
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

