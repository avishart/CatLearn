from site import execsitecustomize
from turtle import up
import numpy as np
import ase
from ase.io import read
from ase.data import covalent_radii
from ase.calculators.singlepoint import SinglePointCalculator
from copy import deepcopy
from ase.io.trajectory import TrajectoryWriter,TrajectoryReader
from scipy.optimize import dual_annealing
import datetime
#from catlearn import __version__
from ase.parallel import parallel_function
from mpi4py import MPI


class mlgo:
    def __init__(self,slab,ads,ase_calc,ads2=None,mlcalc=None,acq=None,\
                 local_opt=None,local_opt_kwargs={},prev_calculations=None,force_consistent=None,\
                 bounds=None,initial_points=2,norelax_points=10,min_steps=8,mic=True,opt_kwargs={},trajectory='evaluated.traj',fullout=False):
        # Setup given parameters
        self.setup_slab_ads(slab,ads,ads2)
        self.ase_calc=ase_calc
        self.opt_kwargs=opt_kwargs
        self.norelax_points=norelax_points
        self.min_steps=min_steps
        self.step=0
        self.error=0
        self.force_consistent=force_consistent
        self.mic=mic
        self.initial_points=initial_points
        self.fullout=fullout
        # Boundary conditions for adsorbate position and angles
        if bounds is None:
            self.bounds=np.array([[0.0,1.0],[0.0,1.0],[0.0,1.0],[0.0,2*np.pi],[0.0,2*np.pi],[0.0,2*np.pi]])
        else:
            self.bounds=bounds.copy()
        if len(self.bounds)==6 and self.ads2:
            self.bounds=np.concatenate([self.bounds,self.bounds],axis=0)
        # Setup other parameters
        self.energies=np.array([])
        # Make it parallel
        self.parallel_setup()
        # Make trajectory file for calculated structures
        self.trajectory=trajectory
        # Setup the ML calculator
        if mlcalc is None:
            mlcalc=self.get_default_mlcalc()
        self.mlcalc=deepcopy(mlcalc)
        # Select an acquisition function 
        if acq is None:
            from .acquisition import AcqULCB
            acq=AcqULCB(objective='min',unc_convergence=0.05,kappa=3.0,kappamax=5)
        self.acq=deepcopy(acq)
        # Define best candidate 
        self.emin=np.inf
        self.best_candidate=None
        # Use restart structures or make one initial point
        self.use_prev_calculations(prev_calculations)
        # Define local optimizer
        if local_opt is None:
            from ase.optimize import MDMin
            local_opt=MDMin
            local_opt_kwargs=dict(dt=0.05,trajectory='local_opt.traj')
        self.local_opt=local_opt
        self.local_opt_kwargs=local_opt_kwargs
        
    def run(self,fmax=0.05,unc_convergence=0.025,steps=200,max_unc=0.050,ml_steps=2000,ml_chains=3,relax=True,local_steps=500):
        " Run the ML adsorption optimizer "
        # Run global search
        self.acq.unc_convergence=unc_convergence
        # Calculate initial data if data is not given
        self.extra_initial_data(self.initial_points)
        for step in range(1,steps+1):
            # Train ML-Model
            self.train_mlmodel()
            # Search after and find the next candidate for calculation
            candidate=self.find_next_candidate(ml_chains,ml_steps,max_unc,relax,fmax,local_steps)
            # Evaluate candidate
            self.evaluate(candidate)
            # Make print of table
            self.print_statement(step)
            # Check for convergence  
            converged=self.check_convergence(unc_convergence,fmax)
            if converged:
                break
        if converged==False:
            self.message_system('MLOPT did not converge!')
        return self.best_candidate
    
    def setup_slab_ads(self,slab,ads,ads2=None):
        " Setup slab and adsorbate with their constrains"
        # Setup slab
        self.slab=slab.copy()
        self.slab.set_tags(0)
        # Setup adsorbate
        self.ads=ads.copy()
        self.ads.set_tags(1)
        # Center adsorbate structure
        pos=self.ads.get_positions().copy()
        self.ads.positions=pos-np.mean(pos,axis=0)
        self.ads.cell=self.slab.cell.copy()
        # Setup second adsorbate
        if ads2:
            self.ads2=ads2.copy()
            self.ads2.set_tags(2)
            # Center adsorbate structure
            pos=self.ads2.get_positions().copy()
            self.ads2.set_positions(pos-np.mean(pos,axis=0))
            self.ads2.cell=self.slab.cell.copy()
        else:
            self.ads2=None
        # Number of atoms and the constraint used
        slab_ads=self.slab.copy()
        slab_ads.extend(self.ads.copy())
        if self.ads2:
            slab_ads.extend(self.ads2.copy())
        self.number_atoms=len(slab_ads)
        return

    def parallel_setup(self):
        " Setup the parallelization. "
        self.comm=MPI.COMM_WORLD
        self.rank,self.size=self.comm.Get_rank(),self.comm.Get_size()
        return
        
    def place_ads(self,pos_angles):
        " Place the adsorbate in the cell of the surface"
        if self.ads2:
            x,y,z,theta1,theta2,theta3,x2,y2,z2,theta12,theta22,theta32=pos_angles
        else:
            x,y,z,theta1,theta2,theta3=pos_angles
        ads=self.rotation_matrix(self.ads.copy(),[theta1,theta2,theta3])
        ads.set_scaled_positions(ads.get_scaled_positions()+np.array([x,y,z]))
        slab_ads=self.slab.copy()
        slab_ads.extend(ads)
        if self.ads2:
            ads2=self.rotation_matrix(self.ads2.copy(),[theta12,theta22,theta32])
            ads2.set_scaled_positions(ads2.get_scaled_positions()+np.array([x2,y2,z2]))
            slab_ads.extend(ads2)
        slab_ads.wrap()
        return slab_ads
    
    def rotation_matrix(self,ads,angles):
        " Rotate the adsorbate "
        theta1,theta2,theta3=angles
        Rz=np.array([[np.cos(theta1),-np.sin(theta1),0.0],[np.sin(theta1),np.cos(theta1),0.0],[0.0,0.0,1.0]])
        Ry=np.array([[np.cos(theta2),0.0,np.sin(theta2)],[0.0,1.0,0.0],[-np.sin(theta2),0.0,np.cos(theta2)]])
        R=np.matmul(Ry,Rz)
        Rz=np.array([[np.cos(theta3),-np.sin(theta3),0.0],[np.sin(theta3),np.cos(theta3),0.0],[0.0,0.0,1.0]])
        R=np.matmul(Rz,R).T
        ads.positions=np.matmul(ads.get_positions(),R)
        return ads
    
    def evaluate(self,candidate):
        " Caculate energy and forces and add training system to ML-model "
        self.message_system('Performing evaluation.',end='\r')
        # Broadcast the system to all cpus
        if self.rank==0:
            candidate=candidate.copy()
        candidate=self.comm.bcast(candidate,root=0)
        self.ase_calc.reset()
        self.comm.barrier()
        # Calculate the energies and forces
        candidate.calc=self.ase_calc
        candidate.calc.reset()
        forces=candidate.get_forces()
        self.energy_true=candidate.get_potential_energy(force_consistent=self.force_consistent)
        self.step+=1
        self.max_abs_forces=np.max(np.linalg.norm(forces,axis=1))
        self.message_system('Single-point calculation finished.')
        # Store the data
        self.add_training([candidate])
        self.mlcalc.mlmodel.database.save_data(trajectory=self.trajectory)
        # Best new point
        self.best_new_point(candidate,self.energy_true)
        return

    def add_training(self,atoms_list):
        " Add atoms_list data to ML model on rank=0. "
        if self.rank==0:
            self.mlcalc.mlmodel.add_training(atoms_list)
        return

    def best_new_point(self,candidate,energy):
        " Best new candidate due to energy "
        if self.rank==0:
            if energy<=self.emin:
                self.emin=energy
                self.best_candidate=self.mlcalc.mlmodel.database.copy_atoms(candidate)
                self.best_x=self.x.copy()
        self.best_candidate,self.emin=self.comm.bcast([self.best_candidate,self.emin],root=0)
        return
    
    def add_random_ads(self):
        " Generate a random slab-adsorbate structure from bounds "
        sol=dual_annealing(self.dual_func_random,self.bounds,maxfun=100,**self.opt_kwargs)
        self.x=sol['x'].copy()
        slab_ads=self.place_ads(sol['x'])
        return slab_ads

    def dual_func_random(self,pos_angles):
        " Dual annealing object function for random structure "
        from ..regression.tprocess.baseline import Repulsion_calculator
        slab_ads=self.place_ads(pos_angles)
        slab_ads.calc=Repulsion_calculator(r_scale=0.7)
        energy=slab_ads.get_potential_energy()
        return energy
    
    def use_prev_calculations(self,prev_calculations):
        " Use previous calculations to restart ML calculator."
        if prev_calculations is None:
            return
        if isinstance(prev_calculations,str):
            prev_calculations=read(prev_calculations,':')
        # Add calculations to the ML model
        self.add_training(prev_calculations)
        return
    
    def train_mlmodel(self):
        " Train the ML-Model on 1 CPU"
        if self.rank==0:
            self.mlcalc.mlmodel.train_model(verbose=self.fullout)
        self.mlcalc=self.comm.bcast(self.mlcalc,root=0)
        self.comm.barrier()
        pass

    def find_next_candidate(self,ml_chains,ml_steps,max_unc,relax,fmax,local_steps):
        " Find the next candidates by using simulated annealing and then chose the candidate from acquisition "
        # Find candidates from dual-annealing
        candidate,energy,unc,x=None,None,None,None
        candidates={'candidates':[],'energies':[],'uncertainties':[],'x':[]}
        r=0
        for chain in range(ml_chains):
            if self.rank==r:
                np.random.seed(chain)
                self.message_system('Starting global search!',end='\r',rank=r)
                candidate,energy,unc,x=self.dual_annealing(maxiter=ml_steps,**self.opt_kwargs)
                self.message_system('Global search converged',rank=r)
                if relax and self.get_training_len()>=self.norelax_points:
                    if unc<=max_unc:
                        self.message_system('Starting local relaxation',end='\r',rank=r)
                        candidate,energy,unc=self.local_relax(candidate,fmax,max_unc,local_steps=local_steps)
                    else:
                        self.message_system('Stopped due to high uncertainty',rank=r)
                if r!=0:
                    self.comm.send([candidate.copy(),energy,unc,x.copy()],dest=0,tag=self.step)
            candidates,r=self.broadcast_candidates_iter(candidates,r,candidate,energy,unc,x)
        self.comm.barrier()
        if self.rank==0:
            if self.fullout:
                self.message_system('Candidates energies: '+str(candidates['energies']))
                self.message_system('Candidates uncertainties: '+str(candidates['uncertainties']))
            candidate=self.acq_next_point(candidates)
        return candidate

    def acq_next_point(self,candidates):
        " Use acquisition functions to chose the next training point "
        # Calculate the acquisition function for each candidate
        acq_values=self.acq.calculate(np.array(candidates['energies']),np.array(candidates['uncertainties']))
        # Chose the minimum value given by the Acq. class
        argmin=self.acq.choose(acq_values)[0]
        # The next training point
        candidate=candidates['candidates'][argmin].copy()
        self.energy=candidates['energies'][argmin]
        self.unc=np.abs(candidates['uncertainties'][argmin])
        self.x=candidates['x'][argmin].copy()
        return candidate
        
    def check_convergence(self,unc_convergence,fmax):
        " Check if the convergence criteria is fulfilled "
        converged=False
        if self.rank==0:
            if self.min_steps<=self.get_training_len():
                if self.max_abs_forces<=fmax and self.unc<unc_convergence:
                    if np.abs(self.energy_true-self.energy)<=2*unc_convergence:
                        if np.abs(self.energy_true-self.emin)<=unc_convergence:
                            self.message_system('Optimization is successfully completed')
                        converged=True
        converged=self.comm.bcast(converged,root=0)
        return converged
    
    def dual_annealing(self,maxiter=5000,**opt_kwargs):
        " Find the candidates structures, energy and forces using dual annealing "
        self.mlcalc.calculate_forces=False
        sol=dual_annealing(self.dual_func,bounds=self.bounds,maxfun=maxiter,**opt_kwargs)
        slab_ads=self.place_ads(sol['x'])
        slab_ads.calc=self.mlcalc
        energy,unc=self.get_predictions(slab_ads)
        return slab_ads.copy(),energy,unc,sol['x'].copy()
    
    def dual_func(self,pos_angles):
        " Dual annealing object function "
        slab_ads=self.place_ads(pos_angles)
        slab_ads.calc=self.mlcalc
        energy=slab_ads.get_potential_energy()
        if self.acq.kappa:
            unc=slab_ads.calc.get_uncertainty()
            return energy-np.abs(self.acq.kappa)*unc
        return energy
    
    def local_relax(self,candidate,fmax,max_unc,local_steps=200):
        " Perform a local relaxation of the candidate "
        # Get forces and reset calculator
        self.mlcalc.calculate_forces=True
        self.mlcalc.results={}
        #candidate=candidate.copy()
        candidate.calc=self.mlcalc
        dyn=self.local_opt(candidate,**self.local_opt_kwargs)
        if max_unc==False:
            dyn.run(fmax=fmax*0.8,steps=local_steps)
            return candidate
        for i in range(1,local_steps+1):
            candidate_backup=candidate.copy()
            # Take step in local relaxation on surrogate surface
            dyn.run(fmax=fmax*0.8,steps=i)
            energy,unc=self.get_predictions(candidate)
            if unc>=max_unc:
                self.message_system('Relaxation on surrogate surface stopped due to high uncertainty!',rank=self.rank)
                break
            if np.isnan(energy):
                candidate=candidate_backup.copy()
                candidate.calc=self.mlcalc
                energy,unc=self.get_predictions(candidate)
                self.message_system('Stopped due to NaN value in prediction!',rank=self.rank)
                break
            if dyn.converged():
                self.message_system('Relaxation on surrogate surface converged!',rank=self.rank)
                break
        return candidate.copy(),energy,unc

    def get_predictions(self,candidate):
        " Calculate the energies and uncertainties with the ML calculator "
        energy=candidate.get_potential_energy()
        unc=candidate.calc.get_uncertainty()
        return energy,unc

    def get_training_len(self):
        " Get the length of the training set "
        return len(self.mlcalc.mlmodel.database)

    def extra_initial_data(self,initial_points):
        " If only initial and final state is given then a third data point is calculated. "
        candidate=None
        for i in range(initial_points):
            if self.get_training_len()<initial_points:
                if self.rank==0:
                    candidate=self.add_random_ads()
                self.evaluate(candidate)
        return 

    def broadcast_candidates_iter(self,candidates,r,candidate,energy,unc,x):
        " Broadcast iteratively candidate with energy, uncertainty, and position "
        if self.rank==0:
            if r!=0:
                candidate,energy,unc,x=self.comm.recv(source=r,tag=self.step)
            candidates['candidates'].append(candidate)
            candidates['energies'].append(energy)
            candidates['uncertainties'].append(unc)
            candidates['x'].append(x)
        r+=1
        if r>=self.size:
            r=0
        return candidates,r

    def message_system(self,message,obj=None,end='\n',rank=0):
        " Print output on rank=0. "
        if self.fullout is True:
            if self.rank==rank:
                if obj is None:
                    print(message,end=end)
                else:
                    print(message,obj,end=end)
        return
        
    def print_statement(self,step):
        " Print the Global optimization process as a table "
        if self.rank==0:
            now=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            try:
                len(self.print_list)
            except:
                self.print_list=['| Step |        Time         |      True energy      | Uncertainty |  True error  |   fmax   |']

            msg='|{0:6d}| '.format(step)+'{} |'.format(now)
            msg+='{0:23f}|'.format(self.energy_true)
            msg+='{0:13f}|'.format(self.unc)
            msg+='{0:14f}|'.format(np.abs(self.energy_true-self.energy))
            msg+='{0:10f}|'.format(self.max_abs_forces)
            self.print_list.append(msg)
            msg='\n'.join(self.print_list)
            self.message_system(msg)
        pass

    def get_default_mlcalc(self,use_derivatives=True,optimize=True,database_reduction=True,npoints=25):
        " Get a default ML calculator if a calculator is not given. This is a recommended ML calculator."
        from ..regression.gaussianprocess.calculator.mlcalc import MLCalculator
        from ..regression.gaussianprocess.calculator.mlmodel import MLModel
        from ..regression.gaussianprocess.gp.gp import GaussianProcess
        from ..regression.gaussianprocess.kernel.se import SE,SE_Derivative
        from ..regression.gaussianprocess.means import Prior_max
        from ..regression.gaussianprocess.hpfitter import HyperparameterFitter
        from ..regression.gaussianprocess.objectfunctions.factorized_likelihood import FactorizedLogLikelihood
        from ..regression.gaussianprocess.optimizers import run_golden,line_search_scale
        from ..regression.gaussianprocess.calculator.database import Database
        from ..regression.gaussianprocess.fingerprint.invdistances import Inv_distances
        from ..regression.gaussianprocess.pdistributions import Normal_prior
        from ..regression.gaussianprocess.baseline.repulsive import Repulsion_calculator
        # Set a fingerprint
        use_fingerprint=True
        # Use inverse distances as fingerprint
        fp=Inv_distances(reduce_dimensions=True,use_derivatives=use_derivatives,mic=self.mic)
        # Use a GP as the model 
        local_kwargs=dict(tol=1e-5,optimize=True,multiple_max=True)
        kwargs_optimize=dict(local_run=run_golden,maxiter=1000,jac=False,bounds=None,ngrid=80,use_bounds=True,local_kwargs=local_kwargs)
        hpfitter=HyperparameterFitter(FactorizedLogLikelihood(),optimization_method=line_search_scale,opt_kwargs=kwargs_optimize,distance_matrix=True)
        kernel=SE_Derivative(use_fingerprint=use_fingerprint) if use_derivatives else SE(use_fingerprint=use_fingerprint)
        model=GaussianProcess(prior=Prior_max(),kernel=kernel,use_derivatives=use_derivatives,hpfitter=hpfitter)
        # Make the data base ready
        if database_reduction:
            from ..regression.tprocess.calculator.database_reduction import DatabaseLast
            database=DatabaseLast(fingerprint=fp,reduce_dimensions=True,use_derivatives=use_derivatives,negative_forces=True,use_fingerprint=use_fingerprint,npoints=npoints,initial_indicies=[])
        else:
            from ..regression.tprocess.calculator.database import Database
            database=Database(fingerprint=fp,reduce_dimensions=True,use_derivatives=use_derivatives,negative_forces=True,use_fingerprint=use_fingerprint)        
        # Make prior distributions for hyperparameters
        prior=dict(length=np.array([Normal_prior(0.0,2.0)]),noise=np.array([Normal_prior(-9.0,2.0)]))
        # Make the ML model with model and database
        ml_opt_kwargs=dict(retrain=True,prior=prior)
        mlmodel=MLModel(model=model,database=database,baseline=Repulsion_calculator(),optimize=optimize,optimize_kwargs=ml_opt_kwargs)
        # Finally make the calculator
        mlcalc=MLCalculator(mlmodel=mlmodel,calculate_uncertainty=True)
        return mlcalc
    
