import numpy as np
import ase
from ase.io import read
from scipy.optimize import dual_annealing
import datetime
from ase.parallel import world, broadcast
from ..regression.gp.calculator.copy_atoms import copy_atoms
from ..regression.gp.baseline.repulsive import RepulsionCalculator


class MLGO:
    def __init__(
        self,
        slab,
        ads,
        ase_calc,
        ads2=None,
        mlcalc=None,
        acq=None,
        prev_calculations=None,
        use_database_check=True,
        apply_constraint=True,
        force_consistent=None,
        scale_fmax=0.8,
        save_memory=False,
        local_opt=None,
        local_opt_kwargs={},
        opt_kwargs={},
        bounds=None,
        initial_points=2,
        norelax_points=10,
        min_steps=8,
        trajectory="evaluated.traj",
        tabletxt="mlgo_summary.txt",
        full_output=False,
        **kwargs,
    ):
        """
        Machine learning accelerated global adsorption optimization
        with active learning.

        Parameters:
            slab : ASE Atoms object.
                The object of the surface or nanoparticle that
                the adsorbate is adsorped to.
                The energy and forces for the structure is not needed.
            ads : ASE Atoms object.
                The object of the adsorbate in vacuum with same cell size and
                pbc as for the slab.
                The energy and forces for the structure is not needed.
            ase_calc : ASE calculator instance.
                ASE calculator as implemented in ASE.
                See:
                https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html
            ads2 : ASE Atoms object (optional).
                The object of a second adsorbate in vacuum that
                is adsorbed simultaneously with the other adsorbate.
            mlcalc : ML-calculator instance.
                The ML-calculator instance used as surrogate surface.
                A default ML-model is used if mlcalc is None.
            acq : Acquisition instance.
                The Acquisition instance used for calculating
                the acq. function and choose a candidate to calculate next.
                A default Acquisition instance is used if acq is None.
            prev_calculations : Atoms list or ASE Trajectory file.
                (optional) The user can feed previously calculated data for the
                same hypersurface. The previous calculations must be fed as an
                Atoms list or Trajectory file.
            use_database_check : bool
                Whether to check if the new structure is within the database.
                If it is in the database, the structure is rattled.
            apply_constraint : boolean
                Whether to apply the constrains of the ASE Atoms instance
                to the calculated forces.
                By default (apply_constraint=True) forces are 0 for
                constrained atoms and directions.
            force_consistent : boolean or None.
                Use force-consistent energy calls (as opposed to the energy
                extrapolated to 0 K). By default (force_consistent=None) uses
                force-consistent energies if available in the calculator, but
                falls back to force_consistent=False if not.
            scale_fmax : float
                The scaling of the fmax for the ML-NEB runs.
                It makes the path converge tighter on surrogate surface.
            save_memory : bool
                Whether to only train the ML calculator and store
                all objects on one CPU.
                If save_memory==True then parallel optimization of
                the hyperparameters can not be achived.
                If save_memory==False no MPI object is used.
            local_opt : ASE local optimizer Object.
                A local optimizer object from ASE.
                If None is given then FIRE is used.
            local_opt_kwargs : dict.
                Arguments used for the ASE local optimizer.
            bounds : (6,2) or (12,2) ndarray (optional).
                The boundary conditions used for the global optimization in
                form of the simulated annealing.
                The boundary conditions are the x, y, and z coordinates of
                the center of the adsorbate and 3 rotations.
                Same boundary conditions can be set for the second adsorbate
                if chosen.
            initial_points : int.
                Number of generated initial structures used for training
                the ML calculator if no previous data is given.
            norelax_points : int.
                The number of structures used for training before
                local relaxation of the structures after
                the global optimization is activated.
            min_steps : int.
                The minimum number of iterations before convergence is checked.
            opt_kwargs : dict.
                Arguments used for the simulated annealing method.
            trajectory : string.
                Trajectory filename to store the evaluated training data.
            tabletxt : string
                Name of the .txt file where the summary table is printed.
                It is not saved to the file if tabletxt=None.
            full_output : bool.
                Whether to print on screen the full output (True).
        """
        # Setup parallelization
        self.parallel_setup(save_memory)
        # Setup given parameters
        self.setup_slab_ads(slab, ads, ads2)
        self.opt_kwargs = opt_kwargs
        self.norelax_points = norelax_points
        self.min_steps = min_steps
        self.use_database_check = use_database_check
        self.initial_points = initial_points
        self.full_output = full_output
        # Set candidate instance with ASE calculator
        self.candidate = self.slab_ads.copy()
        self.candidate.calc = ase_calc
        self.apply_constraint = apply_constraint
        self.force_consistent = force_consistent
        # Set initial parameters
        self.step = 0
        self.error = 0
        self.energies = []
        self.emin = np.inf
        self.best_candidate = None
        # Boundary conditions for adsorbate position and angles
        if bounds is None:
            self.bounds = np.array(
                [
                    [0.0, 1.0],
                    [0.0, 1.0],
                    [0.0, 1.0],
                    [0.0, 2 * np.pi],
                    [0.0, 2 * np.pi],
                    [0.0, 2 * np.pi],
                ]
            )
        else:
            self.bounds = bounds.copy()
        if len(self.bounds) == 6 and self.ads2 is not None:
            self.bounds = np.concatenate([self.bounds, self.bounds], axis=0)
        # Make trajectory file for calculated structures
        self.trajectory = trajectory
        # Summary table file name
        self.tabletxt = tabletxt
        # Setup the ML calculator
        self.set_mlcalc(mlcalc, save_memory=save_memory)
        self.set_verbose(verbose=full_output)
        # Select an acquisition function
        self.set_acq(acq)
        # Scale the fmax on the surrogate surface
        self.scale_fmax = scale_fmax
        # Use restart structures or make one initial point
        self.use_prev_calculations(prev_calculations)
        # Set local optimizer
        self.set_local_opt(
            local_opt=local_opt,
            local_opt_kwargs=local_opt_kwargs,
        )

    def run(
        self,
        fmax=0.05,
        unc_convergence=0.025,
        steps=200,
        max_unc=0.25,
        ml_steps=2000,
        ml_chains=3,
        relax=True,
        local_steps=500,
        seed=0,
        **kwargs,
    ):
        """
        Run the ML adsorption optimizer

        Parameters:
            fmax : float
                Convergence criteria (in eV/Angs).
            unc_convergence: float
                Maximum uncertainty for convergence (in eV).
            steps : int
                Maximum number of evaluations.
            max_unc : float
                Early stopping criteria.
                Maximum uncertainty allowed before local optimization.
            ml_steps : int
                Maximum number of steps for the global optimization
                on the predicted landscape.
            ml_chains : int
                The number of parallel chains in the simulated annealing.
            relax : bool
                Whether to perform local optimization after
                the global optimization.
            local_steps : int
                Maximum number of steps for the local optimization
                on the predicted landscape.
            seed : int (optional)
                The random seed.
        """
        # Set the random seed
        np.random.seed(seed)
        # Update the acquisition function
        self.acq.update_arguments(unc_convergence=unc_convergence)
        # Calculate initial data if enough data is not given
        self.extra_initial_data(self.initial_points)
        # Run global search
        for step in range(1, steps + 1):
            # Train ML-Model
            self.train_mlmodel()
            # Search after and find the next candidate for calculation
            candidate = self.find_next_candidate(
                ml_chains,
                ml_steps,
                max_unc,
                relax,
                fmax * self.scale_fmax,
                local_steps,
            )
            # Evaluate candidate
            self.evaluate(candidate)
            # Make print of table
            self.print_statement(step)
            # Check for convergence
            self.converging = self.check_convergence(unc_convergence, fmax)
            if self.converging:
                break
        if self.converging is False:
            self.message_system("MLGO did not converge!")
        return self.best_candidate

    def get_atoms(self):
        "Return the best candidate structure."
        return self.best_candidate

    def setup_slab_ads(self, slab, ads, ads2=None):
        "Setup slab and adsorbate with their constrains"
        # Setup slab
        self.slab = slab.copy()
        self.slab.set_tags(0)
        # Setup adsorbate
        self.ads = ads.copy()
        self.ads.set_tags(1)
        # Center adsorbate structure
        pos = self.ads.get_positions()
        self.ads.set_positions(pos - np.mean(pos, axis=0))
        self.ads.cell = self.slab.cell.copy()
        self.ads.pbc = self.slab.pbc.copy()
        # Setup second adsorbate
        if ads2:
            self.ads2 = ads2.copy()
            self.ads2.set_tags(2)
            # Center adsorbate structure
            pos = self.ads2.get_positions()
            self.ads2.set_positions(pos - np.mean(pos, axis=0))
            self.ads2.cell = self.slab.cell.copy()
            self.ads2.pbc = self.slab.pbc.copy()
        else:
            self.ads2 = None
        # Number of atoms and the constraint used
        self.slab_ads = self.slab.copy()
        self.slab_ads.extend(self.ads.copy())
        if self.ads2:
            self.slab_ads.extend(self.ads2.copy())
        self.number_atoms = len(self.slab_ads)
        return

    def parallel_setup(self, save_memory=False, **kwargs):
        "Setup the parallelization."
        self.save_memory = save_memory
        self.rank = world.rank
        self.size = world.size
        return self

    def place_ads(self, pos_angles):
        "Place the adsorbate in the cell of the surface"
        if self.ads2:
            (
                x,
                y,
                z,
                theta1,
                theta2,
                theta3,
                x2,
                y2,
                z2,
                theta12,
                theta22,
                theta32,
            ) = pos_angles
        else:
            x, y, z, theta1, theta2, theta3 = pos_angles
        ads = self.rotation_matrix(self.ads.copy(), [theta1, theta2, theta3])
        spos = ads.get_scaled_positions()
        ads.set_scaled_positions(spos + np.array([x, y, z]))
        slab_ads = self.slab.copy()
        slab_ads.extend(ads)
        if self.ads2:
            ads2 = self.rotation_matrix(
                self.ads2.copy(),
                [theta12, theta22, theta32],
            )
            spos = ads2.get_scaled_positions()
            ads2.set_scaled_positions(spos + np.array([x2, y2, z2]))
            slab_ads.extend(ads2)
        slab_ads.wrap()
        return slab_ads

    def rotation_matrix(self, ads, angles):
        "Rotate the adsorbate"
        theta1, theta2, theta3 = angles
        Rz = np.array(
            [
                [np.cos(theta1), -np.sin(theta1), 0.0],
                [np.sin(theta1), np.cos(theta1), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        Ry = np.array(
            [
                [np.cos(theta2), 0.0, np.sin(theta2)],
                [0.0, 1.0, 0.0],
                [-np.sin(theta2), 0.0, np.cos(theta2)],
            ]
        )
        R = np.matmul(Ry, Rz)
        Rz = np.array(
            [
                [np.cos(theta3), -np.sin(theta3), 0.0],
                [np.sin(theta3), np.cos(theta3), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        R = np.matmul(Rz, R).T
        ads.set_positions(np.matmul(ads.get_positions(), R))
        return ads

    def evaluate(self, candidate):
        "Caculate energy and forces and add training system to ML-model"
        # Ensure that the candidate is not already in the database
        if self.use_database_check:
            candidate = self.ensure_not_in_database(candidate)
        # Broadcast the system to all cpus
        if self.rank == 0:
            candidate = candidate.copy()
        candidate = broadcast(candidate, root=0)
        # Calculate the energies and forces
        self.message_system("Performing evaluation.", end="\r")
        self.candidate.set_positions(candidate.get_positions())
        forces = self.candidate.get_forces(
            apply_constraint=self.apply_constraint
        )
        self.energy_true = self.candidate.get_potential_energy(
            force_consistent=self.force_consistent
        )
        self.step += 1
        self.message_system("Single-point calculation finished.")
        # Store the data
        self.max_abs_forces = np.nanmax(np.linalg.norm(forces, axis=1))
        self.add_training([self.candidate])
        self.mlcalc.save_data(trajectory=self.trajectory)
        # Best new point
        self.best_new_point(self.candidate, self.energy_true)
        return

    def add_training(self, atoms_list):
        "Add atoms_list data to ML model on rank=0."
        self.mlcalc.add_training(atoms_list)
        return self.mlcalc

    def best_new_point(self, candidate, energy):
        "Best new candidate due to energy"
        if self.rank == 0:
            if energy <= self.emin:
                self.emin = energy
                self.best_candidate = copy_atoms(candidate)
                self.best_x = self.x.copy()
            # Save the energy
            self.energies.append(energy)
        # Broadcast convergence statement if MPI is used
        self.best_candidate, self.emin = broadcast(
            [self.best_candidate, self.emin],
            root=0,
        )
        return self.best_candidate

    def add_random_ads(self):
        "Generate a random slab-adsorbate structure from bounds"
        sol = dual_annealing(
            self.dual_func_random,
            self.bounds,
            maxfun=100,
            **self.opt_kwargs,
        )
        self.x = sol["x"].copy()
        slab_ads = self.place_ads(sol["x"])
        return slab_ads

    def dual_func_random(self, pos_angles):
        "Dual annealing object function for random structure"
        slab_ads = self.place_ads(pos_angles)
        slab_ads.calc = RepulsionCalculator(
            r_scale=0.7,
            reduce_dimensions=True,
            power=10,
            periodic_softmax=True,
            wrap=True,
        )
        energy = slab_ads.get_potential_energy()
        return energy

    def use_prev_calculations(self, prev_calculations):
        "Use previous calculations to restart ML calculator."
        if prev_calculations is None:
            return
        if isinstance(prev_calculations, str):
            prev_calculations = read(prev_calculations, ":")
        # Add calculations to the ML model
        self.add_training(prev_calculations)
        return

    def set_verbose(self, verbose, **kwargs):
        "Set verbose of MLModel."
        self.mlcalc.mlmodel.update_arguments(verbose=verbose)
        return

    def train_mlmodel(self):
        "Train the ML model."
        if self.save_memory:
            if self.rank != 0:
                return self.mlcalc
        # Update database with the points of interest
        self.update_database_arguments(point_interest=self.best_candidate)
        # Train the ML model
        self.mlcalc.train_model()
        return self.mlcalc

    def is_in_database(self, atoms, **kwargs):
        "Check if the ASE Atoms is in the database."
        return self.mlcalc.is_in_database(atoms, **kwargs)

    def update_database_arguments(self, point_interest=None, **kwargs):
        "Update the arguments in the database."
        self.mlcalc.update_database_arguments(
            point_interest=point_interest,
            **kwargs,
        )
        return self

    def ensure_not_in_database(self, atoms, perturb=0.01, **kwargs):
        """
        Ensure the ASE Atoms object is not in database by perturb it
        if it is.
        """
        # Return atoms if it does not exist
        if atoms is None:
            return atoms
        # Check if atoms object is in the database
        if self.is_in_database(atoms, **kwargs):
            # Get positions
            pos = atoms.get_positions()
            # Rattle the positions
            pos = pos + np.random.uniform(
                low=-perturb,
                high=perturb,
                size=pos.shape,
            )
            atoms.set_positions(pos)
            self.message_system(
                "The system is rattled, since it is already in the database."
            )
        return atoms

    def find_next_candidate(
        self,
        ml_chains,
        ml_steps,
        max_unc,
        relax,
        fmax,
        local_steps,
        **kwargs,
    ):
        """
        Find the next candidates by using simulated annealing and
        then chose the candidate from acquisition.
        """
        # Return None if memory is saved and therefore not in parallel
        if self.save_memory and self.rank != 0:
            return None
        # Initialize candidate dictionary
        candidate, energy, unc, x = None, None, None, None
        candidates = {
            "candidates": [],
            "energies": [],
            "uncertainties": [],
            "x": [],
        }
        r = 0
        # Perform multiple optimizations
        for chain in range(ml_chains):
            # Set a unique optimization for each chain
            np.random.seed(chain)
            if not self.save_memory:
                r = chain % self.size
            if self.rank == r:
                # Find candidates from a global simulated annealing search
                self.message_system(
                    "Starting global search!", end="\r", rank=r
                )
                candidate, energy, unc, x = self.dual_annealing(
                    maxiter=ml_steps,
                    **self.opt_kwargs,
                )
                self.message_system("Global search converged", rank=r)
                # Do a local relaxation if the conditions are met
                if relax and (
                    self.get_training_set_size() >= self.norelax_points
                ):
                    if unc <= max_unc:
                        self.message_system(
                            "Starting local relaxation", end="\r", rank=r
                        )
                        candidate, energy, unc = self.local_relax(
                            candidate,
                            fmax,
                            max_unc,
                            local_steps=local_steps,
                            rank=r,
                        )
                    else:
                        self.message_system(
                            "No local relaxation due to high uncertainty",
                            rank=r,
                        )
                # Append the newest candidate
                candidates = self.append_candidates(
                    candidates,
                    candidate,
                    energy,
                    unc,
                    x,
                )
        # Broadcast all the candidates
        if not self.save_memory:
            candidates = self.broadcast_candidates(candidates)
        # Print the energies and uncertainties for the new candidates
        self.message_system(
            "Candidates energies: " + str(candidates["energies"])
        )
        self.message_system(
            "Candidates uncertainties: " + str(candidates["uncertainties"])
        )
        # Find the new best candidate from the acquisition function
        candidate = self.choose_candidate(candidates)
        return candidate

    def choose_candidate(self, candidates):
        "Use acquisition functions to chose the next training point"
        # Calculate the acquisition function for each candidate
        acq_values = self.acq.calculate(
            np.array(candidates["energies"]),
            np.array(candidates["uncertainties"]),
        )
        # Chose the minimum value given by the Acq. class
        i_min = self.acq.choose(acq_values)[0]
        # The next training point
        candidate = candidates["candidates"][i_min].copy()
        self.energy = candidates["energies"][i_min]
        self.unc = np.abs(candidates["uncertainties"][i_min])
        self.x = candidates["x"][i_min].copy()
        return candidate

    def check_convergence(self, unc_convergence, fmax):
        "Check if the convergence criteria are fulfilled"
        converged = False
        if self.rank == 0:
            # Check the minimum number of steps have been performed
            if self.min_steps <= self.get_training_set_size():
                # Check the force and uncertainty criteria are met
                if self.max_abs_forces <= fmax and self.unc < unc_convergence:
                    # Check the true energy deviation match
                    # the uncertainty prediction
                    e_dif = np.abs(self.energy_true - self.energy)
                    if e_dif <= 2.0 * unc_convergence:
                        # Check the predicted structure has
                        # the lowest observed energy
                        em_dif = np.abs(self.energy - self.emin)
                        if em_dif <= 2.0 * unc_convergence:
                            self.message_system("Optimization is converged.")
                        converged = True
        # Broadcast convergence statement if MPI is used
        converged = broadcast(converged, root=0)
        return converged

    def dual_annealing(self, maxiter=5000, **opt_kwargs):
        """
        Find the candidates structures, energy and forces using dual annealing.
        """
        # Deactivate force predictions
        self.mlcalc.update_arguments(calculate_forces=False)
        # Perform simulated annealing
        sol = dual_annealing(
            self.dual_func,
            bounds=self.bounds,
            maxfun=maxiter,
            **opt_kwargs,
        )
        # Reconstruct the final structure
        slab_ads = self.place_ads(sol["x"])
        # Get the energy and uncertainty predictions
        slab_ads.calc = self.mlcalc
        energy, unc = self.get_predictions(slab_ads)
        return slab_ads.copy(), energy, unc, sol["x"].copy()

    def dual_func(self, pos_angles):
        "Dual annealing object function"
        # Construct the structure
        slab_ads = self.place_ads(pos_angles)
        # Predict the energy and uncertainty
        slab_ads.calc = self.mlcalc
        energy = slab_ads.get_potential_energy()
        unc = slab_ads.calc.get_uncertainty(slab_ads)
        # Calculate the acquisition function
        return self.acq.calculate(energy, uncertainty=unc)

    def local_relax(
        self,
        candidate,
        fmax,
        max_unc,
        local_steps=200,
        rank=0,
        **kwargs,
    ):
        "Perform a local relaxation of the candidate"
        # Activate force predictions and reset calculator
        self.mlcalc.update_arguments(calculate_forces=True)
        self.mlcalc.reset()
        candidate = candidate.copy()
        candidate.calc = self.mlcalc
        # Initialize local optimization
        with self.local_opt(candidate, **self.local_opt_kwargs) as dyn:
            if max_unc is False or max_unc is None:
                converged, candidate = self.local_relax_no_max_unc(
                    dyn,
                    candidate,
                    fmax=fmax,
                    local_steps=local_steps,
                    **kwargs,
                )
            else:
                converged, candidate = self.local_relax_max_unc(
                    dyn,
                    candidate,
                    fmax=fmax,
                    max_unc=max_unc,
                    local_steps=local_steps,
                    rank=rank,
                    **kwargs,
                )
            # Calculate the energy and uncertainty
            energy, unc = self.get_predictions(candidate)
        return candidate.copy(), energy, unc

    def local_relax_no_max_unc(
        self,
        dyn,
        candidate,
        fmax,
        local_steps=200,
        **kwargs,
    ):
        "Run the local optimization without checking uncertainties."
        dyn.run(fmax=fmax, steps=local_steps)
        return dyn.converged(), candidate

    def local_relax_max_unc(
        self,
        dyn,
        candidate,
        fmax,
        max_unc,
        local_steps=200,
        rank=0,
        **kwargs,
    ):
        "Run the local optimization with checking uncertainties."
        for i in range(1, local_steps + 1):
            candidate_backup = candidate.copy()
            # Take a step in local relaxation on surrogate surface
            if ase.__version__ >= "3.23":
                dyn.run(fmax=fmax, steps=1)
            else:
                dyn.run(fmax=fmax, steps=i)
            energy, unc = self.get_predictions(candidate)
            # Check if the uncertainty is too large
            if unc >= max_unc:
                self.message_system(
                    "Relaxation on surrogate surface stopped due "
                    "to high uncertainty!",
                    rank=rank,
                )
                break
            # Check if there is a problem with prediction
            if np.isnan(energy):
                candidate = candidate_backup.copy()
                candidate.calc = self.mlcalc
                self.message_system(
                    "Stopped due to NaN value in prediction!", rank=rank
                )
                break
            # Check if the optimization is converged on the predicted surface
            if dyn.converged():
                self.message_system(
                    "Relaxation on surrogate surface converged!", rank=rank
                )
                break
            # Check the number of steps
            if dyn.get_number_of_steps() >= local_steps:
                break
        return dyn.converged(), candidate

    def get_predictions(self, candidate):
        "Calculate the energies and uncertainties with the ML calculator"
        unc = candidate.calc.get_uncertainty(candidate)
        energy = candidate.get_potential_energy()
        return energy, unc

    def get_training_set_size(self):
        "Get the size of the training set"
        return self.mlcalc.get_training_set_size()

    def extra_initial_data(self, initial_points):
        """
        If only initial and final state is given then a third data point
        is calculated.
        """
        candidate = None
        while self.get_training_set_size() < initial_points:
            candidate = self.add_random_ads()
            self.evaluate(candidate)
        return self.get_training_set_size()

    def append_candidates(
        self,
        candidates,
        candidate,
        energy,
        unc,
        x,
        **kwargs,
    ):
        "Update the candidates by appending the newest one."
        candidates["candidates"].append(candidate)
        candidates["energies"].append(energy)
        candidates["uncertainties"].append(unc)
        candidates["x"].append(x)
        return candidates

    def broadcast_candidates(self, candidates, **kwargs):
        "Broadcast candidates with energies, uncertainties, and positions."
        candidates_broad = {
            "candidates": [],
            "energies": [],
            "uncertainties": [],
            "x": [],
        }
        for r in range(self.size):
            cand_r = broadcast(candidates, root=r)
            for n in range(len(cand_r["candidates"])):
                candidates_broad = self.append_candidates(
                    candidates_broad,
                    cand_r["candidates"][n],
                    cand_r["energies"][n],
                    cand_r["uncertainties"][n],
                    cand_r["x"][n],
                )
        return candidates_broad

    def get_energy_deviation(self, **kwargs):
        """
        Get the absolute energy difference between
        the predicted and true energy.
        """
        return np.abs(self.energy_true - self.energy)

    def message_system(self, message, obj=None, end="\n", rank=0):
        "Print output once."
        if self.full_output is True:
            if self.rank == rank:
                if obj is None:
                    print(message, end=end)
                else:
                    print(message, obj, end=end)
            else:
                if self.rank == 0:
                    if obj is None:
                        print(message, end=end)
                    else:
                        print(message, obj, end=end)
        return

    def converged(self):
        "Whether MLGO is converged."
        return self.converging

    def set_mlcalc(self, mlcalc, save_memory=None, **kwargs):
        """
        Setup the ML calculator.

        Parameters:
            mlcalc : ML-calculator instance.
                The ML-calculator instance used as surrogate surface.
                A default ML-model is used if mlcalc is None.
            save_memory : bool
                Whether to only train the ML calculator and store
                all objects on one CPU.
                If save_memory==True then parallel optimization of
                the hyperparameters can not be achived.
                If save_memory==False no MPI object is used.

        Returns:
            self: The object itself.
        """
        if mlcalc is None:
            from ..regression.gp.calculator import (
                get_default_mlmodel,
                MLCalculator,
            )
            from ..regression.gp.fingerprint import (
                SortedDistances,
            )

            # Check if the save_memory is given
            if save_memory is None:
                try:
                    save_memory = self.save_memory
                except Exception:
                    raise Exception("The save_memory is not given.")

            fp = SortedDistances(
                reduce_dimensions=True,
                use_derivatives=True,
                periodic_softmax=True,
                wrap=True,
            )
            baseline = RepulsionCalculator(
                reduce_dimensions=True,
                power=10,
                periodic_softmax=True,
                wrap=True,
            )
            mlmodel = get_default_mlmodel(
                model="gp",
                fp=fp,
                baseline=baseline,
                use_derivatives=True,
                parallel=(not save_memory),
                database_reduction=False,
            )
            self.mlcalc = MLCalculator(mlmodel=mlmodel)
        else:
            self.mlcalc = mlcalc
        return self

    def set_acq(self, acq=None, **kwargs):
        """
        Set the acquisition function.

        Parameters:
            acq : Acquisition class instance.
                The Acquisition instance used for calculating
                the acq. function and choose a candidate to calculate next.
                If None is given then LCB is used.

        Returns:
            self: The object itself.
        """
        if acq is None:
            from .acquisition import AcqLCB

            self.acq = AcqLCB(objective="min", kappa=3.0)
        else:
            self.acq = acq.copy()
        return self

    def set_local_opt(self, local_opt=None, local_opt_kwargs={}, **kwargs):
        """
        Save local optimizer.

        Parameters:
            local_opt : ASE local optimizer Object.
                A local optimizer object from ASE.
                If None is given then FIRE is used.
            local_opt_kwargs : dict
                Arguments used for the ASE local optimizer.

        Returns:
            self: The object itself.
        """
        local_opt_kwargs_default = dict()
        if not self.full_output:
            local_opt_kwargs_default["logfile"] = None
        if local_opt is None:
            from ase.optimize import FIRE

            local_opt = FIRE
            local_opt_kwargs_default.update(
                dict(
                    dt=0.05,
                    maxstep=0.2,
                    a=1.0,
                    astart=1.0,
                    fa=0.999,
                    downhill_check=True,
                )
            )
        self.local_opt = local_opt
        local_opt_kwargs_default.update(local_opt_kwargs)
        self.local_opt_kwargs = local_opt_kwargs_default.copy()
        return self

    def save_mlcalc(self, filename="mlcalc.pkl", **kwargs):
        """
        Save the ML calculator object to a file.

        Parameters:
            filename : str
                The name of the file where the object is saved.

        Returns:
            self: The object itself.
        """
        self.mlcalc.save_mlcalc(filename, **kwargs)
        return self

    def make_summary_table(self, step, **kwargs):
        "Make the summary of the Global optimization process as table."
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            len(self.print_list)
        except Exception:
            self.print_list = [
                "| Step |        Time         |      True energy      | "
                "Uncertainty |  True error  |   fmax   |"
            ]
        msg = "|{0:6d}| ".format(step)
        msg += "{} |".format(now)
        msg += "{0:23f}|".format(self.energy_true)
        msg += "{0:13f}|".format(self.unc)
        msg += "{0:14f}|".format(self.get_energy_deviation())
        msg += "{0:10f}|".format(self.max_abs_forces)
        self.print_list.append(msg)
        msg = "\n".join(self.print_list)
        return msg

    def save_summary_table(self, **kwargs):
        "Save the summary table in the .txt file."
        if self.tabletxt is not None:
            with open(self.tabletxt, "w") as thefile:
                msg = "\n".join(self.print_list)
                thefile.writelines(msg)
        return

    def print_statement(self, step, **kwargs):
        "Print the Global optimization process as a table"
        msg = ""
        if self.rank == 0:
            msg = self.make_summary_table(step, **kwargs)
            self.save_summary_table()
            self.message_system(msg)
        return msg
