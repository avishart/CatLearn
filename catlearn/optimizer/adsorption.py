from .method import OptimizerMethod
from ase.parallel import world
from ase.constraints import FixAtoms, FixBondLengths
import itertools
import numpy as np
from scipy.optimize import dual_annealing


class AdsorptionOptimizer(OptimizerMethod):
    def __init__(
        self,
        slab,
        adsorbate,
        adsorbate2=None,
        bounds=None,
        opt_kwargs={},
        parallel_run=False,
        comm=world,
        verbose=False,
        **kwargs,
    ):
        """
        The AdsorptionOptimizer is used to run an global optimization of
        an adsorption on a surface.
        A single structure will be created and optimized.
        Simulated annealing will be used to global optimize the structure.
        The AdsorptionOptimizer is applicable to be used with
        active learning.

        Parameters:
            slab: Atoms instance
                The slab structure.
            adsorbate: Atoms instance
                The adsorbate structure.
            adsorbate2: Atoms instance (optional)
                The second adsorbate structure.
            bounds : (6,2) or (12,2) ndarray (optional).
                The boundary conditions used for the global optimization in
                form of the simulated annealing.
                The boundary conditions are the x, y, and z coordinates of
                the center of the adsorbate and 3 rotations.
                Same boundary conditions can be set for the second adsorbate
                if chosen.
            opt_kwargs: dict
                The keyword arguments for the simulated annealing optimization.
            parallel_run: bool
                If True, the optimization will be run in parallel.
            comm: ASE communicator instance
                The communicator instance for parallelization.
            verbose: bool
                Whether to print the full output (True) or
                not (False).
        """
        # Create the atoms object from the slab and adsorbate
        self.create_slab_ads(slab, adsorbate, adsorbate2)
        # Create the boundary conditions
        self.setup_bounds(bounds)
        # Set the parameters
        self.update_arguments(
            opt_kwargs=opt_kwargs,
            parallel_run=parallel_run,
            comm=comm,
            verbose=verbose,
            **kwargs,
        )

    def get_structures(self, get_all=True, **kwargs):
        structures = self.copy_atoms(self.optimizable)
        structures.set_constraint(self.constraints_org)
        return structures

    def create_slab_ads(self, slab, adsorbate, adsorbate2=None):
        """
        Create the structure for the adsorption optimization.

        Parameters:
            slab: Atoms object
                The slab structure.
            adsorbate: Atoms object
                The adsorbate structure.
            adsorbate2: Atoms object (optional)
                The second adsorbate structure.

        Returns:
            self: object
                The object itself.
        """
        # Check the slab and adsorbate are given
        if slab is None or adsorbate is None:
            raise Exception("The slab and adsorbate must be given!")
        # Setup the slab
        self.n_slab = len(slab)
        self.slab = slab.copy()
        self.slab.set_tags(0)
        optimizable = self.slab.copy()
        # Setup the adsorbate
        self.n_ads = len(adsorbate)
        self.adsorbate = adsorbate.copy()
        self.adsorbate.set_tags(1)
        self.adsorbate.cell = optimizable.cell.copy()
        self.adsorbate.pbc = optimizable.pbc.copy()
        pos_ads = self.adsorbate.get_positions()
        pos_ads -= pos_ads.mean(axis=0)
        self.adsorbate.set_positions(pos_ads)
        optimizable.extend(self.adsorbate.copy())
        # Setup the adsorbate2
        if adsorbate2 is not None:
            self.n_ads2 = len(adsorbate2)
            self.adsorbate2 = adsorbate2.copy()
            self.adsorbate2.set_tags(2)
            self.adsorbate2.cell = optimizable.cell.copy()
            self.adsorbate2.pbc = optimizable.pbc.copy()
            pos_ads2 = self.adsorbate2.get_positions()
            pos_ads2 -= pos_ads2.mean(axis=0)
            self.adsorbate2.set_positions(pos_ads2)
            optimizable.extend(self.adsorbate2.copy())
        else:
            self.n_ads2 = 0
            self.adsorbate2 = None
        # Get the full number of atoms
        self.natoms = len(optimizable)
        # Store the positions and cell
        self.positions0 = optimizable.get_positions().copy()
        self.cell = np.array(optimizable.get_cell())
        # Store the original constraints
        self.constraints_org = [c.copy() for c in optimizable.constraints]
        # Make constraints for optimization
        self.constraints_used = [FixAtoms(indices=list(range(self.n_slab)))]
        self.constraints_new = [FixAtoms(indices=list(range(self.n_slab)))]
        if self.n_ads > 1:
            pairs = list(
                itertools.combinations(
                    range(self.n_slab, self.n_slab + self.n_ads),
                    2,
                )
            )
            self.constraints_new.append(FixBondLengths(pairs=pairs))
        if self.n_ads2 > 1:
            pairs = list(
                itertools.combinations(
                    range(self.n_slab + self.n_ads, self.natoms),
                    2,
                )
            )
            self.constraints_new.append(FixBondLengths(pairs=pairs))
        optimizable.set_constraint(self.constraints_new)
        # Setup the optimizable structure
        self.setup_optimizable(optimizable)
        return self

    def setup_bounds(self, bounds=None):
        """
        Setup the boundary conditions for the global optimization.

        Parameters:
            bounds : (6,2) or (12,2) ndarray (optional).
                The boundary conditions used for the global optimization in
                form of the simulated annealing.
                The boundary conditions are the x, y, and z coordinates of
                the center of the adsorbate and 3 rotations.
                Same boundary conditions can be set for the second adsorbate
                if chosen.

        Returns:
            self: object
                The object itself.
        """
        # Check the bounds are given
        if bounds is None:
            # Make default bounds
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
        # Check the bounds have the correct shape
        if self.bounds.shape != (6, 2) and self.bounds.shape != (12, 2):
            raise Exception("The bounds must have shape (6,2) or (12,2)!")
        # Check if the bounds are for two adsorbates
        if self.n_ads2 > 0 and self.bounds.shape[0] == 6:
            self.bounds = np.concatenate([self.bounds, self.bounds], axis=0)
        return self

    def run(
        self,
        fmax=0.05,
        steps=1000000,
        max_unc=None,
        unc_convergence=None,
        **kwargs,
    ):
        # Use original constraints
        self.optimizable.set_constraint(self.constraints_used)
        # Perform the simulated annealing
        sol = dual_annealing(
            self.evaluate_value,
            bounds=self.bounds,
            maxfun=steps,
            **self.opt_kwargs,
        )

        # Set the positions
        self.evaluate_value(sol["x"])
        # Set the new constraints
        self.optimizable.set_constraint(self.constraints_new)
        # Calculate the maximum force to check convergence
        if fmax > self.get_fmax():
            # Check if the optimization is converged
            self._converged = self.check_convergence(
                converged=True,
                max_unc=max_unc,
                unc_convergence=unc_convergence,
            )
        return self._converged

    def is_energy_minimized(self):
        return True

    def is_parallel_allowed(self):
        return False

    def update_arguments(
        self,
        slab=None,
        adsorbate=None,
        adsorbate2=None,
        bounds=None,
        opt_kwargs=None,
        parallel_run=None,
        comm=None,
        verbose=None,
        **kwargs,
    ):
        """
        Update the instance with its arguments.
        The existing arguments are used if they are not given.

        Parameters:
            atoms: Atoms object
                The atoms object to be optimized.
            parallel_run: bool
                If True, the optimization will be run in parallel.
            comm: ASE communicator object
                The communicator object for parallelization.
            verbose: bool
                Whether to print the full output (True) or
                not (False).
        """
        # Set the communicator
        if comm is not None:
            self.comm = comm
            self.rank = comm.rank
            self.size = comm.size
        # Set the verbose
        if verbose is not None:
            self.verbose = verbose
        # Create the atoms object from the slab and adsorbate
        if slab is not None and adsorbate is not None:
            self.create_slab_ads(slab, adsorbate, adsorbate2)
        # Create the boundary conditions
        if bounds is not None:
            self.setup_bounds(bounds)
        if opt_kwargs is not None:
            self.opt_kwargs = opt_kwargs.copy()
        if parallel_run is not None:
            self.parallel_run = parallel_run
            self.check_parallel()
        return self

    def rotation_matrix(self, angles, positions):
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
        positions = np.matmul(positions, R)
        return positions

    def evaluate_value(self, x, **kwargs):
        "Evaluate the value of the adsorption."
        # Get the positions
        pos = self.positions0.copy()
        # Calculate the positions of the adsorbate
        pos_ads = pos[self.n_slab : self.n_slab + self.n_ads]
        pos_ads = self.rotation_matrix(x[3:6], pos_ads)
        pos_ads += np.sum(self.cell * x[:3].reshape(-1, 1), axis=0)
        pos[self.n_slab : self.n_slab + self.n_ads] = pos_ads
        # Calculate the positions of the second adsorbate
        if self.n_ads2 > 0:
            pos_ads2 = pos[self.n_slab + self.n_ads :]
            pos_ads2 = self.rotation_matrix(x[9:12], pos_ads2)
            pos_ads2 += np.sum(self.cell * x[6:9].reshape(-1, 1), axis=0)
            pos[self.n_slab + self.n_ads :] = pos_ads2
        # Set the positions
        self.optimizable.set_positions(pos)
        # Get the potential energy
        return self.optimizable.get_potential_energy()

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            slab=self.slab,
            adsorbate=self.adsorbate,
            adsorbate2=self.adsorbate2,
            bounds=self.bounds,
            opt_kwargs=self.opt_kwargs,
            parallel_run=self.parallel_run,
            comm=self.comm,
            verbose=self.verbose,
        )
        # Get the constants made within the class
        constant_kwargs = dict(steps=self.steps, _converged=self._converged)
        # Get the objects made within the class
        object_kwargs = dict()
        return arg_kwargs, constant_kwargs, object_kwargs
