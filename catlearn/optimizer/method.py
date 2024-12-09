import numpy as np
from ase.parallel import world, broadcast
from ..regression.gp.calculator.copy_atoms import copy_atoms
from ..structures.structure import Structure


class OptimizerMethod:
    def __init__(
        self,
        optimizable,
        parallel_run=False,
        comm=world,
        verbose=False,
        **kwargs,
    ):
        """
        The OptimizerMethod class is a base class for all optimization methods.
        The OptimizerMethod is used to run an optimization on a given
        optimizable.
        The OptimizerMethod is applicable to be used with active learning.

        Parameters:
            optimizable: optimizable instance
                The instance to be optimized.
                Often, an Atoms or NEB instance.
                Here, it assumed to be an Atoms instance.
            parallel_run: bool
                If True, the optimization will be run in parallel.
            comm: ASE communicator instance
                The communicator object for parallelization.
            verbose: bool
                Whether to print the full output (True) or
                not (False).
        """
        # Set the parameters
        self.update_arguments(
            optimizable=optimizable,
            parallel_run=parallel_run,
            comm=comm,
            verbose=verbose,
            **kwargs,
        )

    def update_optimizable(self, structures, **kwargs):
        """
        Update the optimizable instance by given
        dependent structures.

        Parameters:
            structures: Atoms instance or list of Atoms instances
                The structures that the optimizable instance is dependent on.
        """
        # Check if the structures are a list
        if isinstance(structures, list):
            raise NotImplementedError(
                "The method does not support multiple structures"
            )
        # Update optimizable by setting the positions of the optimizable
        self.optimizable.set_positions(structures.get_positions())
        # Reset the optimization
        self.reset_optimization()
        return self

    def get_optimizable(self, **kwargs):
        """
        Get the optimizable that are considered for the optimizer.

        Returns:
            optimizable: The optimizable instance.
            Often, an Atoms or NEB instance.
        """
        return self.optimizable

    def get_structures(self, get_all=True, **kwargs):
        """
        Get the structures that optimizable instance is dependent on.

        Parameters:
            get_all: bool
                If True, all structures are returned.
                Else, only the first structure is returned

        Returns:
            structures: Atoms instance or list of Atoms instances
                The structures that the optimizable instance is dependent on.
        """
        return self.copy_atoms(self.optimizable)

    def get_candidates(self, **kwargs):
        """
        Get the candidate structure instances.
        It is used for active learning.
        """
        return [self.optimizable]

    def reset_optimization(self):
        """
        Reset the optimization.
        """
        self.steps = 0
        self._converged = False
        return self

    def setup_optimizable(self, optimizable):
        """
        Set the optimizable instance.

        Parameters:
            optimizable: optimizable instance
                The instance to be optimized.
                Often, an Atoms or NEB instance.
        """
        self.optimizable = optimizable
        self.reset_optimization()
        return self

    def set_calculator(self, calculator, copy_calc=False, **kwargs):
        """
        Set the calculator for the optimizable instance.

        Parameters:
            calculator: ASE calculator instance
                The calculator to be set.
            copy_calc: bool
                If True, the calculator will be copied.
        """
        if copy_calc:
            self.optimizable.calc = calculator.copy()
        else:
            self.optimizable.calc = calculator
        return self

    def get_calculator(self):
        """
        Get the calculator of the optimizable instance.
        """
        return self.optimizable.calc

    @property
    def calc(self):
        """
        The calculator instance.
        """
        return self.get_calculator()

    @calc.setter
    def calc(self, calculators):
        return self.set_calculator(calculators)

    def get_potential_energy(self, per_candidate=False, **kwargs):
        """
        Get the potential energy of the optimizable.

        Parameters:
            per_candidate: bool
                If True, the potential energy of each candidate is returned.
                Else, the potential energy of the optimizable is returned.

        Returns:
            energy: float or list
                The potential energy of the optimizable.
        """
        if per_candidate:
            if self.is_parallel_used():
                return self.get_potential_energy_parallel(**kwargs)
            energy = [
                atoms.get_potential_energy(**kwargs)
                for atoms in self.get_candidates()
            ]
        else:
            energy = self.optimizable.get_potential_energy(**kwargs)
        return energy

    def get_potential_energy_parallel(self, **kwargs):
        """
        Get the potential energies of the candidates in parallel.

        Returns:
            energy: list of floats
                The potential energies of the candidates.
        """
        energy = []
        for i, atoms in enumerate(self.get_candidates()):
            root = i % self.size
            e = None
            if self.rank == root:
                e = atoms.get_potential_energy(**kwargs)
            e = broadcast(e, root=root, comm=self.comm)
            energy.append(e)
        return energy

    def get_forces(self, per_candidate=False, **kwargs):
        """
        Get the forces of the optimizable.

        Parameters:
            per_candidate: bool
                If True, the forces of each candidate is returned.
                Else, the forces of the optimizable is returned

        Returns:
            force: (N,3) array or list of (N,3) arrays
                The forces of the optimizable.
        """
        if per_candidate:
            if self.is_parallel_used():
                return self.get_forces_parallel(**kwargs)
            forces = [
                atoms.get_forces(**kwargs) for atoms in self.get_candidates()
            ]
        else:
            forces = self.optimizable.get_forces(**kwargs)
        return forces

    def get_forces_parallel(self, **kwargs):
        """
        Get the forces of the candidates in parallel.

        Returns:
            forces: list of (N,3) arrays
                The forces of the candidates.
        """
        forces = []
        for i, atoms in enumerate(self.get_candidates()):
            root = i % self.size
            f = None
            if self.rank == root:
                f = atoms.get_forces(**kwargs)
            f = broadcast(f, root=root, comm=self.comm)
            forces.append(f)
        return forces

    def get_fmax(self, per_candidate=False, **kwargs):
        """
        Get the maximum force of an atom in the optimizable.

        Parameters:
            per_candidate: bool
                If True, the maximum force of each candidate is returned.
                Else, the maximum force of the optimizable is returned.

        Returns:
            fmax: float or list
                The maximum force of the optimizable.
        """
        force = self.get_forces(per_candidate=per_candidate, **kwargs)
        fmax = np.linalg.norm(force, axis=-1).max(axis=-1)
        return fmax

    def get_uncertainty(self, per_candidate=False, **kwargs):
        """
        Get the uncertainty of the optimizable.
        It is used for active learning.

        Parameters:
            per_candidate: bool
                If True, the uncertainty of each candidate is returned.
                Else, the maximum uncertainty of the optimizable is returned.

        Returns:
            uncertainty: float or list
                The uncertainty of the optimizable.
        """
        if self.is_parallel_used():
            uncertainty = self.get_uncertainty_parallel(**kwargs)
        else:
            uncertainty = [
                (
                    atoms.get_uncertainty(**kwargs)
                    if isinstance(atoms, Structure)
                    else atoms.calc.get_property(
                        "uncertainty",
                        atoms=atoms,
                        **kwargs,
                    )
                )
                for atoms in self.get_candidates()
            ]
        if not per_candidate:
            uncertainty = np.max(uncertainty)
        return uncertainty

    def get_uncertainty_parallel(self, **kwargs):
        """
        Get the uncertainty of the candidates in parallel.
        It is used for active learning.

        Returns:
            uncertainty: list of floats
                The uncertainty of the candidates.
        """
        uncertainty = []
        for i, atoms in enumerate(self.get_candidates()):
            root = i % self.size
            unc = None
            if self.rank == root:
                if isinstance(atoms, Structure):
                    unc = atoms.get_uncertainty(**kwargs)
                else:
                    unc = atoms.calc.get_property(
                        "uncertainty",
                        atoms=atoms,
                        **kwargs,
                    )
            unc = broadcast(unc, root=root, comm=self.comm)
            uncertainty.append(unc)
        return uncertainty

    def get_property(
        self,
        name,
        allow_calculation=True,
        per_candidate=False,
        **kwargs,
    ):
        """
        Get or calculate the requested property.

        Parameters:
            name: str
                The name of the requested property.
            allow_calculation: bool
                Whether the property is allowed to be calculated.
            per_candidate: bool
                If True, the property of each candidate is returned.
                Else, the property of the optimizable is returned.

        Returns:
            float or list: The requested property.
        """
        if per_candidate:
            if self.is_parallel_used():
                return self.get_property_parallel(
                    name,
                    allow_calculation,
                    **kwargs,
                )
            output = []
            for atoms in self.get_candidates():
                if name == "energy":
                    result = atoms.get_potential_energy(**kwargs)
                elif name == "forces":
                    result = atoms.get_forces(**kwargs)
                elif name == "fmax":
                    force = atoms.get_forces(**kwargs)
                    result = np.linalg.norm(force, axis=-1).max()
                elif name == "uncertainty" and isinstance(atoms, Structure):
                    result = atoms.get_uncertainty(**kwargs)
                else:
                    result = atoms.calc.get_property(
                        name,
                        atoms=atoms,
                        allow_calculation=allow_calculation,
                        **kwargs,
                    )
                output.append(result)
        else:
            if name == "energy":
                output = self.get_potential_energy(
                    per_candidate=per_candidate,
                    **kwargs,
                )
            elif name == "forces":
                output = self.get_forces(per_candidate=per_candidate, **kwargs)
            elif name == "fmax":
                output = self.get_fmax(per_candidate=per_candidate, **kwargs)
            elif name == "uncertainty":
                output = self.get_uncertainty(
                    per_candidate=per_candidate,
                    **kwargs,
                )
            else:
                output = self.optimizable.calc.get_property(
                    name,
                    atoms=self.optimizable,
                    allow_calculation=allow_calculation,
                    **kwargs,
                )
        return output

    def get_property_parallel(
        self,
        name,
        allow_calculation=True,
        **kwargs,
    ):
        """
        Get the requested property of the candidates in parallel.

        Parameters:
            name: str
                The name of the requested property.
            allow_calculation: bool
                Whether the property is allowed to be calculated.

        Returns:
            list: The list of requested property.
        """
        output = []
        for i, atoms in enumerate(self.get_candidates()):
            root = i % self.size
            result = None
            if self.rank == root:
                if name == "energy":
                    result = atoms.get_potential_energy(**kwargs)
                elif name == "forces":
                    result = atoms.get_forces(**kwargs)
                elif name == "fmax":
                    force = atoms.get_forces(**kwargs)
                    result = np.linalg.norm(force, axis=-1).max()
                elif name == "uncertainty" and isinstance(atoms, Structure):
                    result = atoms.get_uncertainty(**kwargs)
                else:
                    result = atoms.calc.get_property(
                        name,
                        atoms=atoms,
                        allow_calculation=allow_calculation,
                        **kwargs,
                    )
            result = broadcast(result, root=root, comm=self.comm)
            output.append(result)
        return output

    def get_properties(
        self,
        properties,
        allow_calculation=True,
        per_candidate=False,
        **kwargs,
    ):
        """
        Get or calculate the requested properties.

        Parameters:
            properties: list of str
                The names of the requested properties.
            allow_calculation: bool
                Whether the properties are allowed to be calculated.
            per_candidate: bool
                If True, the properties of each candidate are returned.
                Else, the properties of the optimizable are returned.

        Returns:
            dict: The requested properties.
        """
        if per_candidate:
            if self.is_parallel_used():
                return self.get_properties_parallel(
                    properties,
                    allow_calculation,
                    **kwargs,
                )
            results = {name: [] for name in properties}
            for atoms in self.get_candidates():
                for name in properties:
                    if name == "energy":
                        output = atoms.get_potential_energy(**kwargs)
                    elif name == "forces":
                        output = atoms.get_forces(**kwargs)
                    elif name == "fmax":
                        force = atoms.get_forces(**kwargs)
                        output = np.linalg.norm(force, axis=-1).max()
                    elif name == "uncertainty" and isinstance(
                        atoms, Structure
                    ):
                        output = atoms.get_uncertainty(**kwargs)
                    else:
                        output = atoms.calc.get_property(
                            name,
                            atoms=atoms,
                            allow_calculation=allow_calculation,
                            **kwargs,
                        )
                    results[name].append(output)
        else:
            results = {}
            for name in properties:
                results[name] = self.get_property(
                    name=name,
                    allow_calculation=allow_calculation,
                    per_candidate=per_candidate,
                    **kwargs,
                )
        return results

    def get_properties_parallel(
        self,
        properties,
        allow_calculation=True,
        **kwargs,
    ):
        """
        Get the requested properties of the candidates in parallel.

        Parameters:
            properties: list of str
                The names of the requested properties.
            allow_calculation: bool
                Whether the properties are allowed to be calculated.

        Returns:
            dict: The requested properties.
        """
        results = {name: [] for name in properties}
        for i, atoms in enumerate(self.get_candidates()):
            root = i % self.size
            for name in properties:
                output = None
                if self.rank == root:
                    if name == "energy":
                        output = atoms.get_potential_energy(**kwargs)
                    elif name == "forces":
                        output = atoms.get_forces(**kwargs)
                    elif name == "fmax":
                        force = atoms.get_forces(**kwargs)
                        output = np.linalg.norm(force, axis=-1).max()
                    elif name == "uncertainty" and isinstance(
                        atoms, Structure
                    ):
                        output = atoms.get_uncertainty(**kwargs)
                    else:
                        output = atoms.calc.get_property(
                            name,
                            atoms=atoms,
                            allow_calculation=allow_calculation,
                            **kwargs,
                        )
                output = broadcast(output, root=root, comm=self.comm)
                results[name].append(output)
        return results

    def is_within_dtrust(self, per_candidate=False, dtrust=2.0, **kwargs):
        """
        Get whether the structures are within a trust distance to the database.
        It is used for active learning.

        Parameters:
            per_candidate: bool
                If True, the distance of each candidate is returned.
                Else, the maximum distance of the optimizable is returned.
            dtrust: float
                The distance trust criterion.

        Returns:
            within_dtrust: float or list
                Whether the structures are within a trust distance to
                the database.
        """
        within_dtrust = []
        for atoms in self.get_candidates():
            if isinstance(atoms, Structure):
                real_atoms = atoms.get_structure()
                within = real_atoms.calc.is_in_database(
                    real_atoms,
                    dtol=dtrust,
                    **kwargs,
                )
            else:
                within = atoms.calc.is_in_database(
                    atoms,
                    dtol=dtrust,
                    **kwargs,
                )
            within_dtrust.append(within)
        if not per_candidate:
            if False in within_dtrust:
                within_dtrust = False
            else:
                within_dtrust = True
        return within_dtrust

    def get_number_of_steps(self):
        """
        Get the number of steps that have been run.
        """
        return self.steps

    def converged(self):
        """
        Check if the optimization is converged.
        """
        return self._converged

    def is_energy_minimized(self):
        """
        Check if the optimization method minimizes the energy.
        """
        return True

    def is_parallel_allowed(self):
        """
        Check if the optimization method allows parallelization.
        """
        return False

    def is_parallel_used(self):
        """
        Check if the optimization method uses parallelization.
        """
        return self.parallel_run and self.is_parallel_allowed()

    def run(
        self,
        fmax=0.05,
        steps=1000,
        max_unc=None,
        dtrust=None,
        unc_convergence=None,
        **kwargs,
    ):
        """
        Run the optimization.

        Parameters:
            fmax: float
                The maximum force allowed on an atom.
            steps: int
                The maximum number of steps allowed.
            max_unc: float (optional)
                Maximum uncertainty for continuation of the optimization.
            dtrust: float (optional)
                The distance trust criterion.
            unc_convergence: float (optional)
                The uncertainty convergence criterion for convergence.

        Returns:
            coverged: bool
                Whether the optimization is converged.
        """
        raise NotImplementedError("The run method is not implemented")

    def run_max_unc(self, **kwargs):
        """
        Run the optimization with a maximum uncertainty.
        The uncertainty is checked at each optimization step if requested.
        The trust distance is checked at each optimization step if requested.
        It is used for active learning.
        """
        raise NotImplementedError("The run_max_unc method is not implemented")

    def check_convergence(
        self,
        converged,
        max_unc=None,
        dtrust=None,
        unc_convergence=None,
        **kwargs,
    ):
        """
        Check if the optimization is converged also in terms of uncertainty.
        The uncertainty is used for active learning.

        Parameters:
            converged: bool
                Whether the optimization is converged.
            max_unc: float (optional)
                The maximum uncertainty allowed.
            dtrust: float (optional)
                The distance trust criterion.
            unc_convergence: float (optional)
                The uncertainty convergence criterion for convergence.

        Returns:
            converged: bool
                Whether the optimization is converged.
        """
        # Check if the optimization is converged at all
        if not converged:
            return False
        # Check if the optimization is converged in terms of uncertainty
        if max_unc is not None or unc_convergence is not None:
            unc = self.get_uncertainty()
            if max_unc is not None and unc > max_unc:
                return False
            if unc_convergence is not None and unc > unc_convergence:
                return False
        # Check if the optimization is converged in terms of database distance
        if dtrust is not None:
            within_dtrust = self.is_within_dtrust(dtrust=dtrust)
            if not within_dtrust:
                return False
        return converged

    def update_arguments(
        self,
        optimizable=None,
        parallel_run=None,
        comm=None,
        verbose=None,
        **kwargs,
    ):
        """
        Update the instance with its arguments.
        The existing arguments are used if they are not given.

        Parameters:
            optimizable: optimizable instance
                The instance to be optimized.
                Often, an Atoms or NEB instance.
                Here, it assumed to be an Atoms instance.
            parallel_run: bool
                If True, the optimization will be run in parallel.
            comm: ASE communicator instance
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
        if optimizable is not None:
            self.setup_optimizable(optimizable)
        if parallel_run is not None:
            self.parallel_run = parallel_run
            self.check_parallel()
        return self

    def copy_atoms(self, atoms):
        "Copy an atoms instance."
        # Enforce the correct results in the calculator
        if atoms.calc is not None:
            if hasattr(atoms.calc, "results"):
                if len(atoms.calc.results):
                    atoms.get_forces()
        # Save the structure with saved properties
        return copy_atoms(atoms)

    def message(self, message):
        "Print a message."
        if self.verbose and self.rank == 0:
            print(message)
        return self

    def check_parallel(self):
        "Check if the parallelization is allowed."
        if self.parallel_run and not self.is_parallel_allowed():
            self.message("Parallel run is not supported for this method!")
        return self

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            optimizable=self.optimizable,
            parallel_run=self.parallel_run,
            comm=self.comm,
            verbose=self.verbose,
        )
        # Get the constants made within the class
        constant_kwargs = dict(steps=self.steps, _converged=self._converged)
        # Get the objects made within the class
        object_kwargs = dict()
        return arg_kwargs, constant_kwargs, object_kwargs

    def copy(self):
        "Copy the object."
        # Get all arguments
        arg_kwargs, constant_kwargs, object_kwargs = self.get_arguments()
        # Make a clone
        clone = self.__class__(**arg_kwargs)
        # Check if constants have to be saved
        if len(constant_kwargs.keys()):
            for key, value in constant_kwargs.items():
                clone.__dict__[key] = value
        # Check if objects have to be saved
        if len(object_kwargs.keys()):
            for key, value in object_kwargs.items():
                clone.__dict__[key] = value.copy()
        return clone

    def __repr__(self):
        arg_kwargs = self.get_arguments()[0]
        str_kwargs = ",".join(
            [f"{key}={value}" for key, value in arg_kwargs.items()]
        )
        return "{}({})".format(self.__class__.__name__, str_kwargs)
