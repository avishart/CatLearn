import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.build import minimize_rotation_and_translation
from ..structure import Structure
from ...regression.gp.fingerprint.geometry import mic_distance


class OriginalNEB:
    def __init__(
        self,
        images,
        k=0.1,
        climb=False,
        remove_rotation_and_translation=False,
        mic=True,
        save_properties=False,
        parallel=False,
        world=None,
        **kwargs
    ):
        """
        The orginal Nudged Elastic Band method implementation for the tangent
        and parallel force.

        Parameters:
            images: List of ASE Atoms instances
                The ASE Atoms instances used as the images of the initial path
                that is optimized.
            k: List of floats or float
                The (Nimg-1) spring forces acting between each image.
            climb: bool
                Whether to use climbing image in the NEB.
            remove_rotation_and_translation: bool
                Whether to remove rotation and translation in interpolation
                and when predicting forces.
            mic: bool
                Minimum Image Convention (Shortest distances when
                periodic boundary conditions are used).
            save_properties: bool
                Whether to save the properties by making a copy of the images.
            parallel: bool
                Whether to run the calculations in parallel.
            world: ASE communicator instance
                The communicator instance for parallelization.

        """
        # Set images
        if save_properties:
            self.images = [Structure(image) for image in images]
        else:
            self.images = images
        self.nimages = len(images)
        self.natoms = len(images[0])
        # Set the spring constant
        if isinstance(k, (int, float)):
            self.k = np.full(self.nimages - 1, k)
        else:
            self.k = k.copy()
        # Set the parameters
        self.climb = climb
        self.rm_rot_trans = remove_rotation_and_translation
        self.mic = mic
        self.save_properties = save_properties
        # Set the parallelization
        self.parallel = parallel
        if parallel:
            if world is None:
                from ase.parallel import world, parprint

            self.world = world
            if self.nimages % self.world.size != 0:
                parprint(
                    "Warning: The number of images are not chosen optimal for "
                    "the number of processors when running in parallel!"
                )
        else:
            self.world = None
        # Set the properties
        self.reset()

    def interpolate(self, method="linear", mic=True, **kwargs):
        """
        Make an interpolation between the start and end structure.

        Parameters:
            method : str
                The method used for performing the interpolation.
                The optional methods is {linear, idpp, ends}.
            mic : bool
                Whether to use the minimum-image convention.

        Returns:
            self: The instance itself.
        """
        from .interpolate_band import interpolate

        self.images = interpolate(
            self.images[0],
            self.images[-1],
            n_images=self.nimages,
            method=method,
            mic=mic,
            remove_rotation_and_translation=self.rm_rot_trans,
            **kwargs
        )
        return self

    def get_positions(self):
        """
        Get the positions of all the moving images in one array.

        Returns:
            ((Nimg-2)*Natoms,3) array: Coordinates of all atoms in
                all the moving images.
        """
        positions = np.array(
            [image.get_positions() for image in self.images[1:-1]]
        )
        return positions.reshape(-1, 3)

    def set_positions(self, positions, **kwargs):
        """
        Set the positions of all the images in one array.

        Parameters:
            positions : ((Nimg-2)*Natoms,3) array
                Coordinates of all atoms in all the moving images.
        """
        self.reset()
        for i, image in enumerate(self.images[1:-1]):
            image.set_positions(
                positions[i * self.natoms : (i + 1) * self.natoms]
            )
        pass

    def get_potential_energy(self, **kwargs):
        """
        Get the potential energy of the NEB as the sum of energies.

        Returns:
            float: Sum of energies of moving images.
        """
        return (self.get_energies(**kwargs)[1:-1]).sum()

    def get_forces(self, **kwargs):
        """
        Get the forces of the NEB as the stacked forces of the moving images.

        Returns:
            ((Nimg-2)*Natoms,3) array: Forces of all the atoms in
                all the moving images.
        """
        # Remove rotation and translation
        if self.rm_rot_trans:
            for i in range(1, self.nimages):
                minimize_rotation_and_translation(
                    self.images[i - 1],
                    self.images[i],
                )
        # Get the forces for each image
        forces = self.calculate_forces(**kwargs)
        # Get change in the coordinates to the previous and later image
        position_plus, position_minus = self.get_position_diff()
        # Calculate the tangent to the moving images
        tangent = self.get_tangent(position_plus, position_minus)
        # Calculate the parallel forces between images
        parallel_forces = self.get_parallel_forces(
            tangent,
            position_plus,
            position_minus,
        )
        # Calculate the perpendicular forces
        perpendicular_forces = self.get_perpendicular_forces(tangent, forces)
        # Calculate the full force
        forces_new = parallel_forces + perpendicular_forces
        # Calculate the force of the climbing image
        if self.climb:
            forces_new = self.get_climb_forces(forces_new, forces, tangent)
        return forces_new.reshape(-1, 3)

    def get_image_positions(self):
        """
        Get the positions of the images.

        Returns:
            ((Nimg),Natoms,3) array: The positions for all atoms in
                all the images.
        """
        return np.array([image.get_positions() for image in self.images])

    def get_climb_forces(self, forces_new, forces, tangent, **kwargs):
        "Get the forces of the climbing image."
        i_max = np.argmax(self.get_energies()[1:-1])
        forces_parallel = 2.0 * np.vdot(forces[i_max], tangent[i_max])
        forces_parallel = forces_parallel * tangent[i_max]
        forces_new[i_max] = forces[i_max] - forces_parallel
        return forces_new

    def calculate_forces(self, **kwargs):
        "Calculate the forces for all the images separately."
        if self.real_forces is None:
            self.calculate_properties()
        return self.real_forces[1:-1].copy()

    def get_energies(self, **kwargs):
        "Get the individual energy for each image."
        if self.energies is None:
            self.calculate_properties()
        return self.energies

    def calculate_properties(self, **kwargs):
        "Calculate the energy and forces for each image."
        # Initialize the arrays
        self.real_forces = np.zeros((self.nimages, self.natoms, 3))
        self.energies = np.zeros((self.nimages))
        # Get the energy of the fixed images
        self.energies[0] = self.images[0].get_potential_energy()
        self.energies[-1] = self.images[-1].get_potential_energy()
        # Check if the calculation is done in parallel
        if self.parallel:
            return self.calculate_properties_parallel(**kwargs)
        # Calculate the energy and forces for each image
        for i, image in enumerate(self.images[1:-1]):
            self.real_forces[i + 1] = image.get_forces().copy()
            self.energies[i + 1] = image.get_potential_energy()
        return self.energies, self.real_forces

    def calculate_properties_parallel(self, **kwargs):
        "Calculate the energy and forces for each image in parallel."
        # Calculate the energy and forces for each image
        for i, image in enumerate(self.images[1:-1]):
            if self.world.rank == (i % self.world.size):
                self.real_forces[i + 1] = image.get_forces().copy()
                self.energies[i + 1] = image.get_potential_energy()
        # Broadcast the results
        for i in range(1, self.nimages - 1):
            root = (i - 1) % self.world.size
            self.world.broadcast(self.energies[i : i + 1], root=root)
            self.world.broadcast(self.real_forces[i : i + 1], root=root)
        return self.energies, self.real_forces

    def emax(self, **kwargs):
        "Get maximum energy of the moving images."
        return np.nanmax(self.get_energies(**kwargs)[1:-1])

    def get_parallel_forces(self, tangent, pos_p, pos_m, **kwargs):
        "Get the parallel forces between the images."
        # Get the spring constants
        k = self.get_spring_constants()
        k = k.reshape(-1, 1, 1)
        # Calculate the parallel forces
        forces_parallel = (k[1:] * pos_p) - (k[:-1] * pos_m)
        forces_parallel = (forces_parallel * tangent).sum(axis=(1, 2))
        forces_parallel = forces_parallel.reshape(-1, 1, 1) * tangent
        return forces_parallel

    def get_perpendicular_forces(self, tangent, forces, **kwargs):
        "Get the perpendicular forces to the images."
        f_parallel = (forces * tangent).sum(axis=(1, 2))
        f_parallel = f_parallel.reshape(-1, 1, 1) * tangent
        return forces - f_parallel

    def get_position_diff(self):
        """
        Get the change in the coordinates relative to
        the previous and later image.
        """
        positions = self.get_image_positions()
        position_diff = positions[1:] - positions[:-1]
        pbc = np.array(self.images[0].get_pbc())
        if self.mic and pbc.any():
            cell = np.array(self.images[0].get_cell())
            position_diff = mic_distance(
                position_diff,
                cell,
                pbc,
                vector=True,
            )[1]
        return position_diff[1:], position_diff[:-1]

    def get_tangent(self, pos_p, pos_m, **kwargs):
        "Calculate the tangent to the moving images."
        # Normalization factors
        pos_m_norm = np.linalg.norm(pos_m, axis=(1, 2)).reshape(-1, 1, 1)
        pos_p_norm = np.linalg.norm(pos_p, axis=(1, 2)).reshape(-1, 1, 1)
        # Normalization of tangent
        tangent_m = pos_m / pos_m_norm
        tangent_p = pos_p / pos_p_norm
        # Sum them
        tangent = tangent_m + tangent_p
        # Normalization of tangent
        tangent_norm = np.linalg.norm(tangent, axis=(1, 2)).reshape(-1, 1, 1)
        tangent = tangent / tangent_norm
        return tangent

    def get_spring_constants(self, **kwargs):
        "Get the spring constants for the images."
        return self.k

    def reset(self):
        "Reset the stored properties."
        self.energies = None
        self.real_forces = None
        return self

    def get_residual(self, **kwargs):
        "Get the residual of the NEB."
        forces = self.get_forces()
        return np.max(np.linalg.norm(forces, axis=-1))

    def set_calculator(self, calculators, copy_calc=False, **kwargs):
        """
        Set the calculators for all the images.

        Parameters:
            calculators : List of ASE Calculators or ASE Calculator
                The calculator used for all the images if a list is given.
                If a single calculator is given, it is used for all images.
        """
        self.reset()
        if isinstance(calculators, (list, tuple)):
            if len(calculators) != self.nimages - 2:
                raise Exception(
                    "The number of calculators must be "
                    "equal to the number of moving images."
                )
            for i, image in enumerate(self.images[1:-1]):
                if copy_calc:
                    image.calc = calculators[i].copy()
                else:
                    image.calc = calculators[i]
        else:
            for image in self.images[1:-1]:
                if copy_calc:
                    image.calc = calculators.copy()
                else:
                    image.calc = calculators
        return self

    @property
    def calc(self):
        """
        The calculator objects.
        """
        return [image.calc for image in self.images[1:-1]]

    @calc.setter
    def calc(self, calculators):
        return self.set_calculator(calculators)

    def converged(self, forces, fmax):
        return np.linalg.norm(forces, axis=1).max() < fmax

    def is_neb(self):
        return True

    def __ase_optimizable__(self):
        return self

    def __len__(self):
        return int(self.nimages - 2) * self.natoms

    def freeze_results_on_image(self, atoms, **results_to_include):
        atoms.calc = SinglePointCalculator(atoms=atoms, **results_to_include)
        return atoms

    def iterimages(self):
        # Allows trajectory to convert NEB into several images
        for i, atoms in enumerate(self.images):
            if i == 0 or i == self.nimages - 1:
                yield atoms
            else:
                atoms = atoms.copy()
                atoms = self.freeze_results_on_image(
                    atoms,
                    energy=self.energies[i],
                    forces=self.real_forces[i],
                )
                yield atoms
