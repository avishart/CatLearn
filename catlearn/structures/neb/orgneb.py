from numpy import (
    arange,
    argmax,
    array,
    asarray,
    einsum,
    empty,
    full,
    nanmax,
    ones,
    sqrt,
    vdot,
    zeros,
)
from ase.calculators.singlepoint import SinglePointCalculator
from ase.build import minimize_rotation_and_translation
from ase.parallel import world, broadcast
import warnings
from ..structure import Structure
from ...regression.gp.fingerprint.geometry import mic_distance
from ...regression.gp.calculator.copy_atoms import compare_atoms


class OriginalNEB:
    """
    The orginal Nudged Elastic Band method implementation for the tangent
    and parallel force.

    See:
        https://doi.org/10.1142/9789812839664_0016
    """

    def __init__(
        self,
        images,
        k=0.1,
        climb=False,
        remove_rotation_and_translation=False,
        mic=True,
        use_image_permutation=False,
        save_properties=False,
        parallel=False,
        comm=world,
        **kwargs,
    ):
        """
        Initialize the NEB instance.

        Parameters:
            images: List of ASE Atoms instances
                The ASE Atoms instances used as the images of the initial path
                that is optimized.
            k: List of floats or float
                The (Nimg-1) spring forces acting between each image.
            climb: bool
                Whether to use climbing image in the NEB.
                See:
                    https://doi.org/10.1063/1.1329672
            remove_rotation_and_translation: bool
                Whether to remove rotation and translation in interpolation
                and when predicting forces.
            mic: bool
                Minimum Image Convention (Shortest distances when
                periodic boundary conditions are used).
            use_image_permutation: bool
                Whether to permute images to minimize the path length.
                It assumes a greedy algorithm to find the minimum path length
                by selecting the next image that is closest to the previous
                image.
                It is only used in the initialization of the NEB.
            save_properties: bool
                Whether to save the properties by making a copy of the images.
            parallel: bool
                Whether to run the calculations in parallel.
            comm: ASE communicator instance
                The communicator instance for parallelization.

        """
        # Check that the endpoints are the same
        self.check_images(images)
        # Set images
        if save_properties:
            self.images = [Structure(image) for image in images]
        else:
            self.images = images
        self.nimages = len(images)
        self.natoms = len(images[0])
        # Set the spring constant
        if isinstance(k, (int, float)):
            self.k = full(self.nimages - 1, k)
        else:
            self.k = k.copy()
        # Set the parameters
        self.climb = climb
        self.rm_rot_trans = remove_rotation_and_translation
        self.mic = mic
        self.save_properties = save_properties
        self.use_image_permutation = use_image_permutation
        # Set the parallelization
        self.parallel = parallel
        if parallel:
            self.parallel_setup(comm)
            if (self.nimages - 2) % self.size != 0:
                if self.rank == 0:
                    warnings.warn(
                        "The number of moving images are not chosen "
                        "optimal for the number of processors when running in "
                        "parallel!"
                    )
        else:
            self.remove_parallel_setup()
        # Find the minimum path length if requested
        self.permute_images()
        # Set the properties
        self.reset()

    def check_images(
        self,
        images,
        properties_to_check=["atoms", "cell", "pbc"],
    ):
        "Check that the images are the same structures."
        ends_equal = compare_atoms(
            images[0],
            images[-1],
            properties_to_check=properties_to_check,
        )
        ends_move_equal = compare_atoms(
            images[0],
            images[1],
            properties_to_check=properties_to_check,
        )
        if not (ends_equal and ends_move_equal):
            raise ValueError("The images are not the same structures.")
        return self

    def interpolate(self, method="linear", mic=True, **kwargs):
        """
        Make an interpolation between the start and end structure.

        Parameters:
            method: str
                The method used for performing the interpolation.
                The optional methods is {linear, idpp, ends}.
            mic: bool
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
            **kwargs,
        )
        return self

    def get_positions(self):
        """
        Get the positions of all the moving images in one array.

        Returns:
            ((Nimg-2)*Natoms,3) array: Coordinates of all atoms in
                all the moving images.
        """
        positions = array(
            [image.get_positions() for image in self.images[1:-1]]
        )
        return positions.reshape(-1, 3)

    def set_positions(self, positions, **kwargs):
        """
        Set the positions of all the images in one array.

        Parameters:
            positions: ((Nimg-2)*Natoms,3) array
                Coordinates of all atoms in all the moving images.
        """
        self.reset()
        for i, image in enumerate(self.images[1:-1]):
            posi = i * self.natoms
            posip = (i + 1) * self.natoms
            image.set_positions(positions[posi:posip])
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
        return asarray([image.get_positions() for image in self.images])

    def get_climb_forces(self, forces_new, forces, tangent, **kwargs):
        "Get the forces of the climbing image."
        i_max = argmax(self.get_energies()[1:-1])
        forces_parallel = 2.0 * vdot(forces[i_max], tangent[i_max])
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
        self.real_forces = zeros((self.nimages, self.natoms, 3))
        self.energies = zeros((self.nimages))
        # Get the energy of the fixed images
        self.energies[0] = self.images[0].get_potential_energy()
        self.energies[-1] = self.images[-1].get_potential_energy()
        # Check if the calculation is done in parallel
        if self.parallel:
            return self.calculate_properties_parallel(**kwargs)
        # Calculate the energy and forces for each image
        for i, image in enumerate(self.images[1:-1]):
            self.real_forces[i + 1] = image.get_forces()
            self.energies[i + 1] = image.get_potential_energy()
        return self.energies, self.real_forces

    def calculate_properties_parallel(self, **kwargs):
        "Calculate the energy and forces for each image in parallel."
        # Calculate the energy and forces for each image
        for i, image in enumerate(self.images[1:-1]):
            if self.rank == (i % self.size):
                self.real_forces[i + 1] = image.get_forces()
                self.energies[i + 1] = image.get_potential_energy()
        # Broadcast the results
        for i in range(1, self.nimages - 1):
            root = (i - 1) % self.size
            self.energies[i], self.real_forces[i] = broadcast(
                (self.energies[i], self.real_forces[i]),
                root=root,
                comm=self.comm,
            )
        return self.energies, self.real_forces

    def emax(self, **kwargs):
        "Get maximum energy of the moving images."
        return nanmax(self.get_energies(**kwargs)[1:-1])

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
        pbc = self.get_pbc()
        if self.mic and pbc.any():
            cell = self.get_cell()
            _, position_diff = mic_distance(
                position_diff,
                cell=cell,
                pbc=pbc,
                use_vector=True,
            )
        return position_diff[1:], position_diff[:-1]

    def get_tangent(self, pos_p, pos_m, **kwargs):
        "Calculate the tangent to the moving images."
        # Normalization factors
        pos_m_norm = sqrt(einsum("ijk,ijk->i", pos_m, pos_m)).reshape(-1, 1, 1)
        pos_p_norm = sqrt(einsum("ijk,ijk->i", pos_p, pos_p)).reshape(-1, 1, 1)
        # Normalization of tangent
        tangent_m = pos_m / pos_m_norm
        tangent_p = pos_p / pos_p_norm
        # Sum them
        tangent = tangent_m + tangent_p
        # Normalization of tangent
        tangent_norm = sqrt(einsum("ijk,ijk->i", tangent, tangent)).reshape(
            -1, 1, 1
        )
        tangent = tangent / tangent_norm
        return tangent

    def get_spring_constants(self, **kwargs):
        "Get the spring constants for the images."
        return self.k

    def get_path_length(self, **kwargs):
        "Get the path length of the NEB."
        # Get the distances between the images
        pos_p, pos_m = self.get_position_diff()
        # Calculate the path length
        path_len = sqrt(einsum("ijk,ijk->i", pos_p, pos_p)).sum()
        path_len += sqrt(einsum("ij,ij->", pos_m[0], pos_m[0]))
        return path_len

    def permute_images(self, **kwargs):
        """
        Set the minimum path length by minimizing the distance between
        the images by permuting the images.
        """
        # Check if there are enough images to optimize
        if self.nimages <= 3 or not self.use_image_permutation:
            return self
        # Find the minimum path length
        selected_indices = self.find_minimum_path_length(**kwargs)
        # Set the images to the selected indices
        self.images = [self.images[i] for i in selected_indices]
        # Reset energies and forces
        self.reset()
        return self

    def find_minimum_path_length(self, **kwargs):
        """
        Find the minimum path length by minimizing the distance between
        the images.
        """
        # Get the positions of the images
        positions = self.get_image_positions()
        # Get the periodic boundary conditions
        pbc = self.get_pbc()
        cell = self.get_cell()
        use_mic = self.mic and pbc.any()
        if not use_mic:
            positions = positions.reshape(self.nimages, -1)
        # Set the indices for the selected images
        indices = arange(self.nimages, dtype=int)
        selected_indices = empty(self.nimages, dtype=int)
        selected_indices[0] = 0
        selected_indices[-1] = self.nimages - 1
        i_f = 1
        i_b = self.nimages - 2
        i_min_f = 0
        i_min_b = self.nimages - 1
        is_forward = True
        i_min = i_min_f
        # Create a boolean array to keep track of available images
        available = ones(self.nimages, dtype=bool)
        available[0] = available[-1] = False
        # Loop until all images are selected
        while available.any():
            candidates = indices[available]
            # Calculate the distance vectors from the current images
            dist = positions[candidates] - positions[i_min, None]
            if use_mic:
                dist = [
                    mic_distance(
                        dis,
                        cell=cell,
                        pbc=pbc,
                        use_vector=False,
                    )[0]
                    for dis in dist
                ]
                dist = asarray(dist)
            # Calculate the distances
            dist = sqrt(einsum("ij,ij->i", dist, dist))
            # Find the minimum distance from the current images
            i_min = dist.argmin()
            if is_forward:
                # Find the minimum distance from the start image
                i_min_f = candidates[i_min]
                selected_indices[i_f] = i_min_f
                available[i_min_f] = False
                i_f += 1
                i_min = i_min_b
            else:
                # Find the minimum distance from the end image
                i_min_b = candidates[i_min]
                selected_indices[i_b] = i_min_b
                available[i_min_b] = False
                i_b -= 1
                i_min = i_min_f
            # Switch the direction for the next iteration
            is_forward = not is_forward
        return selected_indices

    def reset(self):
        "Reset the stored properties."
        self.energies = None
        self.real_forces = None
        return self

    def parallel_setup(self, comm, **kwargs):
        "Setup the parallelization."
        if comm is None:
            self.comm = world
        else:
            self.comm = comm
        self.rank = self.comm.rank
        self.size = self.comm.size
        return self

    def remove_parallel_setup(self):
        "Remove the parallelization by removing the communicator."
        self.comm = None
        self.rank = 0
        self.size = 1
        return self

    def get_residual(self, **kwargs):
        "Get the residual of the NEB."
        forces = self.get_forces()
        return sqrt(einsum("ij,ij->i", forces, forces)).max()

    def set_calculator(self, calculators, copy_calc=False, **kwargs):
        """
        Set the calculators for all the images.

        Parameters:
            calculators: List of ASE Calculators or ASE Calculator
                The calculator used for all the images if a list is given.
                If a single calculator is given, it is used for all images.
        """
        self.reset()
        if isinstance(calculators, (list, tuple)):
            if len(calculators) != self.nimages - 2:
                raise ValueError(
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
        return sqrt(einsum("ij,ij->i", forces, forces)).max() < fmax

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

    def get_pbc(self):
        """
        Get the periodic boundary conditions of the images.

        Returns:
            (3,) array: The periodic boundary conditions of the images.
        """
        return asarray(self.images[0].get_pbc())

    def get_cell(self):
        """
        Get the cell of the images.

        Returns:
            (3,3) array: The cell of the images.
        """
        return asarray(self.images[0].get_cell())

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            images=self.images,
            k=self.k,
            climb=self.climb,
            remove_rotation_and_translation=self.rm_rot_trans,
            mic=self.mic,
            use_image_permutation=self.use_image_permutation,
            save_properties=self.save_properties,
            parallel=self.parallel,
            comm=self.comm,
        )
        # Get the constants made within the class
        constant_kwargs = dict()
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
