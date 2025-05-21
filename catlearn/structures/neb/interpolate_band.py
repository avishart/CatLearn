from numpy import ndarray
from numpy.linalg import norm
from ase.io import read
from ase.optimize import FIRE
from ase.build import minimize_rotation_and_translation
from .improvedneb import ImprovedTangentNEB
from ...regression.gp.calculator.copy_atoms import copy_atoms
from ...regression.gp.fingerprint.geometry import mic_distance


def interpolate(
    start,
    end,
    ts=None,
    n_images=15,
    method="linear",
    mic=True,
    remove_rotation_and_translation=False,
    **interpolation_kwargs,
):
    """
    Make a NEB interpolation between the start and end structure.
    A transition state structure can be given to guide the NEB interpolation.

    Parameters:
        start: ASE Atoms instance
            The starting structure for the NEB interpolation.
        end: ASE Atoms instance
            The ending structure for the NEB interpolation.
        ts: ASE Atoms instance (optional)
            An intermediate state the NEB interpolation should go through.
            Then, the method should be one of the following: 'linear', 'idpp',
            'rep', 'born', or 'ends'.
        n_images: int
            The number of images in the NEB interpolation.
        method: str or list of ASE Atoms instances
            The method to use for the NEB interpolation. If a list of
            ASE Atoms instances is given, then the interpolation will be
            made between the start and end structure using the images in
            the list. If a string is given, then it should be one of the
            following: 'linear', 'idpp', 'rep', or 'ends'. The string can
            also be the name of a trajectory file. In that case, the
            interpolation will be made using the images in the trajectory
            file. The trajectory file should contain the start and end
            structure.
        mic: bool
            If True, then the minimum-image convention is used for the
            interpolation. If False, then the images are not constrained
            to the minimum-image convention.
        remove_rotation_and_translation: bool
            If True, then the rotation and translation of the end
            structure is removed before the interpolation is made.
        interpolation_kwargs: dict
            Additional keyword arguments to pass to the interpolation
            methods.
    """
    # Copy the start and end structures
    start = copy_atoms(start)
    end = copy_atoms(end)
    # The rotation and translation should be removed the end structure
    # is optimized compared to start structure
    if remove_rotation_and_translation:
        start.center()
        end.center()
        minimize_rotation_and_translation(start, end)
    # If the transition state is not given then make a regular interpolation
    if ts is None:
        images = make_interpolation(
            start,
            end,
            n_images=n_images,
            method=method,
            mic=mic,
            remove_rotation_and_translation=remove_rotation_and_translation,
            **interpolation_kwargs,
        )
        return images
    # Copy the transition state structure
    ts = copy_atoms(ts)
    # Check if the method is compatible with the interpolation for the TS
    if not (
        isinstance(method, str)
        and method in ["linear", "idpp", "rep", "born", "ends"]
    ):
        raise ValueError(
            "The method should be one of the following: "
            "'linear', 'idpp', 'rep', 'born', or 'ends."
        )
    # Get the interpolated path from the start structure to the TS structure
    images = make_interpolation(
        start,
        ts,
        n_images=n_images,
        method=method,
        mic=mic,
        remove_rotation_and_translation=remove_rotation_and_translation,
        **interpolation_kwargs,
    )
    # Get the cumulative distance from the start to the TS structure
    dis_st = get_images_distance(images)
    # Get the interpolated path from the TS structure to the end structure
    images = make_interpolation(
        ts,
        end,
        n_images=n_images,
        method=method,
        mic=mic,
        remove_rotation_and_translation=remove_rotation_and_translation,
        **interpolation_kwargs,
    )
    # Get the cumulative distance from the TS to the end structure
    dis_et = get_images_distance(images)
    # Calculate the number of images from start to the TS from the distance
    n_images_st = int(n_images * dis_st / (dis_st + dis_et))
    n_images_st = 2 if n_images_st < 2 else n_images_st
    # Get the interpolated path from the start structure to
    # the TS structure with the correct number of images
    images1 = make_interpolation(
        start,
        ts,
        n_images=n_images_st,
        method=method,
        mic=mic,
        remove_rotation_and_translation=remove_rotation_and_translation,
        **interpolation_kwargs,
    )
    # Get the interpolated path from the TS structure to
    # the end structure with the corrct number of images
    images2 = make_interpolation(
        ts,
        end,
        n_images=int(n_images - n_images_st + 1),
        method=method,
        mic=mic,
        remove_rotation_and_translation=remove_rotation_and_translation,
        **interpolation_kwargs,
    )[1:]
    return list(images1) + list(images2)


def make_interpolation(
    start,
    end,
    n_images=15,
    method="linear",
    mic=True,
    **interpolation_kwargs,
):
    "Make the NEB interpolation path."
    # Use a premade interpolation path
    if isinstance(method, (list, ndarray)):
        images = [copy_atoms(image) for image in method[1:-1]]
        images = [copy_atoms(start)] + images + [copy_atoms(end)]
    elif isinstance(method, str) and method.lower() not in [
        "linear",
        "idpp",
        "rep",
        "born",
        "ends",
    ]:
        # Import interpolation from a trajectory file
        images = read(method, "-{}:".format(n_images))
        images = [copy_atoms(start)] + images[1:-1] + [copy_atoms(end)]
    else:
        # Make path by the NEB methods interpolation
        images = [start.copy() for _ in range(1, n_images - 1)]
        images = [copy_atoms(start)] + images + [copy_atoms(end)]
        if method.lower() == "ends":
            images = make_end_interpolations(
                images,
                mic=mic,
                **interpolation_kwargs,
            )
        else:
            images = make_linear_interpolation(
                images,
                mic=mic,
                **interpolation_kwargs,
            )
            if method.lower() == "idpp":
                images = make_idpp_interpolation(
                    images,
                    mic=mic,
                    **interpolation_kwargs,
                )
            elif method.lower() == "rep":
                images = make_rep_interpolation(
                    images,
                    mic=mic,
                    **interpolation_kwargs,
                )
            elif method.lower() == "born":
                images = make_born_interpolation(
                    images,
                    mic=mic,
                    **interpolation_kwargs,
                )
    return images


def make_linear_interpolation(images, mic=False, **kwargs):
    "Make the linear interpolation from initial to final state."
    # Get the position of initial state
    pos0 = images[0].get_positions()
    # Get the distance to the final state
    dist_vec = images[-1].get_positions() - pos0
    # Calculate the minimum-image convention if mic=True
    if mic:
        _, dist_vec = mic_distance(
            dist_vec,
            cell=images[0].get_cell(),
            pbc=images[0].pbc,
            use_vector=True,
        )
    # Calculate the distance moved for each image
    dist_vec = dist_vec / float(len(images) - 1)
    # Set the positions
    for i in range(1, len(images) - 1):
        images[i].set_positions(pos0 + (i * dist_vec))
    return images


def make_idpp_interpolation(
    images,
    mic=False,
    fmax=1.0,
    steps=100,
    local_opt=FIRE,
    local_kwargs={},
    **kwargs,
):
    """
    Make the IDPP interpolation from initial to final state
    from NEB optimization.
    """

    from ...regression.gp.baseline import IDPP

    # Get all distances in the system
    dist0 = images[0].get_all_distances(mic=mic)
    # Calculate the differences in the distances in the system for IDPP
    dist = (images[-1].get_all_distances(mic=mic) - dist0) / float(
        len(images) - 1
    )
    # Use IDPP as calculator
    for i, image in enumerate(images[1:-1]):
        target = dist0 + (i + 1) * dist
        image.calc = IDPP(target=target, mic=mic)
    # Make default NEB
    neb = ImprovedTangentNEB(images)
    # Set local optimizer arguments
    local_kwargs_default = dict(trajectory="idpp.traj", logfile="idpp.log")
    if isinstance(local_opt, FIRE):
        local_kwargs_default.update(
            dict(dt=0.05, a=1.0, astart=1.0, fa=0.999, maxstep=0.2)
        )
    local_kwargs_default.update(local_kwargs)
    # Optimize NEB path with IDPP
    with local_opt(neb, **local_kwargs_default) as opt:
        opt.run(fmax=fmax, steps=steps)
    return images


def make_rep_interpolation(
    images,
    mic=False,
    fmax=1.0,
    steps=100,
    local_opt=FIRE,
    local_kwargs={},
    calc_kwargs={},
    trajectory="rep.traj",
    logfile="rep.log",
    **kwargs,
):
    """
    Make a repulsive potential to get the interpolation from NEB optimization.
    """
    from ...regression.gp.baseline import RepulsionCalculator

    # Use Repulsive potential as calculator
    for image in images[1:-1]:
        image.calc = RepulsionCalculator(mic=mic, **calc_kwargs)
    # Make default NEB
    neb = ImprovedTangentNEB(images)
    # Set local optimizer arguments
    local_kwargs_default = dict(trajectory=trajectory, logfile=logfile)
    if isinstance(local_opt, FIRE):
        local_kwargs_default.update(
            dict(dt=0.05, a=1.0, astart=1.0, fa=0.999, maxstep=0.2)
        )
    local_kwargs_default.update(local_kwargs)
    # Optimize NEB path with repulsive potential
    with local_opt(neb, **local_kwargs_default) as opt:
        opt.run(fmax=fmax, steps=steps)
    return images


def make_born_interpolation(
    images,
    mic=False,
    fmax=1.0,
    steps=100,
    local_opt=FIRE,
    local_kwargs={},
    calc_kwargs={},
    trajectory="born.traj",
    logfile="born.log",
    **kwargs,
):
    """
    Make a Born repulsive potential to get the interpolation from NEB
    optimization.
    """
    from ...regression.gp.baseline import BornRepulsionCalculator

    # Use Repulsive potential as calculator
    for image in images[1:-1]:
        image.calc = BornRepulsionCalculator(mic=mic, **calc_kwargs)
    # Make default NEB
    neb = ImprovedTangentNEB(images)
    # Set local optimizer arguments
    local_kwargs_default = dict(trajectory=trajectory, logfile=logfile)
    if isinstance(local_opt, FIRE):
        local_kwargs_default.update(
            dict(dt=0.05, a=1.0, astart=1.0, fa=0.999, maxstep=0.2)
        )
    local_kwargs_default.update(local_kwargs)
    # Optimize NEB path with repulsive potential
    with local_opt(neb, **local_kwargs_default) as opt:
        opt.run(fmax=fmax, steps=steps)
    return images


def make_end_interpolations(images, mic=False, trust_dist=0.2, **kwargs):
    """
    Make the linear interpolation from initial to final state,
    but place the images at the initial and final states with
    the maximum distance as trust_dist.
    """
    # Get the number of images
    n_images = len(images)
    # Get the position of initial state
    pos0 = images[0].get_positions()
    # Get the distance to the final state
    dist_vec = images[-1].get_positions() - pos0
    # Calculate the minimum-image convention if mic=True
    if mic:
        _, dist_vec = mic_distance(
            dist_vec,
            cell=images[0].get_cell(),
            pbc=images[0].pbc,
            use_vector=True,
        )
    # Calculate the scaled distance
    scale_dist = 2.0 * trust_dist / norm(dist_vec)
    # Check if the distance is within the trust distance
    if scale_dist >= 1.0:
        # Calculate the distance moved for each image
        dist_vec = dist_vec / float(n_images - 1)
        # Set the positions
        for i in range(1, n_images - 1):
            images[i].set_positions(pos0 + (i * dist_vec))
        return images
    # Calculate the distance moved for each image
    dist_vec = dist_vec * (scale_dist / float(n_images - 1))
    # Get the position of final state
    posn = images[-1].get_positions()
    # Set the positions
    nfirst = int(0.5 * (n_images - 1))
    for i in range(1, n_images - 1):
        if i <= nfirst:
            images[i].set_positions(pos0 + (i * dist_vec))
        else:
            images[i].set_positions(posn - ((n_images - 1 - i) * dist_vec))
    return images


def get_images_distance(images):
    "Get the cumulative distacnce of the images."
    dis = 0.0
    for i in range(len(images) - 1):
        dis += norm(images[i + 1].get_positions() - images[i].get_positions())
    return dis
