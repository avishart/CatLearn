import numpy as np
from ase.io import read
from ase.optimize import FIRE
from ...regression.gp.calculator.copy_atoms import copy_atoms


def interpolate(
    start,
    end,
    ts=None,
    n_images=15,
    method="linear",
    mic=True,
    remove_rotation_and_translation=True,
    **interpolation_kwargs,
):
    """
    Make an interpolation between the start and end structure.
    A transition state structure can be given to guide the interpolation.
    """
    # The rotation and translation should be removed the end structure
    # is optimized compared to start structure
    if remove_rotation_and_translation:
        from ase.build import minimize_rotation_and_translation

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
    if isinstance(method, (list, np.ndarray)):
        images = [copy_atoms(image) for image in method]
    elif isinstance(method, str) and method.lower() not in [
        "linear",
        "idpp",
        "rep",
        "ends",
    ]:
        # Import interpolation from a trajectory file
        images = read(method, "-{}:".format(n_images))
    else:
        # Make path by the NEB methods interpolation
        images = [start.copy() for i in range(1, n_images - 1)]
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
    return images


def make_linear_interpolation(images, mic=False, **kwargs):
    "Make the linear interpolation from initial to final state."
    from ase.geometry import find_mic

    # Get the position of initial state
    pos0 = images[0].get_positions()
    # Get the distance to the final state
    dist = images[-1].get_positions() - pos0
    # Calculate the minimum-image convention if mic=True
    if mic:
        dist = find_mic(dist, images[0].get_cell(), images[0].pbc)[0]
    # Calculate the distance moved for each image
    dist = dist / float(len(images) - 1)
    # Set the positions
    for i in range(1, len(images) - 1):
        images[i].set_positions(pos0 + (i * dist))
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
    from .improvedneb import ImprovedTangentNEB
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
    **kwargs,
):
    """
    Make a repulsive potential to get the interpolation from NEB optimization.
    """
    from .improvedneb import ImprovedTangentNEB
    from ...regression.gp.baseline import RepulsionCalculator

    # Use Repulsive potential as calculator
    for image in images[1:-1]:
        image.calc = RepulsionCalculator(power=10, mic=mic)
    # Make default NEB
    neb = ImprovedTangentNEB(images)
    # Set local optimizer arguments
    local_kwargs_default = dict(trajectory="rep.traj", logfile="rep.log")
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
    from ase.geometry import find_mic

    # Get the number of images
    n_images = len(images)
    # Get the position of initial state
    pos0 = images[0].get_positions()
    # Get the distance to the final state
    dist = images[-1].get_positions() - pos0
    # Calculate the minimum-image convention if mic=True
    if mic:
        dist = find_mic(dist, images[0].get_cell(), images[0].pbc)[0]
    # Calculate the scaled distance
    scale_dist = 2.0 * trust_dist / np.linalg.norm(dist)
    # Check if the distance is within the trust distance
    if scale_dist >= 1.0:
        # Calculate the distance moved for each image
        dist = dist / float(n_images - 1)
        # Set the positions
        for i in range(1, n_images - 1):
            images[i].set_positions(pos0 + (i * dist))
        return images
    # Calculate the distance moved for each image
    dist = dist * (scale_dist / float(n_images - 1))
    # Get the position of final state
    posn = images[-1].get_positions()
    # Set the positions
    nfirst = int(0.5 * (n_images - 1))
    for i in range(1, n_images - 1):
        if i <= nfirst:
            images[i].set_positions(pos0 + (i * dist))
        else:
            images[i].set_positions(posn - ((n_images - 1 - i) * dist))
    return images


def get_images_distance(images):
    "Get the cumulative distacnce of the images."
    dis = 0.0
    for i in range(len(images) - 1):
        dis += np.linalg.norm(
            images[i + 1].get_positions() - images[i].get_positions()
        )
    return dis
