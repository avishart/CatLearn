import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from ase.io import read
from ..structures.neb import ImprovedTangentNEB


def plot_minimize(
    pred_atoms,
    eval_atoms,
    use_uncertainty=True,
    ax=None,
    loc=0,
    **kwargs,
):
    """
    Plot the predicted and evaluated atoms in a 2D plot.

    Parameters:
        pred_atoms: ASE atoms instance
            The predicted atoms.
        eval_atoms: ASE atoms instance
            The evaluated atoms.
        use_uncertainty: bool
            If True, use the uncertainty of the atoms.
        ax: matplotlib axis instance
            The axis to plot the NEB images.
        loc: int
            The location of the legend.

    Returns:
        ax: matplotlib axis instance
    """
    # Make figure if it is not given
    if ax is None:
        _, ax = plt.subplots()
    # Get the energies of the predicted atoms
    if isinstance(pred_atoms, str):
        pred_atoms = read(pred_atoms, ":")
    pred_energies = [atoms.get_potential_energy() for atoms in pred_atoms]
    if (
        "results" in pred_atoms[0].info
        and "predicted energy" in pred_atoms[0].info["results"]
    ):
        for i, atoms in enumerate(pred_atoms):
            pred_energies[i] = atoms.info["results"]["predicted energy"]
    elif hasattr(pred_atoms[0].calc, "results"):
        if "predicted energy" in pred_atoms[0].calc.results:
            for i, atoms in enumerate(pred_atoms):
                atoms.get_potential_energy()
                pred_atoms[i] = atoms.calc.results["predicted energy"]
    # Get the energies of the evaluated atoms
    if isinstance(eval_atoms, str):
        eval_atoms = read(eval_atoms, ":")
    eval_energies = [atoms.get_potential_energy() for atoms in eval_atoms]
    # Get the reference energy
    e_ref = eval_energies[0]
    # Get the uncertainties of the atoms if requested
    uncertainties = None
    if use_uncertainty:
        if (
            "results" in pred_atoms[0].info
            and "uncertainty" in pred_atoms[0].info["results"]
        ):
            uncertainties = [
                atoms.info["results"]["uncertainty"] for atoms in pred_atoms
            ]
        else:
            uncertainties = [
                atoms.calc.get_uncertainty(atoms) for atoms in pred_atoms
            ]
        uncertainties = np.array(uncertainties)
    # Make the energies relative to the first energy
    pred_energies = np.array(pred_energies) - e_ref
    eval_energies = np.array(eval_energies) - e_ref
    # Make x values
    x_values = np.arange(1, len(eval_energies) + 1)
    x_pred = x_values[-len(pred_energies) :]
    # Plot the energies of the atoms
    ax.plot(x_pred, pred_energies, "o-", color="red", label="Predicted")
    ax.plot(x_values, eval_energies, "o-", color="black", label="Evaluated")
    # Plot the uncertainties of the atoms if requested
    if uncertainties is not None:
        ax.fill_between(
            x_pred,
            pred_energies - uncertainties,
            pred_energies + uncertainties,
            color="red",
            alpha=0.3,
        )
        ax.fill_between(
            x_pred,
            pred_energies - 2.0 * uncertainties,
            pred_energies + 2.0 * uncertainties,
            color="red",
            alpha=0.2,
        )
    # Make labels
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Potential energy / [eV]")
    ax.legend(loc=loc)
    return ax


def get_neb_data(
    images,
    neb_method=ImprovedTangentNEB,
    neb_kwargs={},
    climb=False,
    use_uncertainty=False,
    use_projection=False,
):
    """
    Get the NEB data for plotting.

    Parameters:
        images: list of ASE atoms instances
            The images of the NEB calculation.
        neb_method: class
            The NEB method to use.
        neb_kwargs: dict
            The keyword arguments for the NEB method.
        climb: bool
            If True, use the climbing image method.
        use_uncertainty: bool
            If True, use the uncertainty of the images.
        use_projection: bool
            If True, use the projection of the derivatives on the tangent.
    """
    # Default values for the neb method
    used_neb_kwargs = dict(
        k=3.0,
        remove_rotation_and_translation=False,
        save_properties=True,
        mic=True,
    )
    used_neb_kwargs.update(neb_kwargs)
    # Initialize the NEB method
    neb = neb_method(images, climb=climb, **used_neb_kwargs)
    # Get the energies of the images
    energies = [image.get_potential_energy() for image in images]
    if (
        "results" in images[1].info
        and "predicted energy" in images[1].info["results"]
    ):
        for i, image in enumerate(images[1:-1]):
            energies[i + 1] = image.info["results"]["predicted energy"]
    elif hasattr(images[1].calc, "results"):
        if "predicted energy" in images[1].calc.results:
            for i, image in enumerate(images[1:-1]):
                image.get_potential_energy()
                energies[i + 1] = image.calc.results["predicted energy"]
    energies = np.array(energies) - energies[0]
    # Get the uncertainties of the images if requested
    uncertainties = None
    if use_uncertainty:
        if (
            "results" in images[1].info
            and "uncertainty" in images[1].info["results"]
        ):
            uncertainties = [
                image.info["results"]["uncertainty"] for image in images[1:-1]
            ]

        else:
            uncertainties = [
                image.calc.get_uncertainty(image) for image in images[1:-1]
            ]
        uncertainties = np.concatenate([[0.0], uncertainties, [0.0]])
    # Get the distances between the images
    pos_p, pos_m = neb.get_position_diff()
    distances = np.linalg.norm(pos_p, axis=(1, 2))
    distances = np.concatenate([[0.0], [np.linalg.norm(pos_m[0])], distances])
    distances = np.cumsum(distances)
    # Use projection of the derivatives on the tangent
    if use_projection:
        # Get the forces
        forces = [image.get_forces() for image in images]
        forces = np.array(forces)
        if (
            "results" in images[1].info
            and "predicted forces" in images[1].info["results"]
        ):
            for i, image in enumerate(images[1:-1]):
                forces[i + 1] = image.info["results"][
                    "predicted forces"
                ].copy()
        elif hasattr(images[1].calc, "results"):
            if "predicted forces" in images[1].calc.results:
                for i, image in enumerate(images[1:-1]):
                    image.get_forces()
                    forces[i + 1] = image.calc.results["predicted forces"]
        # Get the tangent
        tangent = neb.get_tangent(pos_p, pos_m)
        tangent = np.concatenate([[pos_m[0]], tangent, [pos_p[0]]], axis=0)
        tangent_norm = np.linalg.norm(tangent, axis=(1, 2)).reshape(-1, 1, 1)
        tangent = tangent / tangent_norm
        # Get the projection of the derivatives on the tangent
        deriv_proj = -np.sum(forces * tangent, axis=(1, 2))
    else:
        deriv_proj = None
    return neb, distances, energies, uncertainties, deriv_proj


def plot_neb(
    images,
    neb_method=ImprovedTangentNEB,
    neb_kwargs={},
    climb=True,
    use_uncertainty=True,
    use_projection=True,
    proj_len_scale=0.4,
    ax=None,
    **kwargs,
):
    """
    Plot the NEB images in a 2D plot.

    Parameters:
        images: list of ASE atoms instances
            The images of the NEB calculation.
        neb_method: class
            The NEB method to use.
        neb_kwargs: dict
            The keyword arguments for the NEB method.
        climb: bool
            If True, use the climbing image method.
        use_uncertainty: bool
            If True, use the uncertainty of the images.
        use_projection: bool
            If True, use the projection of the derivatives on the tangent.
        proj_len_scale: float
            The scale of the projection length.
            It is only used if use_projection is True.
        ax: matplotlib axis instance
            The axis to plot the NEB images.

    Returns:
        ax: matplotlib axis instance
    """
    # Get data from NEB
    _, distances, energies, uncertainties, deriv_proj = get_neb_data(
        images,
        neb_method=neb_method,
        neb_kwargs=neb_kwargs,
        climb=climb,
        use_uncertainty=use_uncertainty,
        use_projection=use_projection,
    )
    # Make figure if it is not given
    if ax is None:
        _, ax = plt.subplots()
    # Plot the NEB images
    ax.plot(distances, energies, "o-", color="black")
    if uncertainties is not None:
        ax.errorbar(
            distances,
            energies,
            yerr=uncertainties,
            color="black",
            capsize=3,
        )
        ax.errorbar(
            distances,
            energies,
            yerr=2.0 * uncertainties,
            color="black",
            capsize=1.5,
        )
    # Plot the projection of the derivatives
    if use_projection:
        # Get length of projection
        proj_len = proj_len_scale * distances[-1] / len(images)
        for i, deriv in enumerate(deriv_proj):
            dist = distances[i]
            energy = energies[i]
            x_range = [dist - proj_len, dist + proj_len]
            y_range = [energy - deriv * proj_len, energy + deriv * proj_len]
            ax.plot(
                x_range,
                y_range,
                color="red",
            )
    # Make labels
    ax.set_xlabel("Distance / [Å]")
    ax.set_ylabel("Potential energy / [eV]")
    title = "Reaction energy = {:.3f} eV \n".format(energies[-1])
    title += "Activation energy = {:.3f} eV".format(energies.max())
    ax.set_title(title)
    return ax


def plot_neb_fit_mlcalc(
    images,
    mlcalc,
    neb_method=ImprovedTangentNEB,
    neb_kwargs={},
    climb=True,
    use_uncertainty=True,
    distance_step=0.01,
    include_noise=True,
    ax=None,
    **kwargs,
):
    """
    Plot the NEB images in a 2D plot with the ML calculator predictions.

    Parameters:
        images: list of ASE atoms instances
            The images of the NEB calculation.
        mlcalc: ML calculator instance
            The ML calculator to use for the predictions.
        neb_method: class
            The NEB method to use.
        neb_kwargs: dict
            The keyword arguments for the NEB method.
        climb: bool
            If True, use the climbing image method.
        use_uncertainty: bool
            If True, use the uncertainty of the predictions.
        distance_step: float
            The step size for the distance between the images.
        include_noise: bool
            Whether to include noise in the uncertainty from the model.
        ax: matplotlib axis instance
            The axis to plot the NEB images.

    Returns:
        ax: matplotlib axis instance
    """
    # Get data from NEB
    neb, distances, energies, _, _ = get_neb_data(
        images,
        neb_method=neb_method,
        neb_kwargs=neb_kwargs,
        climb=climb,
        use_uncertainty=False,
        use_projection=False,
    )
    # Get the reference energy
    e0 = images[0].get_potential_energy()
    # Update whether to include noise in uncertainty prediction
    mlcalc = mlcalc.update_mlmodel_arguments(include_noise=include_noise)
    # Get the first image
    image = images[0].copy()
    image.calc = mlcalc
    pos0 = image.get_positions()
    # Get the distances between the images
    pos_p, pos_m = neb.get_position_diff()
    displacements = np.append([pos_m[0]], pos_p, axis=0)
    # Get the curve positions, energies, and uncertainties
    cum_distance = 0.0
    pred_distance = []
    pred_energies = []
    uncertainties = []
    for i, disp in enumerate(displacements):
        # Get the distance between the points on the curve
        dist = np.linalg.norm(disp)
        scalings = np.arange(0.0, dist, distance_step) / dist
        scalings = np.append(scalings, [1.0])
        if i != 0:
            scalings = scalings[1:]
        for scaling in scalings:
            # Calculate the position for the point on the curve
            pred_pos = pos0 + scaling * disp
            pred_distance.append(cum_distance + scaling * dist)
            image.set_positions(pred_pos)
            # Get the curve energy
            energy = image.get_potential_energy()
            if hasattr(image.calc, "results"):
                if "predicted energy" in image.calc.results:
                    energy = image.calc.results["predicted energy"]
            pred_energies.append(energy)
            # Get the curve uncertainty
            if use_uncertainty:
                unc = image.calc.get_uncertainty(image)
                uncertainties.append(unc)
        pos0 += disp
        cum_distance += dist
    # Make numpy arrays
    pred_distance = np.array(pred_distance)
    pred_energies = np.array(pred_energies) - e0
    uncertainties = np.array(uncertainties)
    # Make figure if it is not given
    if ax is None:
        _, ax = plt.subplots()
    # Plot the NEB images
    ax.plot(distances, energies, "o", color="black")
    ax.plot(pred_distance, pred_energies, "-", color="red")
    if len(uncertainties):
        ax.fill_between(
            pred_distance,
            pred_energies - uncertainties,
            pred_energies + uncertainties,
            color="red",
            alpha=0.3,
        )
        ax.fill_between(
            pred_distance,
            pred_energies - 2.0 * uncertainties,
            pred_energies + 2.0 * uncertainties,
            color="red",
            alpha=0.2,
        )
    # Make labels
    ax.set_xlabel("Distance / [Å]")
    ax.set_ylabel("Potential energy / [eV]")
    title = "Reaction energy = {:.3f} eV \n".format(energies[-1])
    title += "Activation energy = {:.3f} eV".format(pred_energies.max())
    ax.set_title(title)
    return ax


def plot_all_neb(
    neb_traj,
    n_images,
    neb_method=ImprovedTangentNEB,
    neb_kwargs={},
    ax=None,
    cmap=cm.jet,
    alpha=0.7,
    **kwargs,
):
    """
    Plot all the NEB images in a 2D plot.

    Parameters:
        neb_traj: list of ASE atoms instances or str
            The NEB trajectories of the NEB calculation.
            It can be a list of all ASE atoms instances for all NEB bands.
            It can also be a string to the file containing the NEB
            trajectories.
        n_images: int
            The number of images in each NEB band.
        neb_method: class
            The NEB method to use.
        neb_kwargs: dict
            The keyword arguments for the NEB method.
        ax: matplotlib axis instance
            The axis to plot the NEB images.
        cmap: matplotlib colormap
            The colormap to use for the NEB bands.
        alpha: float
            The transparency of the NEB bands except for the last.

    Returns:
        ax: matplotlib axis instance
    """
    # Calculate the number of NEB bands
    if isinstance(neb_traj, str):
        neb_traj = read(neb_traj, ":")
    n_neb = len(neb_traj) // n_images
    # Make figure if it is not given
    if ax is None:
        _, ax = plt.subplots()
    # Plot all NEB bands
    for i in range(n_neb):
        # Get the images of the NEB band
        images = neb_traj[i * n_images : (i + 1) * n_images]
        # Get data from NEB band
        _, distances, energies, _, _ = get_neb_data(
            images,
            neb_method=neb_method,
            neb_kwargs=neb_kwargs,
            climb=False,
            use_uncertainty=False,
            use_projection=False,
        )
        # Get the color
        if n_neb == 1:
            color = cmap(1)
        else:
            color = cmap(i / (n_neb - 1))
        # Get the transparency
        if i + 1 == n_neb:
            alpha = 1.0
        # Plot the NEB images
        ax.plot(distances, energies, "o-", color=color, alpha=alpha)
    # Add colorbar
    if n_neb == 1:
        colors = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    else:
        colors = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, n_neb - 1))
    cbar = plt.colorbar(colors, ax=ax)
    cbar.set_label("NEB band index")
    # Make labels
    ax.set_xlabel("Distance / [Å]")
    ax.set_ylabel("Potential energy / [eV]")
    return ax
