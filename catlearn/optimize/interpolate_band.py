import numpy as np


def interpolate(start,end,ts=None,n_images=15,method='linear',mic=True,remove_rotation_and_translation=True,**interpolation_kwargs):
    """ 
    Make an interpolation between the start and end structure. 
    A transition state structure can be given to guide the interpolation. 
    """
    # The rotation and translation should be removed the end structure is optimized compared to start structure
    if remove_rotation_and_translation:
        from ase.build import minimize_rotation_and_translation
        start.center()
        end.center()
        minimize_rotation_and_translation(start,end)
    # If the transition state is not given then make a regular interpolation
    if ts is None:
        return make_interpolation(start,end,n_images=n_images,method=method,mic=mic,remove_rotation_and_translation=remove_rotation_and_translation,**interpolation_kwargs)
    # Get the interpolated path from the start structure to the TS structure
    images=make_interpolation(start,ts,n_images=n_images,method=method,mic=mic,remove_rotation_and_translation=remove_rotation_and_translation,**interpolation_kwargs)
    # Get the cumulative distance from the start to the TS structure 
    dis_st=get_images_distance(images)
    # Get the interpolated path from the TS structure to the end structure
    images=make_interpolation(ts,end,n_images=n_images,method=method,mic=mic,remove_rotation_and_translation=remove_rotation_and_translation,**interpolation_kwargs)
    # Get the cumulative distance from the TS to the end structure
    dis_et=get_images_distance(images)
    # Calculate the number of images from start to the TS from the distance
    n_images_st=int(n_images*dis_st/(dis_st+dis_et))
    n_images_st=2 if n_images_st<2 else n_images_st
    # Get the interpolated path from the start structure to the TS structure with the correct number of images
    images=make_interpolation(start,ts,n_images=n_images_st,method=method,mic=mic,remove_rotation_and_translation=remove_rotation_and_translation,**interpolation_kwargs)
    # Get the interpolated path from the TS structure to the end structure with the corrct number of images
    images=images+make_interpolation(ts,end,n_images=int(n_images-n_images_st+1),method=method,mic=mic,remove_rotation_and_translation=remove_rotation_and_translation,**interpolation_kwargs)[1:]
    return images

def make_interpolation(start,end,n_images=15,method='linear',mic=True,**interpolation_kwargs):
    " Make the NEB interpolation path. "
    # Use a premade interpolation path
    if isinstance(method,(list,np.ndarray)):
        images=method.copy()
    elif isinstance(method,str) and method not in ['linear','idpp']:
        # Import interpolation from a trajectory file
        from ase.io import read
        images=read(method,'-{}:'.format(n_images))
    else:
        # Make path by the NEB methods interpolation
        from ase.neb import NEB
        images=[start.copy() for i in range(n_images-1)]+[end.copy()]
        images=make_linear_interpolation(images,mic=mic)
        if method.lower()=='idpp':
            images=make_idpp_interpolation(images,mic=mic,**interpolation_kwargs)
    return images

def make_linear_interpolation(images,mic=False,**kwargs):
    " Make the linear interpolation from initial to final state. "
    from ase.geometry import find_mic
    # Get the position of initial state
    pos0=images[0].get_positions()
    # Get the distance to the final state
    dist=images[-1].get_positions()-pos0
    # Calculate the minimum-image convention if mic=True
    if mic:
        dist=find_mic(dist,images[0].get_cell(),images[0].pbc)[0]
    # Calculate the distance moved for each image
    dist=dist/float(len(images)-1)
    # Set the positions
    for i in range(1,len(images)-1):
        images[i].set_positions(pos0+(i*dist))
    return images

def make_idpp_interpolation(images,mic=False,fmax=0.1,steps=100,local_kwargs={},**kwargs):
    " Make the IDPP interpolation from initial to final state from NEB optimization. "
    from ase.geometry import find_mic
    from ase.neb import IDPP,NEB
    from ase.optimize import MDMin
    # Get all distances in the system
    dist0=images[0].get_all_distances(mic=mic)
    # Calculate the differences in the distances in the system for IDPP
    dist=(images[-1].get_all_distances(mic=mic)-dist0)/float(len(images)-1)
    # Use IDPP as calculator
    new_images=[]
    for i in range(len(images)):
        image=images[i].copy()
        image.calc=IDPP(dist0+i*dist,mic=mic)
        new_images.append(image)
    # Make default NEB 
    neb=NEB(new_images)
    # Set local optimizer arguments
    local_kwargs_default=dict(trajectory='idpp.traj',logfile='idpp.log',dt=0.05)
    local_kwargs_default.update(local_kwargs)
    # Optimize NEB path with IDPP
    with MDMin(neb,**local_kwargs_default) as opt:
        opt.run(fmax=fmax,steps=steps)
    return new_images
    

def get_images_distance(images):
    " Get the cumulative distacnce of the images. "
    dis=0.0
    for i in range(len(images)-1):
        dis+=np.linalg.norm(images[i+1].get_positions()-images[i].get_positions())
    return dis


