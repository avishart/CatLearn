import numpy as np

def interpolate(start,end,ts=None,n_images=15,method='linear',mic=True,remove_rotation_and_translation=False,**interpolation_kwargs):
    """ Make an interpolation between the start and end structure. 
        A transition state structure can be given to guide the interpolation. """
    # The rotation and translation should be removed the end structure is optimized compared to start structure
    if remove_rotation_and_translation:
        from ase.build import minimize_rotation_and_translation
        start.center()
        end.center()
        minimize_rotation_and_translation(start,end)
    # If the transition state is not given then make a regular interpolation
    if ts is not None:
        return make_interpolation(start,end,n_images=n_images,method=method,mic=mic,**interpolation_kwargs)
    # Get the interpolated path from the start structure to the TS structure
    images=make_interpolation(start,ts,n_images=n_images,method=method,mic=mic,**interpolation_kwargs)
    # Get the cumulative distance from the start to the TS structure 
    dis_st=get_images_distance(images)
    # Get the interpolated path from the TS structure to the end structure
    images=make_interpolation(ts,end,n_images=n_images,method=method,mic=mic,**interpolation_kwargs)
    # Get the cumulative distance from the TS to the end structure
    dis_et=get_images_distance(images)
    # Calculate the number of images from start to the TS from the distance
    n_images_st=int(n_images*dis_st/(dis_st+dis_et))
    n_images_st=2 if n_images_st<2 else n_images_st
    # Get the interpolated path from the start structure to the TS structure with the correct number of images
    images=make_interpolation(start,ts,n_images=n_images_st,method=method,mic=mic,**interpolation_kwargs)
    # Get the interpolated path from the TS structure to the end structure with the corrct number of images
    images=images+make_interpolation(ts,end,n_images=int(n_images-n_images_st+1),method=method,mic=mic,**interpolation_kwargs)[1:]
    return images

def make_interpolation(start,end,n_images=15,method='linear',mic=True,**interpolation_kwargs):
    " Make the NEB interpolation path. "
    # Use a premade interpolation path
    if isinstance(method,(list,np.ndarray)):
        images=method.copy()
    else:
        # Use the linear or Image Dependent Pair Potential interpolation method
        if method in ['linear','idpp']:
            # Make path by the NEB methods interpolation
            from ase.neb import NEB
            neb=NEB(images)
            if method.lower()=='linear':
                neb.interpolate(mic=mic,**interpolation_kwargs)
            elif method.lower()=='idpp':
                neb.interpolate(method='idpp',mic=mic,**interpolation_kwargs)
        else:
            # Import interpolation from a trajectory file
            from ase.io import read
            images=read(method,'-{}:'.format(n_images))
    return images

def get_images_distance(images):
    " Get the cumulative distacnce of the images. "
    dis=0.0
    for i in range(len(images)-1):
        dis+=np.linalg.norm(images[i+1].get_positions()-images[i].get_positions())
    return dis


