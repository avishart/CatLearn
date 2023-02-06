import numpy as np
from ase.neb import NEB

def interpolation_ts(start,end,ts,n_images=15,interpolation='linear',mic=True,**interpolation_kwargs):
    " Make an interpolation with a TS guess. "
    images=make_interpolation(start,ts,n_images=n_images,interpolation=interpolation,mic=mic,**interpolation_kwargs)
    dis_st=get_images_distance(images)
    images=make_interpolation(ts,end,n_images=n_images,interpolation=interpolation,mic=mic,**interpolation_kwargs)
    dis_et=get_images_distance(images)
    n_images_st=int(n_images*dis_st/(dis_st+dis_et))
    n_images_st=2 if n_images_st<2 else n_images_st
    images=make_interpolation(start,ts,n_images=n_images_st,interpolation=interpolation,mic=mic,**interpolation_kwargs)
    images=images+make_interpolation(ts,end,n_images=int(n_images-n_images_st+1),interpolation=interpolation,mic=mic,**interpolation_kwargs)[1:]
    return images

def make_interpolation(start,end,n_images=15,interpolation='linear',mic=True,**interpolation_kwargs):
    " Make the NEB interpolation path. "
    # Make path by the NEB methods interpolation
    images=[start.copy() for i in range(n_images-1)]+[end.copy()]
    neb=NEB(images)
    if interpolation=='linear':
        neb.interpolate(mic=mic,**interpolation_kwargs)
    elif interpolation=='idpp':
        neb.interpolate(method='idpp',mic=mic,**interpolation_kwargs)
    return images

def get_images_distance(images):
    " Get the cumulative distacnce of the images. "
    dis=0.0
    for i in range(len(images)-1):
        dis+=np.linalg.norm(images[i+1].get_positions()-images[i].get_positions())
    return dis


