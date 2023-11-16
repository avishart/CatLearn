import numpy as np
import itertools
from scipy.spatial.distance import cdist

def get_all_distances(atoms,not_masked=None,mic=False,vector=False,**kwargs):
    " Get the cartesian distances between the atomes and including the vectors if vector=True. "
    # If a not masked list is given all atoms is treated to be not masked
    if not_masked is None:
        not_masked=np.arange(len(atoms))
    # Get the periodic boundary conditions
    pbc=atoms.pbc.copy()
    # Check if the minimum image convention is used and if there is any pbc
    if not mic or sum(pbc)==0:
        # Get the atomic positions
        pos=atoms.get_positions()
        # Get only the distances
        if not vector:
            return cdist(pos[not_masked],pos),None
        # Get the distances and the distance vectors
        dist_vec=pos-pos[not_masked,None]
        return np.linalg.norm(dist_vec,axis=2),dist_vec    
    # Get the minimum image convention distances and distance vectors
    return mic_distance(atoms,not_masked,pbc,vector=vector,**kwargs)
    
def mic_distance(atoms,not_masked,pbc,vector=False,**kwargs):
    " Get the minimum image convention of the distances. "
    # Get the atomic positions
    pos=atoms.get_positions()
    # Get the initial distance vectors
    dist_vec=pos-pos[not_masked,None]
    # Save the shortest distances
    v2min=dist_vec**2
    if vector:
        vmin=dist_vec.copy()
    else:
        vmin=None
    # Get the cell and its norm
    cell=np.array(atoms.cell)
    cell2=cell**2
    # Find what dimensions have cubic unit cells and not
    d_c=[]
    pbc_nc=[False,False,False]
    if pbc[0]:
        if cell2[0,1]+cell2[0,2]+cell2[1,0]+cell2[2,0]==0.0:
            d_c.append(0)
        else:
            pbc_nc[0]=True
    if pbc[1]:
        if cell2[1,0]+cell2[1,2]+cell2[0,1]+cell2[2,1]==0.0:
            d_c.append(1)
        else:
            pbc_nc[1]=True
    if pbc[2]:
        if cell2[2,0]+cell2[2,1]+cell2[0,2]+cell2[1,2]==0.0:
            d_c.append(2)
        else:
            pbc_nc[2]=True
    # Check if the cell is cubic to do a simpler mic
    if len(d_c):
        v2min,vmin=mic_cubic_distance(dist_vec,v2min,vmin,d_c,cell,vector=vector,**kwargs)
    else:
        v2min=np.sum(v2min,axis=-1)
    if sum(pbc_nc):
        # Do an extensive mic for the dimension that is not cubic
        v2min,vmin=mic_general_distance(dist_vec,v2min,vmin,pbc_nc,cell,vector=vector,**kwargs)
    return np.sqrt(v2min),vmin

def mic_cubic_distance(dist_vec,v2min,vmin,d_c,cell,vector=False,**kwargs):
    " Get the minimum image convention of the distances for cubic unit cells. It is faster than the extensive mic. "
    # Iterate over the x-, y-, and z-dimensions if they are periodic and cubic
    for d in d_c:
        # Calculate the distances to the atoms in the next unit cell
        dv_new=dist_vec[:,:,d]+cell[d,d]
        dv2_new=dv_new**2
        ## Save the new distances if they are shorter
        ix,iy=np.where(dv2_new<v2min[:,:,d])
        v2min[ix,iy,d]=dv2_new[ix,iy]
        if vector:
            vmin[ix,iy,d]=dv_new[ix,iy]
        # Calculate the distances to the atoms in the previous unit cell
        dv_new=dist_vec[:,:,d]-cell[d,d]
        dv2_new=dv_new**2
        ## Save the new distances if they are shorter
        ix,iy=np.where(dv2_new<v2min[:,:,d])
        v2min[ix,iy,d]=dv2_new[ix,iy]
        if vector:
            vmin[ix,iy,d]=dv_new[ix,iy]
    # Calculate the distances
    if vector:
        return np.sum(v2min,axis=-1),vmin
    return np.sum(v2min,axis=-1),None

def mic_general_distance(dist_vec,Dmin,vmin,pbc_nc,cell,vector=False,**kwargs):
    " Get the minimum image convention of the distances for any unit cells with an extensive mic search. "
    # Make all periodic combinations
    b=[[-1,0,1] if p else [0] for p in pbc_nc]
    p_arrays=list(itertools.product(*b))
    # Remove the initial combination
    p_arrays.remove((0,0,0))
    # Calculate all displacement vector from the cell vectors 
    p_arrays=np.array(p_arrays)
    cells_p=np.matmul(p_arrays,cell)
    # Iterate over all combinations
    for p in range(len(p_arrays)):
        # Calculate the distances to the atoms in the next unit cell
        dv_new=dist_vec[:,:]+cells_p[p]
        D_new=np.sum(dv_new**2,axis=-1)
        ## Save the new distances if they are shorter
        ix,iy=np.where(D_new<Dmin) #np.where(Dmin-D_new<-1e-8) if small changes lead to change
        Dmin[ix,iy]=D_new[ix,iy]
        if vector:
            vmin[ix,iy]=dv_new[ix,iy]
    # Calculate the distances
    if vector:
        return Dmin,vmin
    return Dmin,None

