import numpy as np
import itertools
from scipy.spatial.distance import cdist
from ase.data import covalent_radii

def get_full_distance_matrix(atoms,not_masked=None,mic=False,vector=False,wrap=False,**kwargs):
    " Get the full cartesian distance matrix between the atomes and including the vectors if vector=True. "
    # If a not masked list is given all atoms is treated to be not masked
    if not_masked is None:
        not_masked=np.arange(len(atoms))
    # Get the atomic positions
    pos=atoms.get_positions(wrap=wrap)
    # Get distance vectors
    if vector or mic:
        dist_vec=pos-pos[not_masked,None]
    # Get the periodic boundary conditions
    pbc=atoms.pbc.copy()
    # Check if the minimum image convention is used and if there is any pbc
    if not mic or sum(pbc)==0:
        # Get only the distances
        if not vector:
            return cdist(pos[not_masked],pos),None
        return np.linalg.norm(dist_vec,axis=-1),dist_vec    
    # Get the cell vectors
    cell=np.array(atoms.cell)
    # Get the minimum image convention distances and distance vectors
    return mic_distance(dist_vec,not_masked,pbc,cell,vector=vector,**kwargs)

def get_all_distances(atoms,not_masked=None,masked=None,nmi=None,nmi_ind=None,nmj_ind=None,mic=False,vector=False,wrap=False,**kwargs):
    " Get the unique cartesian distances between the atomes and including the vectors if vector=True. "
    # If a not masked list is given all atoms is treated to be not masked
    if not_masked is None:
        not_masked=np.arange(len(atoms))
    if masked is None:
        masked=np.array(list(set(np.arange(len(atoms))).difference(set(not_masked))))
    # Make indicies
    if nmi is None or nmi_ind is None or nmj_ind is None:
        nmi,nmj=np.triu_indices(len(not_masked),k=1,m=None)
        nmi_ind=not_masked[nmi]
        nmj_ind=not_masked[nmj]
    # Get the atomic positions
    pos=atoms.get_positions(wrap=wrap)
    # Get distance vectors
    if vector or mic:
        dist_vec=np.concatenate([(pos[masked]-pos[not_masked,None]).reshape(-1,3),pos[nmj_ind]-pos[nmi_ind]],axis=0)
    # Get the periodic boundary conditions
    pbc=atoms.pbc.copy()
    # Check if the minimum image convention is used and if there is any pbc
    if not mic or sum(pbc)==0:
        if not vector:
            d=cdist(pos[not_masked],pos)
            return np.concatenate([d[:,masked].reshape(-1),d[nmi,nmj_ind]],axis=0),None
        return np.linalg.norm(dist_vec,axis=-1),dist_vec    
    # Get the cell vectors
    cell=np.array(atoms.cell)
    # Get the minimum image convention distances and distance vectors
    return mic_distance(dist_vec,not_masked,pbc,cell,vector=vector,**kwargs)
    
def mic_distance(dist_vec,not_masked,pbc,cell,vector=False,**kwargs):
    " Get the minimum image convention of the distances. "
    # Get the squared cell vectors
    cell2=cell**2
    # Save the shortest distances
    v2min=dist_vec**2
    if vector:
        vmin=dist_vec.copy()
    else:
        vmin=None
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
        dv_new=dist_vec[...,d]+cell[d,d]
        dv2_new=dv_new**2
        ## Save the new distances if they are shorter
        i=np.where(dv2_new<v2min[...,d])
        #ix,iy=np.where(dv2_new-v2min[...,d]<-1e-8)
        v2min[*i,d]=dv2_new[*i]
        if vector:
            vmin[*i,d]=dv_new[*i]
        # Calculate the distances to the atoms in the previous unit cell
        dv_new=dist_vec[...,d]-cell[d,d]
        dv2_new=dv_new**2
        ## Save the new distances if they are shorter
        i=np.where(dv2_new<v2min[...,d])
        #ix,iy=np.where(dv2_new-v2min[...,d]<-1e-8)
        v2min[*i,d]=dv2_new[*i]
        if vector:
            vmin[*i,d]=dv_new[*i]
    # Calculate the distances
    if vector:
        return np.sum(v2min,axis=-1),vmin
    return np.sum(v2min,axis=-1),None

def mic_general_distance(dist_vec,Dmin,vmin,pbc_nc,cell,vector=False,**kwargs):
    " Get the minimum image convention of the distances for any unit cells with an extensive mic search. "
    # Calculate all displacement vectors from the cell vectors 
    cells_p=get_periodicities(pbc_nc,cell)
    # Iterate over all combinations
    for p_array in cells_p:
        # Calculate the distances to the atoms in the next unit cell
        dv_new=dist_vec+p_array
        D_new=np.sum(dv_new**2,axis=-1)
        ## Save the new distances if they are shorter
        ix,iy=np.where(D_new<Dmin) 
        #ix,iy=np.where(Dmin-D_new<-1e-8)# if small changes lead to change
        Dmin[ix,iy]=D_new[ix,iy]
        if vector:
            vmin[ix,iy]=dv_new[ix,iy]
    # Calculate the distances
    if vector:
        return Dmin,vmin
    return Dmin,None

def get_periodicities(pbc,cell,remove0=True,**kwargs):
    " Get all displacement vectors from the periodicity and cell vectors. "
    # Make all periodic combinations
    b=[[-1,0,1] if p else [0] for p in pbc]
    p_arrays=list(itertools.product(*b))
    # Remove the initial combination
    if remove0:
        p_arrays.remove((0,0,0))
    # Calculate all displacement vector from the cell vectors 
    p_arrays=np.array(p_arrays)
    cells_p=np.matmul(p_arrays,cell)
    return cells_p

def get_inverse_distances(atoms,not_masked=None,masked=None,nmi=None,nmj=None,nmi_ind=None,nmj_ind=None,use_derivatives=True,use_covrad=True,periodic_softmax=True,mic=False,wrap=True,eps=1e-16,**kwargs):
    " Get the inverse cartesian distances between the atomes. The derivatives can also be obtained. "
    # If a not masked list is given all atoms is treated to be not masked
    if not_masked is None:
        not_masked=np.arange(len(atoms))
    if masked is None:
        masked=np.array(list(set(np.arange(len(atoms))).difference(set(not_masked))))
    # Make indicies
    if nmi is None or nmj is None or nmi_ind is None or nmj_ind is None:
        nmi,nmj=np.triu_indices(len(not_masked),k=1,m=None)
        nmi_ind=not_masked[nmi]
        nmj_ind=not_masked[nmj]
    # Get the covalent radii
    if use_covrad:
        covrad=covalent_radii[atoms.get_atomic_numbers()]
        covrad=np.concatenate([(covrad[masked]+covrad[not_masked,None]).reshape(-1),covrad[nmj_ind]+covrad[nmi_ind]],axis=0)
    else:
        covrad=1.0
    # Get inverse distances
    if periodic_softmax and atoms.pbc.any():
        # Use a softmax function to weight the inverse distances
        distances,vec_distances=get_all_distances(atoms,not_masked=not_masked,masked=masked,nmi=nmi,nmj_ind=nmj_ind,mic=False,vector=True,wrap=wrap,**kwargs)
        # Calculate all displacement vectors from the cell vectors 
        cells_p=get_periodicities(atoms.pbc,atoms.get_cell(),remove0=False)
        c_dim=len(cells_p)
        # Calculate the distances to the atoms in all unit cell
        d=vec_distances+cells_p.reshape(c_dim,1,3)
        # Add small number to avoid division by zero to the distances
        dnorm=np.linalg.norm(d,axis=-1)+eps
        # Calculate weights
        dcov=dnorm/covrad
        w=np.exp(-(dcov**2))
        w=w/np.sum(w,axis=0)
        # Calculate inverse distances
        finner=w/dcov
        f=np.sum(finner,axis=0)
        # Calculate derivatives of inverse distances
        if use_derivatives:
            inner=((2.0*(1.0-(dcov*f)))/(covrad**2))+(1.0/(dnorm**2))
            gij=np.sum(d*(finner*inner).reshape(c_dim,-1,1),axis=0)
    else:
        distances,vec_distances=get_all_distances(atoms,not_masked=not_masked,masked=masked,nmi=nmi,nmj_ind=nmj_ind,mic=mic,vector=use_derivatives,wrap=wrap,**kwargs)
        # Add small number to avoid division by zero to the distances
        distances=distances+eps
        # Calculate inverse distances
        f=covrad/distances
        # Calculate derivatives of inverse distances
        if use_derivatives:
            gij=vec_distances*(covrad/(distances**3)).reshape(-1,1)
    if use_derivatives:
        # Convert derivatives to the right matrix form
        n_total=len(f)
        g=np.zeros((n_total,len(not_masked)*3))
        # The derivative of not fixed (not masked) and fixed atoms
        n_nm_m=len(not_masked)*len(masked)
        if n_nm_m:
            i_g=np.repeat(np.arange(n_nm_m),3)
            j_g=np.tile(3*np.arange(len(not_masked)).reshape(-1,1)+np.array([0,1,2]),(1,len(masked))).reshape(-1)
            g[i_g,j_g]=gij[:n_nm_m].reshape(-1)
        # The derivative of not fixed (not masked) and not fixed atoms
        if len(nmi):
            i_g=np.repeat(np.arange(n_nm_m,n_total),3)
            j_gi=(3*nmi.reshape(-1,1)+np.array([0,1,2])).reshape(-1)
            j_gj=(3*nmj.reshape(-1,1)+np.array([0,1,2])).reshape(-1)
            g[i_g,j_gi]=gij[n_nm_m:].reshape(-1)
            g[i_g,j_gj]=-g[i_g,j_gi]
        return f,g
    return f,None


