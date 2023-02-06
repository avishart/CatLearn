import numpy as np
from .fingerprint import Fingerprint

class Inv_distances(Fingerprint):
    " The inverse distance fingerprint where the sizes are sorted and scaled with covalent radii "
    
    def make_fingerprint(self,atoms,not_masked,**kwargs):
        " The calculation of the inverse distance fingerprint "
        vector,derivative=self.loop_elements_combi(atoms,not_masked,use_derivatives=self.use_derivatives,mic=self.mic)
        return vector,derivative
    
    def loop_elements_combi(self,atoms,not_masked,use_derivatives=True,mic=True):
        " Looping over each element type to calculate inverse distances "
        import itertools
        # Make list of elements and their combinations
        elements_list=self.element_setup(atoms)
        elements_combi=self.unique_comb(list(range(len(elements_list))))
        # Get all distances
        cov_dis,distances,vec_distances=self.get_cov_dis(atoms,mic,use_derivatives)
        # Make the fingerprint and gradient if needed
        fp=[]
        g=[] if use_derivatives else None
        n_atoms_g=len(not_masked)
        for ei,ej in elements_combi:
            if ei==ej:
                ele_combi=self.unique_comb(elements_list[ei])
            else:
                ele_combi=list(itertools.product(elements_list[ei],elements_list[ej]))
            fp,g=self.fp_deriv_iter(ele_combi,cov_dis,distances,vec_distances,n_atoms_g,not_masked,fp,g,use_derivatives)
        return np.concatenate(fp),g

    def fp_deriv_iter(self,ele_combi,cov_dis,distances,vec_distances,n_atoms_g,not_masked,fp,g,use_derivatives):
        " Calculate the derivative of the fingerprint simultaneously with calculation of the fingerprint "
        fpij=[]
        gtemp=[]
        not_masked_combi=False
        for elei,elej in ele_combi:
            if elei!=elej:
                if elei in not_masked or elej in not_masked:
                    not_masked_combi=True
                    cov_dis_ij=cov_dis[elei,elej]/distances[elei,elej]
                    fpij.append(cov_dis_ij)
                    if use_derivatives:
                        gij=np.zeros((n_atoms_g*3))
                        if elei in not_masked or elej in not_masked:
                            gij_value=(cov_dis_ij/(distances[elei,elej]**2))*vec_distances[elei,elej]
                            if elei in not_masked:
                                i=not_masked.index(elei)
                                gij[3*i:3*i+3]=gij_value
                            if elej in not_masked:
                                j=not_masked.index(elej)
                                gij[3*j:3*j+3]=-gij_value
                        gtemp.append(gij)
        if not_masked_combi:
            i_sort=np.argsort(fpij)[::-1]
            fp.append(np.array(fpij)[i_sort])
            if use_derivatives:
                if len(g)==0:
                    g=np.array(np.array(gtemp)[i_sort])
                else:
                    g=np.append(g,np.array(gtemp)[i_sort],axis=0)
        return fp,g
        
    def element_setup(self,atoms):
        " Get all informations of the atoms "
        tags=atoms.get_tags()
        tags_set=np.array(list(set(tags)))
        elements=atoms.get_atomic_numbers()
        elements_set=np.array(list(set(elements)))
        elements_list=[]
        for tag in tags_set:
            for e in elements_set:
                indicies=np.where((tags==tag) & (elements==e))[0]
                if len(indicies):               
                    elements_list.append(indicies)
        return elements_list

    def get_cov_dis(self,atoms,mic,use_derivatives):
        " Calculate the distances and scale them with the covalent radii "
        from ase.data import covalent_radii
        distances=atoms.get_all_distances(mic=mic)
        range_atoms=range(len(atoms))
        distances[range_atoms,range_atoms]=1.0
        cov_dis=covalent_radii[atoms.get_atomic_numbers()]
        cov_dis=cov_dis.reshape(-1,1)+cov_dis
        cov_dis[range_atoms,range_atoms]=0.0
        if use_derivatives:
            vec_distances=atoms.get_all_distances(mic=mic,vector=True)
            return cov_dis,distances,vec_distances
        return cov_dis,distances,None
    
    def unique_comb(self,a):
        " Make all unique combinations "
        comb=[]
        n=len(a)
        for i in range(n):
            for j in range(i,n):
                comb.append((a[i],a[j]))
        return comb
