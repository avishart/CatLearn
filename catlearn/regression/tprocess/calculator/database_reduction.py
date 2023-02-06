import numpy as np
from scipy.spatial.distance import cdist
from .database import Database

class Database_Reduction(Database):
    def __init__(self,fingerprint=None,reduce_dimensions=True,use_derivatives=True,negative_forces=True,use_fingerprint=True,npoints=15,reduction_method='hybrid'):
        """ Database of ASE atoms objects that are converted into fingerprints and targets with only a limitted number in the database. 
            Parameters:
                fingerprint : Fingerprint object
                    An object as a fingerprint class that convert atoms to fingerprint.
                reduce_dimensions: bool
                    Whether to reduce the fingerprint space if constrains are used.
                use_derivatives : bool
                    Whether to use derivatives/forces in the targets.
                negative_forces : bool
                    Whether derivatives (True) or forces (False) are used.
                use_fingerprint : bool
                    Whether the kernel uses fingerprint objects (True) or arrays (False).
        """
        Database.__init__(self,fingerprint=fingerprint,reduce_dimensions=reduce_dimensions,use_derivatives=use_derivatives,negative_forces=negative_forces,use_fingerprint=use_fingerprint)
        self.npoints=npoints
        self.reduction_method=reduction_method
        self.update_indicies=True

    def append(self,atoms):
        " Append the atoms object, the fingerprint, and target to lists. "
        self.update_indicies=True
        atoms=self.copy_atoms(atoms)
        self.atoms_list.append(atoms)
        if self.use_fingerprint:
            self.features.append(self.fingerprint(atoms))
        else:
            self.features.append(self.fingerprint(atoms).get_vector())
        self.targets.append(self.get_target(atoms,use_derivatives=self.use_derivatives,negative_forces=self.negative_forces))
        return self

    def get_reduction_indicies(self):
        " Get the indicies of the data with the largest dispersion. "
        if not self.update_indicies:
            return self.indicies
        self.update_indicies=False
        features=np.array(self.features).copy()
        all_indicies=np.array(list(range(len(features))))
        if len(features)<=self.npoints:
            self.indicies=all_indicies.copy()
            return self.indicies
        if self.reduction_method=='distance':
            indicies=self.reduction_distances(all_indicies,features)
        elif self.reduction_method=='random':
            indicies=self.reduction_random(all_indicies)
        elif self.reduction_method=='hybrid':
            indicies=self.reduction_hybrid(all_indicies,features)
        elif self.reduction_method=='min':
            indicies=self.reduction_min()
        elif self.reduction_method=='last':
            indicies=self.reduction_last(all_indicies)
        self.indicies=indicies.copy()
        return indicies
        
    def reduction_distances(self,all_indicies,features):
        " Reduce the training set with the points farthest from each other. "
        indicies=np.array([0,1])
        for i in range(2,self.npoints):
            not_indicies=[j for j in all_indicies if j not in indicies]
            dist=cdist(features[indicies],features[not_indicies])
            i_max=np.argmax(np.nanmin(dist,axis=0))
            indicies=np.append(indicies,[all_indicies[not_indicies][i_max]])
        return indicies

    def reduction_random(self,all_indicies):
        " Random select the training points. "
        indicies=np.array([0,1])
        indicies=np.append(indicies,np.random.permutation(all_indicies[2:])[:int(self.npoints-2)])
        return indicies

    def reduction_hybrid(self,all_indicies,features):
        " Use a combination of random sampling and farthest distance to reduce training set. "
        indicies=np.array([0,1])
        for i in range(2,self.npoints):
            not_indicies=[j for j in all_indicies if j not in indicies]
            if i%3==0:
                indicies=np.append(indicies,[np.random.choice(all_indicies[not_indicies])])
            else:
                dist=cdist(features[indicies],features[not_indicies])
                i_max=np.argmax(np.nanmin(dist,axis=0))
                indicies=np.append(indicies,[all_indicies[not_indicies][i_max]])                
        return indicies

    def reduction_min(self):
        " Use the targets with smallest norms in the training set. "
        indicies=np.argsort(np.linalg.norm(np.array(self.targets),axis=1))[:int(self.npoints)]
        return indicies

    def reduction_last(self,all_indicies):
        " Use the targets with smallest norms in the training set. "
        indicies=np.append(all_indicies[:2],all_indicies[-int(self.npoints-2):])
        return indicies
    
    def get_atoms(self):
        " Get the list of atoms in the database. "
        indicies=self.get_reduction_indicies()
        return [atoms for i,atoms in enumerate(self.atoms_list.copy()) if i in indicies]
    
    def get_features(self):
        " Get all the fingerprints of the atoms in the database. "
        indicies=self.get_reduction_indicies()
        return np.array(self.features).copy()[indicies]
    
    def get_targets(self):
        " Get all the targets of the atoms in the database. "
        indicies=self.get_reduction_indicies()
        return np.array(self.targets).copy()[indicies]
    
    def __repr__(self):
        if self.use_derivatives:
            return "Database_Reduction({} Atoms objects without forces)".format(len(self.atoms_list))
        return "Database_Reduction({} Atoms objects with forces)".format(len(self.atoms_list))
    
    