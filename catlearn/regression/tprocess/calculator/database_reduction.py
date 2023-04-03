import numpy as np
from scipy.spatial.distance import cdist
from .database import Database

class Database_Reduction(Database):
    def __init__(self,fingerprint=None,reduce_dimensions=True,use_derivatives=True,negative_forces=True,use_fingerprint=True,npoints=25,initial_indicies=[0]):
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
                npoints : int
                    Number of points that are used from the database.
                initial_indicies : list
                    The indicies of the data points that must be included in the final data base.
        """
        super.__init__(fingerprint=fingerprint,reduce_dimensions=reduce_dimensions,use_derivatives=use_derivatives,negative_forces=negative_forces,use_fingerprint=use_fingerprint)
        self.npoints=npoints
        self.initial_indicies=np.array(initial_indicies).copy()
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
        " Get the indicies of the data used. "
        # If the indicies is already calculated then give them
        if not self.update_indicies:
            return self.indicies
        # Set up all the indicies 
        self.update_indicies=False
        data_len=self.__len__()
        all_indicies=np.array(list(range(data_len)))
        # No reduction is needed if the database is not large  
        if data_len<=self.npoints:
            self.indicies=all_indicies.copy()
            return self.indicies
        # Reduce the data base 
        self.indicies=self.make_reduction(all_indicies).copy()
        return self.indicies
    
    def make_reduction(self,all_indicies):
        " Make the reduction of the data base with a chosen method. "
        raise NotImplementedError()
    
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
    
    

class DatabaseDistance(Database_Reduction):
    """ Database of ASE atoms objects that are converted into fingerprints and targets with only a limitted number in the database defined from the distances. """

    def make_reduction(self,all_indicies):
        " Reduce the training set with the points farthest from each other. "
        indicies=self.initial_indicies.copy()
        features=np.array(self.features).copy()
        for i in range(len(indicies),self.npoints):
            not_indicies=[j for j in all_indicies if j not in indicies]
            dist=cdist(features[indicies],features[not_indicies])
            i_max=np.argmax(np.nanmin(dist,axis=0))
            indicies=np.append(indicies,[not_indicies[i_max]])
        return indicies
    
class DatabaseRandom(Database_Reduction):
    """ Database of ASE atoms objects that are converted into fingerprints and targets with only a limitted number in the database defined from random. """

    def make_reduction(self,all_indicies):
        " Random select the training points. "
        indicies=self.initial_indicies.copy()
        not_indicies=[j for j in all_indicies if j not in indicies]
        indicies=np.append(indicies,np.random.permutation(not_indicies)[:int(self.npoints-len(indicies))])
        return indicies
    
class DatabaseHybrid(Database_Reduction):
    """ Database of ASE atoms objects that are converted into fingerprints and targets with only a limitted number in the database defined from distance and random. """

    def make_reduction(self,all_indicies):
        " Use a combination of random sampling and farthest distance to reduce training set. "
        indicies=self.initial_indicies.copy()
        features=np.array(self.features).copy()
        for i in range(len(indicies),self.npoints):
            not_indicies=[j for j in all_indicies if j not in indicies]
            if i%3==0:
                indicies=np.append(indicies,[np.random.choice(not_indicies)])
            else:
                dist=cdist(features[indicies],features[not_indicies])
                i_max=np.argmax(np.nanmin(dist,axis=0))
                indicies=np.append(indicies,[not_indicies[i_max]])                
        return indicies
    
class DatabaseMin(Database_Reduction):
    """ Database of ASE atoms objects that are converted into fingerprints and targets with only a limitted number in the database defined from the smallest targets. """

    def make_reduction(self,all_indicies):
        " Use the targets with smallest norms in the training set. "
        indicies=self.initial_indicies.copy()
        not_indicies=np.array([j for j in all_indicies if j not in indicies])
        i_sort=np.argsort(np.linalg.norm(np.array(self.targets)[not_indicies],axis=1))
        indicies=np.append(indicies,not_indicies[i_sort[:int(self.npoints-len(indicies))]])
        return indicies
    
class DatabaseLast(Database_Reduction):
    """ Database of ASE atoms objects that are converted into fingerprints and targets with only a limitted number in the database defined from the last data points. """

    def make_reduction(self,all_indicies):
        " Use the last data points. "
        indicies=self.initial_indicies.copy()
        not_indicies=[j for j in all_indicies if j not in indicies]
        indicies=np.append(indicies,not_indicies[-int(self.npoints-len(indicies)):])
        return indicies