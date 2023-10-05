import numpy as np
from scipy.spatial.distance import cdist
from .database import Database

class Database_Reduction(Database):
    def __init__(self,fingerprint=None,reduce_dimensions=True,use_derivatives=True,use_fingerprint=True,npoints=25,initial_indicies=[0],**kwargs):
        """ 
        Database of ASE atoms objects that are converted into fingerprints and targets. 
        The used Database is a reduced set of the full Database. 
        The reduced data set is selected from a method. 

        Parameters:
            fingerprint : Fingerprint object
                An object as a fingerprint class that convert atoms to fingerprint.
            reduce_dimensions: bool
                Whether to reduce the fingerprint space if constrains are used.
            use_derivatives : bool
                Whether to use derivatives/forces in the targets.
            use_fingerprint : bool
                Whether the kernel uses fingerprint objects (True) or arrays (False).
            npoints : int
                Number of points that are used from the database.
            initial_indicies : list
                The indicies of the data points that must be included in the used data base.
        """
        # The negative forces have to be used since the derivatives are used in the ML models
        self.use_negative_forces=True
        # Set initial indicies
        self.indicies=[]
        # Use default fingerprint if it is not given
        if fingerprint is None:
            from ..fingerprint.cartesian import Cartesian
            fingerprint=Cartesian(reduce_dimensions=reduce_dimensions,use_derivatives=use_derivatives,mic=True)
        # Set the arguments
        self.update_arguments(fingerprint=fingerprint,
                              reduce_dimensions=reduce_dimensions,
                              use_derivatives=use_derivatives,
                              use_fingerprint=use_fingerprint,
                              npoints=npoints,
                              initial_indicies=initial_indicies,
                              **kwargs)

    def update_arguments(self,fingerprint=None,reduce_dimensions=None,use_derivatives=None,use_fingerprint=None,npoints=None,initial_indicies=None,**kwargs):
        """
        Update the class with its arguments. The existing arguments are used if they are not given.

        Parameters:
            fingerprint : Fingerprint object
                An object as a fingerprint class that convert atoms to fingerprint.
            reduce_dimensions: bool
                Whether to reduce the fingerprint space if constrains are used.
            use_derivatives : bool
                Whether to use derivatives/forces in the targets.
            use_fingerprint : bool
                Whether the kernel uses fingerprint objects (True) or arrays (False).
            npoints : int
                Number of points that are used from the database.
            initial_indicies : list
                The indicies of the data points that must be included in the used data base.

        Returns:
            self: The updated object itself.
        """
        if fingerprint is not None:
            self.fingerprint=fingerprint.copy()
            # Copy attributes from fingerprint
            self.mic=self.fingerprint.mic
        if reduce_dimensions is not None:
            self.reduce_dimensions=reduce_dimensions
        if use_derivatives is not None:
            self.use_derivatives=use_derivatives
        if use_fingerprint is not None:
            self.use_fingerprint=use_fingerprint
        if npoints is not None:
            self.npoints=int(npoints)
        if initial_indicies is not None:
            self.initial_indicies=np.array(initial_indicies,dtype=int)
        # Check that the database and the fingerprint have the same attributes
        self.check_attributes()
        # Reset the database
        self.reset_database()
        # Store that the data base has changed
        self.update_indicies=True
        return self
    
    def get_all_atoms(self,**kwargs):
        """
        Get the list of all atoms in the database.

        Returns:
            list: A list of the saved ASE Atoms objects.
        """
        return self.atoms_list.copy()
    
    def get_atoms(self,**kwargs):
        """
        Get the list of atoms in the reduced database.

        Returns:
            list: A list of the saved ASE Atoms objects.
        """
        indicies=self.get_reduction_indicies()
        return [atoms for i,atoms in enumerate(self.get_all_atoms(**kwargs)) if i in indicies]
    
    def get_features(self,**kwargs):
        """
        Get the fingerprints of the atoms in the reduced database.

        Returns:
            array: A matrix array with the saved features or fingerprints.
        """
        indicies=self.get_reduction_indicies()
        return np.array(self.features)[indicies]
    
    def get_all_feature_vectors(self,**kwargs):
        " Get all the features in numpy array form. "
        if self.use_fingerprint:
            return np.array([feature.get_vector() for feature in self.features])
        return np.array(self.features)
    
    def get_targets(self,**kwargs):
        """
        Get the targets of the atoms in the reduced database.

        Returns:
            array: A matrix array with the saved targets.
        """
        indicies=self.get_reduction_indicies()
        return np.array(self.targets)[indicies]
    
    def get_initial_indicies(self,**kwargs):
        """
        Get the initial indicies of the used atoms in the database.

        Returns:
            array: The initial indicies of the atoms used.
        """
        return self.initial_indicies.copy()
    
    def save_data(self,trajectory='data.traj',**kwargs):
        """
        Save the ASE Atoms data to a trajectory.

        Parameters:
            trajectory : str
                The name of the trajectory file where the data is saved.

        Returns:
            self: The updated object itself.
        """
        from ase.io import write
        write(trajectory,self.get_all_atoms())
        return self

    def append(self,atoms,**kwargs):
        " Append the atoms object, the fingerprint, and target(s) to lists. "
        # Store that the data base has changed
        self.update_indicies=True
        # Append to the data base
        super().append(atoms,**kwargs)
        return self

    def get_reduction_indicies(self,**kwargs):
        " Get the indicies of the reduced data used. "
        # If the indicies is already calculated then give them
        if not self.update_indicies:
            return self.indicies
        # Set up all the indicies 
        self.update_indicies=False
        data_len=self.__len__()
        all_indicies=np.arange(data_len)
        # No reduction is needed if the database is not large 
        if data_len<=self.npoints:
            self.indicies=all_indicies.copy()
            return self.indicies
        # Reduce the data base 
        self.indicies=self.make_reduction(all_indicies)
        return self.indicies
    
    def make_reduction(self,all_indicies,**kwargs):
        " Make the reduction of the data base with a chosen method. "
        raise NotImplementedError()
    
    def get_arguments(self):
        " Get the arguments of the class itself. "
        # Get the arguments given to the class in the initialization
        arg_kwargs=dict(fingerprint=self.fingerprint,
                        reduce_dimensions=self.reduce_dimensions,
                        use_derivatives=self.use_derivatives,
                        use_fingerprint=self.use_fingerprint,
                        npoints=self.npoints,
                        initial_indicies=self.initial_indicies)
        # Get the constants made within the class
        constant_kwargs=dict(mic=self.mic,
                             update_indicies=self.update_indicies)
        # Get the objects made within the class
        object_kwargs=dict(atoms_list=self.atoms_list.copy(),
                           features=self.features.copy(),
                           targets=self.targets.copy(),
                           indicies=self.indicies.copy())
        return arg_kwargs,constant_kwargs,object_kwargs


class DatabaseDistance(Database_Reduction):
    def __init__(self,fingerprint=None,reduce_dimensions=True,use_derivatives=True,use_fingerprint=True,npoints=25,initial_indicies=[0],**kwargs):
        """ 
        Database of ASE atoms objects that are converted into fingerprints and targets. 
        The used Database is a reduced set of the full Database. 
        The reduced data set is selected from the distances. 

        Parameters:
            fingerprint : Fingerprint object
                An object as a fingerprint class that convert atoms to fingerprint.
            reduce_dimensions: bool
                Whether to reduce the fingerprint space if constrains are used.
            use_derivatives : bool
                Whether to use derivatives/forces in the targets.
            use_fingerprint : bool
                Whether the kernel uses fingerprint objects (True) or arrays (False).
            npoints : int
                Number of points that are used from the database.
            initial_indicies : list
                The indicies of the data points that must be included in the used data base.
        """
        super().__init__(fingerprint=fingerprint,
                         reduce_dimensions=reduce_dimensions,
                         use_derivatives=use_derivatives,
                         use_fingerprint=use_fingerprint,
                         npoints=npoints,
                         initial_indicies=initial_indicies,
                         **kwargs)

    def make_reduction(self,all_indicies,**kwargs):
        " Reduce the training set with the points farthest from each other. "
        indicies=self.get_initial_indicies()
        if len(indicies)==0:
            indicies=np.array([np.random.choice(all_indicies)])
        features=self.get_all_feature_vectors()
        fdim=len(features[0])
        for i in range(len(indicies),self.npoints):
            not_indicies=[j for j in all_indicies if j not in indicies]
            dist=cdist(features[indicies].reshape(-1,fdim),features[not_indicies].reshape(-1,fdim))
            i_max=np.argmax(np.nanmin(dist,axis=0))
            indicies=np.append(indicies,[not_indicies[i_max]])
        return np.array(indicies,dtype=int)

    
class DatabaseRandom(Database_Reduction):
    def __init__(self,fingerprint=None,reduce_dimensions=True,use_derivatives=True,use_fingerprint=True,npoints=25,initial_indicies=[0],**kwargs):
        """ 
        Database of ASE atoms objects that are converted into fingerprints and targets. 
        The used Database is a reduced set of the full Database. 
        The reduced data set is selected from random. 

        Parameters:
            fingerprint : Fingerprint object
                An object as a fingerprint class that convert atoms to fingerprint.
            reduce_dimensions: bool
                Whether to reduce the fingerprint space if constrains are used.
            use_derivatives : bool
                Whether to use derivatives/forces in the targets.
            use_fingerprint : bool
                Whether the kernel uses fingerprint objects (True) or arrays (False).
            npoints : int
                Number of points that are used from the database.
            initial_indicies : list
                The indicies of the data points that must be included in the used data base.
        """
        super().__init__(fingerprint=fingerprint,
                         reduce_dimensions=reduce_dimensions,
                         use_derivatives=use_derivatives,
                         use_fingerprint=use_fingerprint,
                         npoints=npoints,
                         initial_indicies=initial_indicies,
                         **kwargs)

    def make_reduction(self,all_indicies,**kwargs):
        " Random select the training points. "
        indicies=self.get_initial_indicies()
        not_indicies=[j for j in all_indicies if j not in indicies]
        indicies=np.append(indicies,np.random.permutation(not_indicies)[:int(self.npoints-len(indicies))])
        return np.array(indicies,dtype=int)
    
class DatabaseHybrid(Database_Reduction):
    def __init__(self,fingerprint=None,reduce_dimensions=True,use_derivatives=True,use_fingerprint=True,npoints=25,initial_indicies=[0],**kwargs):
        """ 
        Database of ASE atoms objects that are converted into fingerprints and targets. 
        The used Database is a reduced set of the full Database. 
        The reduced data set is selected from a mix of the distances and random. 

        Parameters:
            fingerprint : Fingerprint object
                An object as a fingerprint class that convert atoms to fingerprint.
            reduce_dimensions: bool
                Whether to reduce the fingerprint space if constrains are used.
            use_derivatives : bool
                Whether to use derivatives/forces in the targets.
            use_fingerprint : bool
                Whether the kernel uses fingerprint objects (True) or arrays (False).
            npoints : int
                Number of points that are used from the database.
            initial_indicies : list
                The indicies of the data points that must be included in the used data base.
        """
        super().__init__(fingerprint=fingerprint,
                         reduce_dimensions=reduce_dimensions,
                         use_derivatives=use_derivatives,
                         use_fingerprint=use_fingerprint,
                         npoints=npoints,
                         initial_indicies=initial_indicies,
                         **kwargs)

    def make_reduction(self,all_indicies,**kwargs):
        " Use a combination of random sampling and farthest distance to reduce training set. "
        indicies=self.get_initial_indicies()
        if len(indicies)==0:
            indicies=[np.random.choice(all_indicies)]
        features=self.get_all_feature_vectors()
        fdim=len(features[0])
        for i in range(len(indicies),self.npoints):
            not_indicies=[j for j in all_indicies if j not in indicies]
            if i%3==0:
                indicies=np.append(indicies,[np.random.choice(not_indicies)])
            else:
                dist=cdist(features[indicies].reshape(-1,fdim),features[not_indicies].reshape(-1,fdim))
                i_max=np.argmax(np.nanmin(dist,axis=0))
                indicies=np.append(indicies,[not_indicies[i_max]])                
        return np.array(indicies,dtype=int)
    
    
class DatabaseMin(Database_Reduction):
    def __init__(self,fingerprint=None,reduce_dimensions=True,use_derivatives=True,use_fingerprint=True,npoints=25,initial_indicies=[0],**kwargs):
        """ 
        Database of ASE atoms objects that are converted into fingerprints and targets. 
        The used Database is a reduced set of the full Database. 
        The reduced data set is selected from the smallest targets. 

        Parameters:
            fingerprint : Fingerprint object
                An object as a fingerprint class that convert atoms to fingerprint.
            reduce_dimensions: bool
                Whether to reduce the fingerprint space if constrains are used.
            use_derivatives : bool
                Whether to use derivatives/forces in the targets.
            use_fingerprint : bool
                Whether the kernel uses fingerprint objects (True) or arrays (False).
            npoints : int
                Number of points that are used from the database.
            initial_indicies : list
                The indicies of the data points that must be included in the used data base.
        """
        super().__init__(fingerprint=fingerprint,
                         reduce_dimensions=reduce_dimensions,
                         use_derivatives=use_derivatives,
                         use_fingerprint=use_fingerprint,
                         npoints=npoints,
                         initial_indicies=initial_indicies,
                         **kwargs)

    def make_reduction(self,all_indicies,**kwargs):
        " Use the targets with smallest norms in the training set. "
        indicies=self.get_initial_indicies()
        not_indicies=np.array([j for j in all_indicies if j not in indicies])
        i_sort=np.argsort(np.linalg.norm(np.array(self.targets)[not_indicies],axis=1))
        indicies=np.append(indicies,not_indicies[i_sort[:int(self.npoints-len(indicies))]])
        return np.array(indicies,dtype=int)
    
class DatabaseLast(Database_Reduction):
    def __init__(self,fingerprint=None,reduce_dimensions=True,use_derivatives=True,use_fingerprint=True,npoints=25,initial_indicies=[0],**kwargs):
        """ 
        Database of ASE atoms objects that are converted into fingerprints and targets. 
        The used Database is a reduced set of the full Database. 
        The reduced data set is selected from the last data points. 

        Parameters:
            fingerprint : Fingerprint object
                An object as a fingerprint class that convert atoms to fingerprint.
            reduce_dimensions: bool
                Whether to reduce the fingerprint space if constrains are used.
            use_derivatives : bool
                Whether to use derivatives/forces in the targets.
            use_fingerprint : bool
                Whether the kernel uses fingerprint objects (True) or arrays (False).
            npoints : int
                Number of points that are used from the database.
            initial_indicies : list
                The indicies of the data points that must be included in the used data base.
        """
        super().__init__(fingerprint=fingerprint,
                         reduce_dimensions=reduce_dimensions,
                         use_derivatives=use_derivatives,
                         use_fingerprint=use_fingerprint,
                         npoints=npoints,
                         initial_indicies=initial_indicies,
                         **kwargs)

    def make_reduction(self,all_indicies,**kwargs):
        " Use the last data points. "
        indicies=self.get_initial_indicies()
        not_indicies=[j for j in all_indicies if j not in indicies]
        indicies=np.append(indicies,not_indicies[-int(self.npoints-len(indicies)):])
        return np.array(indicies,dtype=int)
    
class DatabaseRestart(Database_Reduction):
    def __init__(self,fingerprint=None,reduce_dimensions=True,use_derivatives=True,use_fingerprint=True,npoints=25,initial_indicies=[0],**kwargs):
        """ 
        Database of ASE atoms objects that are converted into fingerprints and targets. 
        The used Database is a reduced set of the full Database. 
        The reduced data set is selected from restarts after npoints are used.
        The initial indicies and the last data point is used at each restart.
        
        Parameters:
            fingerprint : Fingerprint object
                An object as a fingerprint class that convert atoms to fingerprint.
            reduce_dimensions: bool
                Whether to reduce the fingerprint space if constrains are used.
            use_derivatives : bool
                Whether to use derivatives/forces in the targets.
            use_fingerprint : bool
                Whether the kernel uses fingerprint objects (True) or arrays (False).
            npoints : int
                Number of points that are used from the database.
            initial_indicies : list
                The indicies of the data points that must be included in the used data base.
        """
        super().__init__(fingerprint=fingerprint,
                         reduce_dimensions=reduce_dimensions,
                         use_derivatives=use_derivatives,
                         use_fingerprint=use_fingerprint,
                         npoints=npoints,
                         initial_indicies=initial_indicies,
                         **kwargs)

    def make_reduction(self,all_indicies,**kwargs):
        " Make restart of used data set. "
        indicies=self.get_initial_indicies()
        # Get the data set size
        data_len=len(all_indicies)
        # Get the data point after the first restart
        n_use=data_len-self.npoints
        if n_use<=0:
            return all_indicies
        else:
            # Get the number of points that are not initial indicies
            nfree=self.npoints-len(indicies)
            # Get the excess of data points after each restart
            n_extra=int(n_use%nfree)
            if n_extra==0:
                # Last iteration before restart
                indicies=np.append(indicies,all_indicies[-nfree:])
            else:
                # Restarted indicies
                indicies=np.append(indicies,all_indicies[-n_extra:])
        return np.array(indicies,dtype=int)
