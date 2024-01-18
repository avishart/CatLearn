import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.build import minimize_rotation_and_translation

class OriginalNEB:
    def __init__(self,images,k=0.1,climb=False,remove_rotation_and_translation=False,**kwargs):
        """
        The orginal Nudged Elastic Band method implementation for the tangent and parallel force. 

        Parameters:
            images : List of ASE Atoms instances
                The ASE Atoms instances used as the images of the initial path that is optimized.
            k : List of floats or float
                The spring force acting between each image. 
            climb : bool
                Whether to use climbing image in the NEB. 
            remove_rotation_and_translation : bool
                Whether to remove rotation and translation in interpolation and when predicting forces.
        """
        self.images=images
        self.nimages=len(images)
        self.natoms=len(images[0])
        if isinstance(k,(int,float)):
            self.k=np.full(self.nimages-2,k)
        else:
            self.k=k.copy()
        self.climb=climb
        self.remove_rotation_and_translation=remove_rotation_and_translation

    def interpolate(self,method='linear',mic=True,**kwargs):
        """
        Make an interpolation between the start and end structure.

        Parameters:
            method : str
                The method used for performing the interpolation.
                The optional methods is {linear, idpp, ends}.
            mic : bool
                Whether to use the minimum-image convention. 

        Returns:
            self: The instance itself.
        """
        from ..interpolate_band import interpolate
        self.images=interpolate(self.images[0].copy(),self.images[-1].copy(),n_images=self.nimages,method=method,mic=mic,remove_rotation_and_translation=self.remove_rotation_and_translation,**kwargs)
        return self

    def get_positions(self):
        """
        Get the positions of all the moving images in one array.

        Returns:
            ((Nimg-2)*Natoms,3) array: Coordinates of all atoms in all the moving images.
        """
        return np.array([image.get_positions() for image in self.images[1:-1]]).reshape(-1,3)
    
    def set_positions(self,positions,**kwargs):
        """
        Set the positions of all the images in one array.

        Parameters:
            positions : ((Nimg-2)*Natoms,3) array
                Coordinates of all atoms in all the moving images.
        """
        for i,image in enumerate(self.images[1:-1]):
            image.set_positions(positions[i*self.natoms:(i+1)*self.natoms])
        pass

    def get_potential_energy(self,**kwargs):
        """ 
        Get the potential energy of the NEB as the sum of energies. 
        
        Returns:
            float: Sum of energies of moving images.
        """
        return np.sum(self.get_energies(**kwargs)[1:-1])
    
    def get_forces(self,**kwargs):
        """ 
        Get the forces of the NEB as the stacked forces of the moving images. 
        
        Returns:
            ((Nimg-2)*Natoms,3) array: Forces of all the atoms in all the moving images.
        """
        # Remove rotation and translation
        if self.remove_rotation_and_translation:
            for i in range(1,self.nimages):
                minimize_rotation_and_translation(self.images[i-1],self.images[i])
        # Get the forces for each image
        forces=self.calculate_forces()
        # Get the change in the coordinates relative to the previous and later image
        position_plus,position_minus=self.get_position_diff()
        # Calculate the tangent to the moving images
        tangent=self.get_tangent(position_plus,position_minus)
        # Calculate the parallel forces between images
        parallel_forces=self.get_parallel_forces(tangent,position_plus,position_minus)
        # Calculate the perpendicular forces
        perpendicular_forces=self.get_perpendicular_forces(tangent,forces)
        # Calculate the full force
        forces_new=parallel_forces+perpendicular_forces
        # Calculate the force of the climbing image
        if self.climb:
            i_max=np.argmax(self.get_energies()[1:-1])
            forces_new[i_max]=forces[i_max]-((2.0*np.vdot(forces[i_max],tangent[i_max]))*tangent[i_max])
        return forces_new.reshape(-1,3)
    
    def get_image_positions(self):
        """
        Get the positions of the images.

        Returns:
            ((Nimg),Natoms,3) array: The positions for all atoms in all the images.
        """
        return np.array([image.get_positions() for image in self.images])
    
    def calculate_forces(self,**kwargs):
        " Calculate the forces for all the images separately. "
        forces=np.array([image.get_forces() for image in self.images[1:-1]])
        self.real_forces=np.zeros((self.nimages,self.natoms,3))
        self.real_forces[1:-1]=forces.copy()
        return forces
    
    def get_energies(self,**kwargs):
        " Get the individual energy for each image. "
        self.energies=np.array([image.get_potential_energy(**kwargs) for image in self.images])
        return self.energies
    
    def emax(self,**kwargs):
        " Get maximum energy of the moving images. "
        return np.nanmax(self.get_energies(**kwargs)[1:-1])
    
    def get_parallel_forces(self,tangent,pos_p,pos_m,**kwargs):
        " Get the parallel forces between the images. "
        forces_parallel=np.linalg.norm(pos_p,axis=(1,2))-np.linalg.norm(pos_m,axis=(1,2))
        forces_parallel=(self.k*forces_parallel).reshape(-1,1,1)*tangent
        return forces_parallel
    
    def get_perpendicular_forces(self,tangent,forces,**kwargs):
        " Get the perpendicular forces to the images. "
        return forces-(np.sum(forces*tangent,axis=(1,2)).reshape(-1,1,1)*tangent)
    
    def get_position_diff(self):
        " Get the change in the coordinates relative to the previous and later image. "
        positions=self.get_image_positions()
        position_plus=positions[2:]-positions[1:-1]
        position_minus=positions[1:-1]-positions[:-2]
        return position_plus,position_minus
    
    def get_tangent(self,pos_p,pos_m,**kwargs):
        " Calculate the tangent to the moving images. "
        # Normalization
        tangent_m=pos_m/(np.linalg.norm(pos_m,axis=(1,2)).reshape(-1,1,1))
        tangent_p=pos_p/(np.linalg.norm(pos_p,axis=(1,2)).reshape(-1,1,1))
        # Sum them
        tangent=tangent_m+tangent_p
        return tangent/(np.linalg.norm(tangent,axis=(1,2)).reshape(-1,1,1))
    
    def __len__(self):
        return int(self.nimages-2)*self.natoms
    
    def freeze_results_on_image(self,atoms,**results_to_include):
        atoms.calc=SinglePointCalculator(atoms=atoms,**results_to_include)
        return atoms
    
    def iterimages(self):
        # Allows trajectory to convert NEB into several images
        for i,atoms in enumerate(self.images):
            if i==0 or i==self.nimages-1:
                yield atoms
            else:
                atoms=atoms.copy()
                self.freeze_results_on_image(atoms,energy=self.energies[i],forces=self.real_forces[i])
                yield atoms
