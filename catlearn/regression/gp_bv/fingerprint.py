import numpy as np

class Fingerprint:
    def __init__(self,reduce_dim=False):
        self.reduce_dim=reduce_dim

    def reduce(self,Data_F,train=True):
        '''Reduce the fingerprint dimension by removing unchanged coloumns
        Parameters:
            Data_F : (M,Df) array
                An array of the converted fingerprint for each Atoms systems
            train : bool
                True if the reduction is determined on Data_F 
                or False if it is determined on a previous Data_F
        Returns:
            Data_F : (M,Df-s) array
                An reducd 2D array, where unchanged columns (std=0) removed  
        '''
        if train:
            self.col_rm=np.where(~np.isclose(np.std(Data_F,axis=0),0))[0]
        try:
            return Data_F[:,self.col_rm].reshape(-1,len(self.col_rm))
        except:
            return Data_F

    def create(self,atoms_list,train=True):
        '''Convert a list of ASE Atoms into the fingerprint
        Parameters:
            atoms_list : M list
                A list of M ASE Atoms
            train : bool
                True if the reduction is determined on atoms_list 
                or False if it is determined on a previous atoms_list
        Returns:
            Data_F : (M,Df) array
                An array of the converted fingerprint for each Atoms systems
        '''
        Data_F=np.array([self.convert(atoms) for atoms in atoms_list])
        if self.reduce_dim:
            return self.reduce(Data_F,train=train)
        return Data_F

    def recreate(self,atoms_list,Data_F,gradient=False):
        '''Convert fingerprint back to atoms list
        Parameters:
            atoms_list : M list
                A list of M ASE Atoms
            Data_F : (M,Df) array
                An array of the converted fingerprint for each Atoms system
            gradient : bool
                True if Data_F is graidents of fingerprint else Fase
        Returns:
            Data_F : (M,Df) array
                An array of the coordinates for each Atoms systems or the gradient matrix
        '''
        return np.array([self.reverse(atoms,Data_F[a],gradient) for a,atoms in enumerate(atoms_list)])

    def convert(self,atoms):
        '''Convert ASE atoms into the cartesian fingerprint'''
        return atoms.get_positions().flatten()
    
    def gradient_check(self,grad_array):
        '''Check if the gradients can be reduced less than fingerprint 
           and give the reduced gradient matrix
        Parameters:
            Gradient matrix : (M,E*3)
                Gradient matrix with for each atoms system in all cordinates
        Returns:
            Reduced gradient matrix
            Bool if the Fingerprint matrix should be rereduced
        '''
        col_rm=np.copy(self.col_rm)
        grad_red=self.reduce(grad_array,train=True)
        try:
            if np.allclose(self.col_rm,col_rm):
                return grad_red,False
            self.col_rm=np.concatenate(self.col_rm,col_rm)
            grad_red=self.reduce(grad_array,train=False)
            return grad_red,True
        except:
            if len(self.col_rm)<len(col_rm):
                self.col_rm=np.copy(col_rm)
                grad_red=self.reduce(grad_array,train=False)
                return grad_red,False
            return grad_red,True
        



class Fingerprint_cartessian(Fingerprint):
    def convert(self,atoms):
        '''Convert ASE atoms into the cartesian fingerprint'''
        return atoms.get_positions().flatten()

    def reverse(self,atoms,Data_F,gradient):
        '''Convert fingerprint back to atoms'''
        atoms_f=self.convert(atoms)
        if gradient:
            atoms_f=np.zeros(atoms_f.shape)
        if self.reduce_dim:
            atoms_f[self.col_rm]=Data_F
        else:
            atoms_f=Data_F
        return atoms_f

class Fingerprint_coulombmatrix(Fingerprint):
    def convert(self,atoms):
        '''Convert ASE atoms into the Coulomb Matrix fingerprint from dScribe'''
        q = atoms.get_atomic_numbers()
        qiqj = q[None, :]*q[:, None]
        idmat = atoms.get_inverse_distance_matrix()
        np.fill_diagonal(idmat, 0)
        cmat = qiqj*idmat
        np.fill_diagonal(cmat, 0.5 * q ** 2.4)
        return cmat.flatten()
