import numpy as np
from .orgneb import OriginalNEB

class ImprovedTangentNEB(OriginalNEB):
    def __init__(self,images,k=0.1,climb=False,remove_rotation_and_translation=False,**kwargs):
        super().__init__(images,k=k,climb=climb,remove_rotation_and_translation=remove_rotation_and_translation,**kwargs)
    
    def get_parallel_forces(self,tangent,pos_p,pos_m,**kwargs):
        forces_parallel=(self.k[1:]*np.linalg.norm(pos_p,axis=(1,2)))-(self.k[:-1]*np.linalg.norm(pos_m,axis=(1,2)))
        forces_parallel=forces_parallel.reshape(-1,1,1)*tangent
        return forces_parallel

    def get_tangent(self,pos_p,pos_m,**kwargs):
        tangent=np.empty((int(self.nimages-2),self.natoms,3))
        energies=self.get_energies()
        for i in range(1,self.nimages-1):
            if energies[i+1]>energies[i] and energies[i]>energies[i-1]:
                tangent[i-1]=pos_p[i-1]
            elif energies[i+1]<energies[i] and energies[i]<energies[i-1]:
                tangent[i-1]=pos_m[i-1]
            elif energies[i+1]>energies[i-1]:
                energy_dif=[abs(energies[i+1]-energies[i]),abs(energies[i-1]-energies[i])]
                tangent[i-1]=(pos_p[i-1]*max(energy_dif))+(pos_m[i-1]*min(energy_dif))
            elif energies[i+1]<energies[i-1]:
                energy_dif=[abs(energies[i+1]-energies[i]),abs(energies[i-1]-energies[i])]
                tangent[i-1]=(pos_p[i-1]*min(energy_dif))+(pos_m[i-1]*max(energy_dif))
            else:
                tangent[i-1]=(pos_p[i-1]/np.linalg.norm(pos_p[i-1]))+(pos_m[i-1]/np.linalg.norm(pos_m[i-1]))
        return tangent/(np.linalg.norm(tangent,axis=(1,2)).reshape(-1,1,1))
