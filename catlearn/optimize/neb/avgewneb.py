import numpy as np
from .ewneb import EWNEB

class AvgEWNEB(EWNEB):
    
    def get_parallel_forces(self,tangent,pos_p,pos_m,**kwargs):
        energies=self.get_energies()
        if self.use_minimum:
            e0=np.min([energies[0],energies[-1]])
        else:
            e0=np.max([energies[0],energies[-1]])
        emax=np.max(energies)
        k_l=self.k*self.kl_scale
        if e0<emax:
            a=(emax-energies)/(emax-e0)
            a=np.where(a<1.0,a,1.0)
            a=0.5*(a[1:]+a[:-1])
            k=((1.0-a)*self.k)+(a*k_l)
        else:
            k=k_l.copy()
        forces_parallel=(k[1:]*np.linalg.norm(pos_p,axis=(1,2)))-(k[:-1]*np.linalg.norm(pos_m,axis=(1,2)))
        forces_parallel=forces_parallel.reshape(-1,1,1)*tangent
        return forces_parallel
