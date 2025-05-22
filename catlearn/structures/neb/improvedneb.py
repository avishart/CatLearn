from numpy import einsum, empty, sqrt
from numpy.linalg import norm
from .orgneb import OriginalNEB


class ImprovedTangentNEB(OriginalNEB):
    """
    The improved tangent Nudged Elastic Band method implementation.
    The improved tangent method uses energy weighting to calculate the tangent.

    See:
        https://doi.org/10.1063/1.1323224
    """

    def get_parallel_forces(self, tangent, pos_p, pos_m, **kwargs):
        # Get the spring constants
        k = self.get_spring_constants()
        # Calculate the parallel forces
        forces_parallel = k[1:] * sqrt(einsum("ijk,ijk->i", pos_p, pos_p))
        forces_parallel -= k[:-1] * sqrt(einsum("ijk,ijk->i", pos_m, pos_m))
        forces_parallel = forces_parallel.reshape(-1, 1, 1) * tangent
        return forces_parallel

    def get_tangent(self, pos_p, pos_m, **kwargs):
        tangent = empty((int(self.nimages - 2), self.natoms, 3))
        energies = self.get_energies()
        for i in range(1, self.nimages - 1):
            if energies[i + 1] > energies[i] and energies[i] > energies[i - 1]:
                tangent[i - 1] = pos_p[i - 1]
            elif (
                energies[i + 1] < energies[i] and energies[i] < energies[i - 1]
            ):
                tangent[i - 1] = pos_m[i - 1]
            elif energies[i + 1] > energies[i - 1]:
                energy_dif = [
                    abs(energies[i + 1] - energies[i]),
                    abs(energies[i - 1] - energies[i]),
                ]
                tangent[i - 1] = pos_p[i - 1] * max(energy_dif)
                tangent[i - 1] += pos_m[i - 1] * min(energy_dif)
            elif energies[i + 1] < energies[i - 1]:
                energy_dif = [
                    abs(energies[i + 1] - energies[i]),
                    abs(energies[i - 1] - energies[i]),
                ]
                tangent[i - 1] = pos_p[i - 1] * min(energy_dif)
                tangent[i - 1] += pos_m[i - 1] * max(energy_dif)
            else:
                tangent[i - 1] = pos_p[i - 1] / norm(pos_p[i - 1])
                tangent[i - 1] += pos_m[i - 1] / norm(pos_m[i - 1])
        # Normalization of tangent
        tangent_norm = sqrt(einsum("ijk,ijk->i", tangent, tangent)).reshape(
            -1, 1, 1
        )
        tangent = tangent / tangent_norm
        return tangent
