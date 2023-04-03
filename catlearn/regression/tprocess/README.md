# Student t Process Source Code
The Student t process (TP) class is implemented to use different classes for prior means, kernels, fingerprints, and hyperparameter fitter. The TP class itself that can be trained and predict with uncertainties. The derivatives of the targets can be used by using the True bool for the use_derivatives argument in the initialization of the TP. Furthermore, a noise correction can be added to the covariance matrix to always make it invertible by a True bool for the correction argument. The hyperparameters are in ln values due to robustness. The noise hyperparameter is the relative-noise hyperparameter (or noise-to-signal), which corresponds to a replacement of the noise hyperparameter divided with the prefactor hyperparameter defined as a new free hyperparameter.
The TP class is imported from the gp module.  


## Baseline 
The baseline class used for the TP is implemeted in the Baseline module. 
The repulisive part of the Lennard-Jones potential as a baseline is implemeted as a baseline class. 

## Boundary
The boundary class used for constructing boundary conditions for the hyperparameters.

## Calculator 
The calculator module include the scripts needed for converging the TP to an ASE calculator. 
The scripts are:
- mlmodel: MLModel is a class that calculate energies, forces, and uncertainties for ASE Atoms.
- mlcalc: MLCalculator is an ASE calculator class that uses the MLModel as calculator.
- database: Database is class that collects the atomic structures with their energies, forces, and fingerprints.
- database_reduction: Database_Reduction is a Database class that reduces the number of training structures.

## Educated
The Educated_guess class make educated guesses of the MLE and boundary conditions of the hyperparameters.

## Fingerprint
The Fingerprint class convert ASE Atoms into a FingerprintObject class. The FingerprintObject contain a fingerprint vector and derivatives with respect to the Cartesian coordinates. The Fingerprint class has children of different fingerprints:
- cartesian: Cartesian coordinates of the ASE Atoms in the order of the Atoms index.
- coulomb: The Coulomb matrix fingerprint.
- fpwrapper: A wrapper of fingerprints from ASE-GPATOM to the FingerprintObject class.
- invdistances: The inverse distance of all element combinations, where the blocks of each combination is sorted with magnitude. The inverse distances is scaled with the sum of the elements covalent radii.  
- sumdistances: The summed inverse distances for each element combinations scaled with elements covalent radii. 
- sumdistancespower: The summed inverse distances for each element combinations scaled with elements covalent radii to different orders.

## Hpfitter
The hyperparameter fitter class that optimize the hyperparameters of the TP. The hyperparameter fitter needs an objective function and a optimization method as arguments. 
A fully-Bayesian mimicking TP can be achived by the fbpmgp class. 

## Hptrans
A variable transformation of the hyperparameters is performed with Variable_Transformation class. The region of interest in hyperparameter space is enlarged without restricting any value. 

## Kernel 
The kernel function is a fundamental part of the TP. The kernel function uses a distance meassure.
The Distance_matrix class construct a distance matrix of the features that can be used by the kernel function. The Distance_matrix_per_dimension class is used when derivatives of the targets are used, since the distances in each feature dimension needs to be saved.
A parent Kernel class is defined when only targets are used. The Kernel_Derivative class is used when derivatives of the targets are needed. 
An implemented kernel function is the squared exponential kernel (SE) class. 

## Means
In the means module different prior mean classes is defined. The prior mean is a key part of the TP. Constant value prior means classes is implemented as the parent Prior_constant class in constant submodule. The implemented children prior means classes are:
- first: Use the value of the first target.
- max: Use the value of the target with the largest value.
- mean: Use the mean value of the targets.
- median: Use the median value of the targets.
- min: Use the value of the target with the smallest value.  

## Objectfunctions
The parent Object_functions class give the form of the objective functions used for optimizing the hyperparameters. The implemented children objective function classes are:
- factorized_gpp: Calculate the minimum of the GPP objective function value over all relative-noise hyperparameter values for each length-scale hyperparameter. 
- factorized_likelihood_svd: Calculate the minimum of the negative log-likelihood objective function value over all relative-noise hyperparameter values for each length-scale hyperparameter. SVD is used for finding the singular values and therefore a noise correction is not needed and the inversion is robust. 
- factorized_likelihood: Calculate the minimum of the negative log-likelihood objective function value over all relative-noise hyperparameter values for each length-scale hyperparameter. 
- likelihood: The negative log-likelihood is calculated.

## Optimizers
Different optimizers can be used for optimizing the hyperparameters of the TP. The optimizers are split into local and global optimizers.

## Pdistributions
Prior distributions for the hyperparameters can be applied to the objective function. Thereby, the log-posterior is maximized instead of the log-likelihood. The prior distributions are important for the optimization of hyperparameters, since it gives prior knowledge about decent hyperparameters. The hyperparameter values are in log-scale.
The parent prior distribution class is Prior_distribution in pdistributions. The children classes are:
- gamma: The gamma distribution.
- gen_normal: The generalized normal distribution.
- invgamma: The inverse-gamma distribution.
- normal: The normal distribution.
- uniform: The uniform prior distribution within an interval.


