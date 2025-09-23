# Gaussian Process Source Code
The Gaussian process class and the Student T process are implemented to use different classes for prior means, kernels, fingerprints, and hyperparameter fitter. The Gaussian process class itself can be trained and used to predict with uncertainties. The derivatives of the targets can be used by using the True bool for the `use_derivatives` argument in the initialization of the Gaussian process. Furthermore, a noise correction can be added to the covariance matrix to always make it invertible by a True bool for the `use_correction` argument. The hyperparameters are in natural log-scale to enforce robustness. The noise hyperparameter is the relative-noise hyperparameter (or noise-to-signal), which corresponds to a replacement of the noise hyperparameter divided by the prefactor hyperparameter, defined as a new free hyperparameter.
The Gaussian process class and the Student T process are imported from the models module.  


## Baseline 
The baseline class used for the Gaussian process is implemented in the `baseline` module. 
The repulsive part of the Lennard-Jones potential is implemented as the `RepulsionCalculator` class. 
The born repulsion with a cutoff at a scaled sum of covalent radii is implemented as the `BornRepulsionCalculator` class.
A Mie potential is also implemented as the `MieCalculator`class.

## HPBoundary
### Boundary conditions
The boundary classes are used for constructing boundary conditions for the hyperparameters in the `hpboundary` module. 
### Hptrans
A variable transformation of the hyperparameters is performed with the `VariableTransformation` class. The region of interest in hyperparameter space is enlarged without restricting any value.

## Calculator 
The calculator module includes the scripts needed for converting the Gaussian process to an ASE calculator. 
The modules are:
- `MLModel` is a class that calculates energies, forces, and uncertainties for ASE Atoms.
- `MLCalculator` is an ASE calculator class that uses the MLModel as a calculator.
- `BOCalculator` is like the `MLCalculator`, but the calculated potential energy is added together with the uncertainty times kappa.
- `Database` is a class that collects the atomic structures with their energies, forces, and fingerprints.
- `DatabaseReduction` is a `Database` class that reduces the number of training structures.

## Ensemble
The ensemble model uses multiple models that are trained and make an ensemble of their predictions in different ways. One way is the mean of the prediction, and another is the variance-weighted prediction. The EnsembleClustering class uses one of the clustering algorithms from the clustering module to split the training data for the different machine learning models.
The clustering algorithms are:
- `K_means` is the K-means++ clustering method that uses the distances to the defined centroids. Each datapoint is assigned to each cluster. A training point can only be included in one cluster. 
- `K_means_number`: This method uses distances similar to the K-means++ clustering method. However, the clusters are of the same size, and the number of clusters is defined by the number of training points. A training point can be included in multiple clusters if the data points can not be split equally. 
- `K_means_auto` is similar to `K_means_number`, but it uses a range of the number of training points that the clusters include. A training point can be included in multiple clusters. 
- `K_means_enumeration` uses the training point in the order it is given.
- `FixedClustering` uses predefined centroids where the data points are assigned to each cluster. A training point can only be included in one cluster. 
- `RandomClustering` randomly places training points in the given number of clusters.
- `RandomClustering_number` randomly places training points in clusters so that they match the number of points in each cluster as requested.

## Fingerprint
The Fingerprint class converts ASE Atoms into a `FingerprintObject` instance. The `FingerprintObject` contains a fingerprint vector and derivatives with respect to the Cartesian coordinates. The Fingerprint class has child classes with different fingerprints, which are:
- `Cartesian` is the Cartesian coordinates of the ASE Atoms in the order of the atom index.
- `InvDistances` is the inverse distance of all atom combinations. The inverse distances are scaled with the sum of the elements' covalent radii.
- `SortedInvDistances` is the inverse distances of all element combinations where the blocks of each combination are sorted by magnitude.   
- `SumDistances` is the summed inverse distances for each element combination, scaled with the sum of the elements' covalent radii. 
- `SumDistancesPower` is the summed inverse distances for each element combination, scaled with the sum of the elements' covalent radii to different orders.
- `MeanDistances` is the mean inverse distances for each element combination, scaled with the sum of the elements' covalent radii. 
- `MeanDistancesPower` is the mean inverse distances for each element combination, scaled with the sum of the elements' covalent radii to different orders.
- `FingerprintWrapperGPAtom` is a wrapper of fingerprints from ASE-GPATOM to the `FingerprintObject` instance.
- `FingerprintWrapperDScribe` is a wrapper of fingerprints from DScribe to the `FingerprintObject` instance.

## Hpfitter
The hyperparameter fitter class that optimizes the hyperparameters of the Gaussian process. The hyperparameter fitter needs an objective function and an optimization method as arguments. 
A fully Bayesian mimicking Gaussian process can be achieved by the `FBPMGP` class.  

## Kernel 
The kernel function is a fundamental part of the Gaussian process. Derivatives of the kernel function can be used by setting the `use_derivatives` argument. 
An implemented kernel function is the squared exponential kernel (SE) class. 

## Means
In the means module, different prior mean classes are defined. The prior mean is a key part of the Gaussian process. Constant value prior means classes are implemented as the parent `Prior_constant` class in the constant submodule. The implemented child prior means classes are:
- `Prior_first` uses the value of the first target.
- `Prior_max` uses the value of the target with the largest value.
- `Prior_mean` uses the mean value of the targets.
- `Prior_median` uses the median value of the targets.
- `Prior_min` uses the value of the target with the smallest value.  

## Models
The Gaussian process and the Student t process are imported from this module. 

## Objectivefunctions
The parent `ObjectiveFuction` class gives the form of the objective functions used for optimizing the hyperparameters. The implemented child objective function classes are split into Gaussian process and Student t process objective functions. 
### GP
The Gaussian process objective functions are:
- `LogLikelihood` is the negative log-likelihood.
- `MaximumLogLikelihood` is the negative maximum log-likelihood calculated by using an analytical expression of the prefactor hyperparameter.
- `FactorizedGPP` calculates the minimum of the GPP objective function value over all relative-noise hyperparameter values for each length-scale hyperparameter. The prefactor hyperparameter is determined analytically.
- `FactorizedLogLikelihood` calculates the minimum of the negative log-likelihood objective function value over all relative-noise hyperparameter values for each length-scale hyperparameter. The prefactor hyperparameter is determined analytically by maximum-likelihood estimation.  
- `GPE` is Geisser's predictive mean square error objective function.
- `GPP` is Geisser's surrogate predictive probability objective function. 
- `LOO` is the leave-one-out cross-validation from a single covariance matrix inversion is calculated. A modification can also be used to get good values for the prefactor hyperparameter. 
### TP
- `LogLikelihood` is the negative log-likelihood.
- `FactorizedLogLikelihood` calculates the minimum of the negative log-likelihood objective function value over all relative-noise hyperparameter values for each length-scale hyperparameter. 

## Optimizers
Different optimizers can be used for optimizing the hyperparameters of the Gaussian process. The optimizers are split into local, global, line search, and noise line search optimizers.

## Pdistributions
Prior distributions for the hyperparameters can be applied to the objective function. Thereby, the log-posterior is maximized instead of the log-likelihood. The prior distributions are important for the optimization of hyperparameters, since they give prior knowledge about decent hyperparameters. The hyperparameter values are on a natural log-scale.
The parent prior distribution class is the `Prior_distribution` in the `pdistributions`module. The child classes are:
- `Gamma_prior` is the gamma distribution.
- `Gen_normal_prior` is the generalized normal distribution.
- `Invgamma_prior` is the inverse-gamma distribution.
- `Normal_prior` is the normal distribution.
- `Uniform_prior` is the uniform prior distribution within an interval.

