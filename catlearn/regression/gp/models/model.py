from numpy import (
    array,
    asarray,
    diag,
    einsum,
    empty,
    exp,
    finfo,
    inf,
    matmul,
    nan_to_num,
)
from scipy.linalg import cho_factor, cho_solve
import pickle
from ..means.mean import Prior_mean
from ..kernel import SE
from ..hpfitter import HyperparameterFitter
from ..objectivefunctions.gp.likelihood import LogLikelihood


class ModelProcess:
    """
    The Model Process Regressor.
    The Model process uses Cholesky decomposition for
    inverting the kernel matrix.
    The hyperparameters can be optimized.
    """

    def __init__(
        self,
        prior=Prior_mean(),
        kernel=SE(use_derivatives=False, use_fingerprint=False),
        hpfitter=HyperparameterFitter(func=LogLikelihood()),
        hp={},
        use_derivatives=False,
        use_correction=True,
        dtype=float,
        **kwargs,
    ):
        """
        Initialize the Model Process Regressor.

        Parameters:
            prior: Prior class
                The prior mean given for the data.
            kernel: Kernel class
                The kernel function used for the kernel matrix.
            hpfitter: HyperparameterFitter class
                A class to optimize hyperparameters
            hp: dictionary
                A dictionary of hyperparameters like noise and length scale.
                The hyperparameters are used in the log-space.
            use_derivatives: bool
                Use derivatives/gradients of the targets for
                training and predictions.
            use_correction: bool
                Use the noise correction on the covariance matrix.
            dtype: type
                The data type of the arrays.
        """
        # Set default descriptors
        self.trained_model = False
        self.corr = 0.0
        self.features = []
        self.L = empty(0, dtype=dtype)
        self.low = False
        self.coef = empty(0, dtype=dtype)
        self.prefactor = 1.0
        # Set default relative-noise hyperparameter
        self.hp = {"noise": asarray([-8.0], dtype=dtype)}
        # Set all the arguments
        self.update_arguments(
            prior=prior,
            kernel=kernel,
            hpfitter=hpfitter,
            hp=hp,
            use_derivatives=use_derivatives,
            use_correction=use_correction,
            dtype=dtype,
            **kwargs,
        )

    def train(self, features, targets, **kwargs):
        """
        Train the model with training features and targets.

        Parameters:
            features: (N,D) array or (N) list of fingerprint objects
                Training features with N data points.
            targets: (N,1) array or (N,1+D) array
                Training targets with N data points.
                If use_derivatives=True, the training targets is in
                first column and derivatives is in the next columns.

        Returns:
            self: The trained object itself.
        """
        # Note that the model is trained
        self.trained_model = True
        # Store features
        self.features = features.copy()
        # Make the kernel matrix decomposition
        self.L, self.low = self.calculate_kernel_decomposition(features)
        # Modify the targets with the prior mean and rearrangement
        targets_mod = self.modify_targets(features, targets)
        # Calculate the coefficients
        self.coef = self.calculate_coefficients(features, targets_mod)
        # Calculate the prefactor for variance predictions
        self.prefactor = self.calculate_prefactor(features, targets_mod)
        return self

    def optimize(
        self,
        features,
        targets,
        retrain=True,
        hp=None,
        pdis=None,
        verbose=False,
        **kwargs,
    ):
        """
        Optimize the hyperparameter of the model and its kernel.

        Parameters:
            features: (N,D) array or (N) list of fingerprint objects
                Training features with N data points.
            targets: (N,1) array or (N,D+1) array
                Training targets with or without derivatives
                with N data points.
            retrain: bool
                Whether to retrain the model after the optimization.
            hp: dict
                Use a set of hyperparameters to optimize from
                else the current set is used.
                The hyperparameters are used in the log-space.
            maxiter: int
                Maximum number of iterations used by local or
                global optimization method.
            pdis: dict
                A dict of prior distributions for each hyperparameter type.
            verbose: bool
                Print the optimized hyperparameters and
                the object function value.

        Returns:
            dict: A solution dictionary with objective function value and
                hyperparameters.
        """
        # Ensure the targets are in the right format
        if not self.use_derivatives:
            targets = array(targets[:, 0:1], dtype=self.dtype)
        # Optimize the hyperparameters
        sol = self.hpfitter.fit(
            features,
            targets,
            model=self,
            hp=hp,
            pdis=pdis,
            retrain=retrain,
            **kwargs,
        )
        # Print the solution
        if verbose:
            print(sol)
        # Retrain the model with the new hyperparameters
        if retrain:
            if "prior" in sol.keys():
                self.prior.update_arguments(sol["prior"])
            self.set_hyperparams(sol["hp"])
            self.train(features, targets)
        return sol

    def predict(
        self,
        features,
        get_derivatives=False,
        get_variance=False,
        include_noise=False,
        get_derivtives_var=False,
        get_var_derivatives=False,
        **kwargs,
    ):
        """
        Predict the mean and variance for test features by using data and
        coefficients from training data.

        Parameters:
            features: (M,D) array or (M) list of fingerprint objects
                Test features with M data points.
            get_derivatives: bool
                Whether to predict the derivatives of the prediction mean.
            get_variance: bool
                Whether to predict the variance of the targets.
            include_noise: bool
                Whether to include the noise of data in the predicted variance.
            get_derivtives_var: bool
                Whether to predict the variance of the derivatives
                of the targets.
            get_var_derivatives: bool
                Whether to calculate the derivatives of the predicted variance
                of the targets.

        Returns:
            Y_predict: (M,1) or (M,1+D) array
                The predicted mean values with or without derivatives.
            var: (M,1) or (M,1+D) array
                The predicted variance of the targets with or
                without derivatives.
            var_deriv: (M,D) array
                The derivatives of the predicted variance of the targets.
        """
        # Check if the model is trained
        if not self.trained_model:
            raise AttributeError("The model is not trained!")
        # Calculate the kernel matrix of test and training data
        if (
            get_derivatives
            or (get_derivtives_var and get_variance)
            or get_var_derivatives
        ):
            KQX = self.get_kernel(
                features,
                self.features,
                get_derivatives=True,
            )
        else:
            KQX = self.get_kernel(
                features,
                self.features,
                get_derivatives=False,
            )
        # Calculate the prediction mean
        Y_predict = self.predict_mean(
            features,
            KQX=KQX,
            get_derivatives=get_derivatives,
        )
        # Calculate the predicted variance
        if get_variance:
            var = self.predict_variance(
                features,
                KQX=KQX,
                get_derivatives=get_derivtives_var,
                include_noise=include_noise,
            )
        else:
            var = None
        # Calculate the derivatives of the predicted variance
        if get_var_derivatives:
            var_deriv = self.calculate_variance_derivatives(features, KQX=KQX)
        else:
            var_deriv = None
        return Y_predict, var, var_deriv

    def predict_mean(
        self,
        features,
        KQX=None,
        get_derivatives=False,
        **kwargs,
    ):
        """
        Predict the mean for test features by using data and
        coefficients from training data.

        Parameters:
            features: (M,D) array or (M) list of fingerprint objects
                Test features with M data points.
            KQX: (M,N) or (M,N+N*D) or (M+M*D,N+N*D) array
                The kernel matrix of the test and training features.
                If KQX=None, it is calculated.
            get_derivatives: bool
                Whether to predict the derivatives of the prediction mean.

        Returns:
            Y_predict: (M,1) array
                The predicted mean values if get_derivatives=False.
            or
            Y_predict: (M,1+D) array
                The predicted mean values and its derivatives
                if get_derivatives=True.
        """
        # Check if the model is trained
        if not self.trained_model:
            raise AttributeError("The model is not trained!")
        # Get the number of test points
        m_data = len(features)
        # Calculate the kernel of test and training data if it is not given
        if KQX is None:
            KQX = self.get_kernel(
                features,
                self.features,
                get_derivatives=get_derivatives,
            )
        else:
            if not get_derivatives:
                KQX = KQX[:m_data]
        # Calculate the prediction mean
        Y_predict = matmul(KQX, self.coef)
        # Rearrange prediction
        Y_predict = Y_predict.reshape(m_data, -1, order="F")
        # Add the prior mean
        Y_predict += self.get_priormean(
            features,
            Y_predict,
            get_derivatives=get_derivatives,
        )
        return Y_predict

    def predict_variance(
        self,
        features,
        KQX=None,
        get_derivatives=False,
        include_noise=False,
        **kwargs,
    ):
        """
        Calculate the predicted variance of the test targets.

        Parameters:
            features: (M,D) array or (M) list of fingerprint objects
                Test features with M data points.
            KQX: (M,N) or (M,N+N*D) or (M+M*D,N+N*D) array
                The kernel matrix of the test and training features.
                If KQX=None, it is calculated.
            get_derivatives: bool
                Whether to predict the uncertainty of the derivatives of
                the targets.
            include_noise: bool
                Whether to include the noise of data in the predicted variance

        Returns:
            var: (M,1) array
                The predicted variance of the targets if get_derivatives=False.
            or
            var: (M,1+D) array
                The predicted variance of the targets and its derivatives
                if get_derivatives=True.

        """
        # Check if the model is trained
        if not self.trained_model:
            raise AttributeError("The model is not trained!")
        # Get the number of test points
        m_data = len(features)
        # Calculate the kernel of test and training data if it is not given
        if KQX is None:
            KQX = self.get_kernel(
                features,
                self.features,
                get_derivatives=get_derivatives,
            )
        else:
            if not get_derivatives:
                KQX = KQX[:m_data]
        # Calculate the diagonal elements of the kernel matrix of the test data
        k = self.kernel_diag(
            features,
            m_data=m_data,
            get_derivatives=get_derivatives,
            include_noise=include_noise,
        )
        # Calculate predicted variance
        var = (
            k - einsum("ij,ji->i", KQX, self.calculate_CinvKQX(KQX))
        ).reshape(-1, 1)
        # Scale prediction variance with the prefactor
        var = var * self.prefactor
        # Rearrange the predicted variance
        return var.reshape(m_data, -1, order="F")

    def calculate_variance_derivatives(self, features, KQX=None, **kwargs):
        """
        Calculate the derivatives of the predicted variance of
        the test targets.

        Parameters:
            features: (M,D) array or (M) list of fingerprint objects
                Test features with M data points.
            KQX: (M,N) or (M,N+N*D) or (M+M*D,N+N*D) array
                The kernel matrix of the test and training features.
                If KQX=None, it is calculated.

        Returns:
            var_deriv: (M,D) array
                The derivatives of the predicted variance of the targets.
        """
        # Check if the model is trained
        if not self.trained_model:
            raise AttributeError("The model is not trained!")
        # Get the number of test points
        m_data = len(features)
        # Calculate the kernel matrix of test and training data
        if KQX is None:
            KQX = self.get_kernel(
                features,
                self.features,
                get_derivatives=True,
            )
        # Calculate derivative of the diagonal wrt. the test features
        k_deriv = self.kernel_deriv_diag(features)
        # Calculate derivative of the predicted variance
        var_deriv = k_deriv - 2.0 * einsum(
            "ij,ji->i",
            KQX[m_data:],
            self.calculate_CinvKQX(KQX[:m_data]),
        ).reshape(-1, 1)
        # Scale prediction variance with the prefactor
        var_deriv = var_deriv * self.prefactor
        # Rearrange derivative of variance
        return var_deriv.reshape(m_data, -1, order="F")

    def predict_covariance(
        self,
        features,
        KQX=None,
        get_derivatives=False,
        include_noise=False,
        **kwargs,
    ):
        """
        Calculate the predicted covariance matrix of the test targets.

        Parameters:
            features: (M,D) array or (M) list of fingerprint objects
                Test features with M data points.
            KQX: (M,N) or (M,N+N*D) or (M+M*D,N+N*D) array
                The kernel matrix of the test and training features.
                If KQX=None, it is calculated.
            get_derivatives: bool
                Whether to predict the uncertainty of the derivatives of
                the targets.
            include_noise: bool
                Whether to include the noise of data in the predicted variance

        Returns:
            var: (M,M) array
                The predicted covariance matrix of the targets
                if get_derivatives=False.
            or
            var: (M*(1+D),M*(1+D)) array
                The predicted covariance matrix of the targets
                and its derivatives if get_derivatives=True.

        """
        # Check if the model is trained
        if not self.trained_model:
            raise AttributeError("The model is not trained!")
        # Get the number of test points
        n_data = len(features)
        # Calculate the kernel of test and training data if it is not given
        if KQX is None:
            KQX = self.get_kernel(
                features,
                self.features,
                get_derivatives=get_derivatives,
            )
        else:
            if not get_derivatives:
                KQX = KQX[:n_data]
        # Calculate the kernel matrix of the test data
        KQQ = self.get_kernel(
            features,
            get_derivatives=get_derivatives,
        )
        # Add noise to the diagonal of the kernel matrix
        if include_noise:
            add_v = self.inf_to_num(exp(2.0 * self.hp["noise"][0])) + self.corr
            m_data = len(KQQ)
            if "noise_deriv" in self.hp:
                KQQ[range(n_data), range(n_data)] += add_v
                add_v = self.inf_to_num(exp(2.0 * self.hp["noise_deriv"][0]))
                add_v += self.corr
                KQQ[range(n_data, m_data), range(n_data, m_data)] += add_v
            else:
                KQQ[range(m_data), range(m_data)] += add_v
        # Calculate predicted variance
        var = KQQ - matmul(KQX, self.calculate_CinvKQX(KQX))
        # Scale prediction variance with the prefactor
        var = var * self.prefactor
        # Return the predicted covariance matrix
        return var

    def set_hyperparams(self, new_params, **kwargs):
        """
        Set or update the hyperparameters for the model.

        Parameters:
            new_params: dictionary
                A dictionary of hyperparameters that are added or updated.
                The hyperparameters are used in the log-space.

        Returns:
            self: The object itself with the new hyperparameters.
        """
        # Set the hyperparameters in the kernel
        self.kernel.set_hyperparams(new_params)
        # Set the relative-noise hyperparameter
        if "noise" in new_params:
            self.hp["noise"] = array(
                new_params["noise"],
                dtype=self.dtype,
            ).reshape(-1)
        if "noise_deriv" in new_params:
            self.hp["noise_deriv"] = array(
                new_params["noise_deriv"],
                dtype=self.dtype,
            ).reshape(-1)
        return self

    def get_hyperparams(self, **kwargs):
        """
        Get the hyperparameters for the model and the kernel.

        Returns:
            dict: The hyperparameters in the log-space from
                the model and kernel class.
        """
        hp = {para: value.copy() for para, value in self.hp.items()}
        hp.update(self.kernel.get_hyperparams())
        return hp

    def get_kernel(
        self,
        features,
        features2=None,
        get_derivatives=True,
        **kwargs,
    ):
        """
        Make the kernel matrix.

        Parameters:
            features: (N,D) array or (N) list of fingerprint objects
                Features with N data points.
            features2: (M,D) array or (M) list of fingerprint objects
                Features with M data points and D dimensions.
                If it is not given a squared kernel from features is generated.
            get_derivatives: bool
                Whether to predict derivatives of target.

        Returns:
            KXX: array
                The symmetric kernel matrix if features2=None.
                The number of rows in the array is N, or N*(D+1)
                if get_derivatives=True.
                The number of columns in the array is N, or N*(D+1)
                if use_derivatives=True.
            or
            KQX: array
                The kernel matrix if features2 is not None.
                The number of rows in the array is N, or N*(D+1)
                if get_derivatives=True.
                The number of columns in the array is M, or M*(D+1)
                if use_derivatives=True.
        """
        return self.kernel(
            features,
            features2=features2,
            get_derivatives=get_derivatives,
            **kwargs,
        )

    def get_prefactor(self):
        """
        Get the prefactor that the prediction uncertainty is scaled with.

        Returns:
            float: The scaling of the prediction uncertainty.
        """
        return self.prefactor

    def update_priormean(self, features, targets, **kwargs):
        """
        Update the prior mean with the data.

        Parameters:
            features: (N,D) array or (N) list of fingerprint objects
                Training features with N data points.
            targets: (N,1) array or (N,1+D) array
                Training targets with N data points.
                If use_derivatives=True, the training targets is in
                first column and derivatives is in the next columns.

        Returns:
            self: The updated instance itself.
        """
        self.prior.update(features, targets, **kwargs)
        return self

    def get_priormean(
        self,
        features,
        targets,
        get_derivatives=False,
        **kwargs,
    ):
        """
        Get the prior mean for the given features.

        Parameters:
            features: (N,D) array or (N) list of fingerprint objects
                Features with N data points.
            targets: (N,1) array or (N,1+D) array
                Targets with N data points.
                If get_derivatives=True, the targets is in first column and
                derivatives is in the next columns.
            get_derivatives: bool
                Whether to give the prior mean of the derivatives of targets.

        Returns:
            (N,1) array or (N,1+D) array: The prior mean.
        """
        return self.prior.get(
            features,
            targets,
            get_derivatives=get_derivatives,
            **kwargs,
        )

    def get_prior_parameters(self, **kwargs):
        """
        Get the prior mean parameters.

        Returns:
            dict: A dictionary with the parameters used in the prior mean.
        """
        return self.prior.get_parameters(**kwargs)

    def get_gradients(self, features, hp, KXX, **kwargs):
        """
        Get the gradients of the covariance matrix with noise
        wrt.the hyperparameters.

        Parameters:
            features: (N,D) array
                Features with N data points and D dimensions.
            hp: list
                A list with elements of the hyperparameters that are optimized.
            KXX: (N,N) array
                The kernel matrix of training data.

        Returns:
            dict: A dictionary with gradients of the covariance matrix with
                noise wrt. the hyperparameters.
        """
        raise NotImplementedError()

    def get_use_derivatives(self):
        "Get whether the derivatives of the targets are used."
        return self.use_derivatives

    def get_use_fingerprint(self):
        "Get whether a fingerprint is used as the features."
        return self.kernel.get_use_fingerprint()

    def set_dtype(self, dtype, **kwargs):
        """
        Set the data type of the arrays.

        Parameters:
            dtype: type
                The data type of the arrays.

        Returns:
            self: The updated object itself.
        """
        # Set the data type
        self.dtype = dtype
        # Set the machine precision
        self.eps = 1.1 * finfo(self.dtype).eps
        # Set the data type of the attributes
        self.prior.set_dtype(dtype=dtype, **kwargs)
        self.kernel.set_dtype(dtype=dtype, **kwargs)
        self.hpfitter.set_dtype(dtype=dtype, **kwargs)
        # Set the data type of the hyperparameters
        self.set_hyperparams(self.hp)
        return self

    def set_seed(self, seed, **kwargs):
        """
        Set the random seed.

        Parameters:
            seed: int (optional)
                The random seed.
                The seed can be an integer, RandomState, or Generator instance.
                If not given, the default random number generator is used.

        Returns:
            self: The instance itself.
        """
        self.hpfitter.set_seed(seed)
        return self

    def set_use_derivatives(self, use_derivatives, **kwargs):
        """
        Set whether to use derivatives/gradients for training and predictions.

        Parameters:
            use_derivatives: bool
                Use derivatives/gradients for training and predictions.

        Returns:
            self: The updated object itself.
        """
        # Set whether to use derivatives for the target
        self.use_derivatives = use_derivatives
        # Set whether to use derivatives for the kernel
        self.kernel.set_use_derivatives(use_derivatives)
        return self

    def set_use_fingerprint(self, use_fingerprint, **kwargs):
        """
        Set whether to use a fingerprint as the features.

        Parameters:
            use_fingerprint: bool
                Use a fingerprint as the features.

        Returns:
            self: The updated object itself.
        """
        # Set whether to use a fingerprint for the features
        self.kernel.set_use_fingerprint(use_fingerprint)
        return self

    def save_model(self, filename="model.pkl", **kwargs):
        """
        Save the model object to a file.

        Parameters:
            filename: str
                The name of the file where the object is saved.

        Returns:
            self: The object itself.
        """
        with open(filename, "wb") as file:
            pickle.dump(self, file)
        return self

    def load_model(self, filename="model.pkl", **kwargs):
        """
        Load the model object from a file.

        Parameters:
            filename: str
                The name of the file where the object is saved.

        Returns:
            model: The loaded model object.
        """
        with open(filename, "rb") as file:
            model = pickle.load(file)
        return model

    def update_arguments(
        self,
        prior=None,
        kernel=None,
        hpfitter=None,
        hp={},
        use_derivatives=None,
        use_correction=None,
        dtype=None,
        **kwargs,
    ):
        """
        Update the Model Process Regressor with its arguments.
        The existing arguments are used if they are not given.

        Parameters:
            prior: Prior class
                The prior given for new data.
            kernel: Kernel class
                The kernel function used for the kernel matrix.
            hpfitter: HyperparameterFitter class
                A class to optimize hyperparameters
            hp: dictionary
                A dictionary of hyperparameters like noise and length scale.
                The hyperparameters are used in the log-space.
            use_derivatives: bool
                Use derivatives/gradients for training and predictions.
            use_correction: bool
                Use the noise correction on the covariance matrix.
            dtype: type
                The data type of the arrays.

        Returns:
            self: The updated instance itself.
        """
        # Set the prior mean class
        if prior is not None:
            self.prior = prior.copy()
        # Set the kernel class
        if kernel is not None:
            self.kernel = kernel.copy()
        # Set whether to use derivatives for the target
        if use_derivatives is not None:
            self.set_use_derivatives(use_derivatives)
        # Set noise correction
        if use_correction is not None:
            self.use_correction = use_correction
        # The hyperparameter optimization method
        if hpfitter is not None:
            self.hpfitter = hpfitter.copy()
        # Set the data type
        if dtype is not None or not hasattr(self, "dtype"):
            self.set_dtype(dtype=dtype)
        # Set hyperparameters
        self.set_hyperparams(hp)
        # Check if the attributes agree
        self.check_attributes()
        return self

    def add_regularization(self, K, n_data, overwrite=True, **kwargs):
        """
        Add the regularization to the diagonal elements of
        the squared kernel matrix.
        (K will be overwritten if overwrite=True)
        """
        # Whether to make a copy of the kernel matrix
        if not overwrite:
            K = K.copy()
        m_data = len(K)
        # Calculate the correction, so the kernel matrix is invertible
        self.corr = self.get_correction(diag(K))
        add_v = self.inf_to_num(exp(2.0 * self.hp["noise"][0])) + self.corr
        if "noise_deriv" in self.hp:
            K[range(n_data), range(n_data)] += add_v
            add_v = self.inf_to_num(exp(2.0 * self.hp["noise_deriv"][0]))
            add_v += self.corr
            K[range(n_data, m_data), range(n_data, m_data)] += add_v
        else:
            K[range(m_data), range(m_data)] += add_v
        return K

    def inf_to_num(self, value, replacing=1e300):
        "Check if a value is infinite and then replace it with a large number."
        if value == inf:
            return replacing
        return value

    def get_correction(self, K_diag=None, **kwargs):
        """
        Get the noise correction, so that the training covariance matrix
        is always invertible.

        Parameters:
            K_diag: N or N*(D+1) array (optional)
                The diagonal elements of the kernel matrix.
                If it is not given, the stored noise correction is used.
        """
        if self.use_correction and K_diag is not None:
            K_sum = K_diag.sum()
            n = len(K_diag)
            corr = (K_sum**2) * (1.0 / ((1.0 / self.eps) - (n**2)))
        elif self.use_correction and K_diag is None:
            corr = self.corr
        else:
            corr = 0.0
        return corr

    def calculate_kernel_decomposition(self, features, **kwargs):
        "Do the Cholesky decomposition of the kernel matrix."
        # Make kernel matrix with noise
        K = self.get_kernel(features, get_derivatives=self.use_derivatives)
        K = self.add_regularization(K, len(features))
        # Do Cholesky decomposition
        return cho_factor(K)

    def modify_targets(self, features, targets, **kwargs):
        "Modify the targets with the prior mean and rearrangement."
        # Subtracting prior mean from target
        targets_mod = array(targets, dtype=self.dtype)
        self.update_priormean(features, targets_mod, L=self.L, low=self.low)
        targets_mod -= self.get_priormean(
            features,
            targets,
            get_derivatives=self.use_derivatives,
        )
        # Rearrange targets if derivatives are used
        if self.use_derivatives:
            targets_mod = targets_mod.T.reshape(-1, 1)
        else:
            targets_mod = targets_mod[:, 0:1]
        return targets_mod

    def calculate_coefficients(self, features, targets, **kwargs):
        "Calculate the coefficients for the prediction mean."
        return cho_solve((self.L, self.low), targets, check_finite=False)

    def calculate_prefactor(self, features=None, targets=None, **kwargs):
        """
        Calculate the prefactor that the prediction uncertainty is scaled with.
        """
        raise NotImplementedError()

    def kernel_diag(
        self,
        features,
        m_data,
        get_derivatives=False,
        include_noise=False,
        **kwargs,
    ):
        "Calculate the diagonal of the kernel matrix of the test data."
        # Calculate diagonal of the kernel of the test data without noise
        k = self.kernel.diag(features, get_derivatives=get_derivatives)
        # Add noise to the kernel elements
        if include_noise:
            noise = nan_to_num(exp(2.0 * self.hp["noise"][0]))
            noise += self.corr
            if get_derivatives and "noise_deriv" in self.hp:
                k[:m_data] += noise
                k[m_data:] += (
                    nan_to_num(exp(2.0 * self.hp["noise_deriv"][0]))
                    + self.corr
                )
            else:
                k += noise
        return k

    def kernel_deriv_diag(self, features, **kwargs):
        """
        Calculate the derivative of the diagonal elements in the kernel
        wrt. the features.
        """
        return self.kernel.diag_deriv(features, **kwargs)

    def calculate_CinvKQX(self, KQX, **kwargs):
        "Calculate the CinvKQX matrix."
        return cho_solve((self.L, self.low), KQX.T, check_finite=False)

    def check_attributes(self):
        "Check if all attributes agree between the class and subclasses."
        if self.use_derivatives != self.kernel.get_use_derivatives():
            raise ValueError(
                "The Model and the Kernel do not agree "
                "whether to use derivatives!"
            )
        return True

    def get_arguments(self):
        "Get the arguments of the class itself."
        # Get the arguments given to the class in the initialization
        arg_kwargs = dict(
            prior=self.prior,
            kernel=self.kernel,
            hpfitter=self.hpfitter,
            hp=self.get_hyperparams(),
            use_derivatives=self.use_derivatives,
            use_correction=self.use_correction,
            dtype=self.dtype,
        )
        # Get the constants made within the class
        constant_kwargs = dict(
            trained_model=self.trained_model,
            corr=self.corr,
            low=self.low,
            prefactor=self.prefactor,
        )
        # Get the objects made within the class
        object_kwargs = dict(features=self.features, L=self.L, coef=self.coef)
        return arg_kwargs, constant_kwargs, object_kwargs

    def copy(self):
        "Copy the object."
        # Get all arguments
        arg_kwargs, constant_kwargs, object_kwargs = self.get_arguments()
        # Make a clone
        clone = self.__class__(**arg_kwargs)
        # Check if constants have to be saved
        if len(constant_kwargs.keys()):
            for key, value in constant_kwargs.items():
                clone.__dict__[key] = value
        # Check if objects have to be saved
        if len(object_kwargs.keys()):
            for key, value in object_kwargs.items():
                clone.__dict__[key] = value.copy()
        return clone

    def __repr__(self):
        arg_kwargs = self.get_arguments()[0]
        str_kwargs = ",".join(
            [f"{key}={value}" for key, value in arg_kwargs.items()]
        )
        return "{}({})".format(self.__class__.__name__, str_kwargs)
