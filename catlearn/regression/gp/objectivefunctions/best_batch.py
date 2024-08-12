import numpy as np
from .batch import BatchFuction


class BestBatchFuction(BatchFuction):
    def __init__(
        self,
        func,
        get_prior_mean=False,
        batch_size=25,
        equal_size=False,
        use_same_prior_mean=True,
        seed=1,
        **kwargs,
    ):
        """
        The objective function that is used to optimize the hyperparameters.
        The instance splits the training data into batches.
        A given objective function is then used as
        an objective function for the batches.
        The lowest function value and it corresponding hyperparameters
        from a single batch are used.
        BestBatchFuction is not recommended for gradient-based optimization!

        Parameters:
            func : ObjectiveFunction class
                A class with the objective function used
                to optimize the hyperparameters.
            get_prior_mean : bool
                Whether to get the parameters of the prior mean
                in the solution.
            equal_size : bool
                Whether the clusters are forced to have the same size.
            use_same_prior_mean : bool
                Whether to use the same prior mean for all models.
            seed : int (optional)
                The random seed used to permute the indicies.
                If seed=None or False or 0, a random seed is not used.
        """
        # Set the arguments
        super().__init__(
            func=func,
            get_prior_mean=get_prior_mean,
            batch_size=batch_size,
            equal_size=equal_size,
            use_same_prior_mean=use_same_prior_mean,
            seed=seed,
            **kwargs,
        )

    def function(
        self,
        theta,
        parameters,
        model,
        X,
        Y,
        pdis=None,
        jac=False,
        **kwargs,
    ):
        # Get number of data points
        n_data = len(X)
        # Return a single function evaluation if the data set is a batch size
        if n_data <= self.batch_size:
            output = self.func.function(
                theta,
                parameters,
                model,
                X,
                Y,
                pdis=pdis,
                jac=jac,
                **kwargs,
            )
            self.sol = self.func.sol
            return output
        # Update the model with hyperparameters and prior mean
        hp, parameters_set = self.make_hp(theta, parameters)
        model = self.update_model(model, hp)
        self.set_same_prior_mean(model, X, Y)
        # Calculate the number of batches
        n_batches = self.get_number_batches(n_data)
        indicies = np.arange(n_data)
        i_batches = self.randomized_batches(
            indicies, n_data, n_batches, **kwargs
        )
        # Sum function values together from batches
        fvalue = np.inf
        deriv = None
        for i_batch in i_batches:
            # Get the feature and target batch
            X_split = X[i_batch]
            Y_split = Y[i_batch]
            # Reset solution so results can be extracted
            self.func.reset_solution()
            # Evaluate the function
            f = self.func.function(
                theta,
                parameters,
                model,
                X_split,
                Y_split,
                pdis=pdis,
                jac=jac,
                **kwargs,
            )
            if jac:
                f, d = f
            # Check if it is the best solution
            if f < fvalue:
                fvalue = f
                if jac:
                    deriv = d
                # Extract the hp from the solution
                hp = self.extract_hp_sol(hp, **kwargs)
        if jac:
            self.update_solution(
                fvalue,
                theta,
                hp,
                model,
                jac=jac,
                deriv=deriv,
            )
            return fvalue, deriv
        self.update_solution(fvalue, theta, hp, model, jac=False)
        return fvalue

    def extract_hp_sol(self, hp, **kwargs):
        "Extract the hyperparameter solution from the objective function"
        if self.use_analytic_prefactor or self.use_optimized_noise:
            sol = self.func.get_stored_solution()
            if self.use_analytic_prefactor:
                hp["prefactor"] = sol["hp"]["prefactor"]
            if self.use_optimized_noise:
                hp["noise"] = sol["hp"]["noise"]
        return hp
