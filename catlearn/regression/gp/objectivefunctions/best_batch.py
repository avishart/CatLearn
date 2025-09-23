from numpy import arange, inf
from .batch import BatchFuction


class BestBatchFuction(BatchFuction):
    """
    The objective function that is used to optimize the hyperparameters.
    The instance splits the training data into batches.
    A given objective function is then used as
    an objective function for the batches.
    The lowest function value and it corresponding hyperparameters
    from a single batch are used.
    BestBatchFuction is not recommended for gradient-based optimization!
    """

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
        hp, _ = self.make_hp(theta, parameters)
        model = self.update_model(model, hp)
        self.set_same_prior_mean(model, X, Y)
        # Calculate the number of batches
        n_batches = self.get_number_batches(n_data)
        indices = arange(n_data)
        i_batches = self.randomized_batches(
            indices, n_data, n_batches, **kwargs
        )
        # Sum function values together from batches
        fvalue = inf
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
