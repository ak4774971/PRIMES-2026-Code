import numpy as np


class Generic:
    def __init__(self):
        self.nb_params = None  # Number of parameters in the layer
        self.save_X = None  # Saved layer input (set in forward)

    def set_params(self, params):
        # Set the layer parameters; input is a vector of length self.nb_params
        pass

    def get_params(self):
        # Returns a vector of length self.nb_params containing the layer parameters
        return None

    def forward(self, X):
        # Forward pass; X is the input data
        self.save_X = np.copy(X)
        return None

    def backward(self, grad_sortie):
        # Backpropagation through the layer.
        # grad_sortie is the gradient w.r.t. the layer output.
        # Returns:
        # grad_local: vector of length self.nb_params, gradient w.r.t. local parameters
        # grad_entree: gradient w.r.t. layer input
        grad_local = None
        grad_entree = None
        return grad_local, grad_entree


class Arctan(Generic):
    def __init__(self):
        self.nb_params = 0
        self.save_X = None

    def set_params(self, params):
        pass

    def get_params(self):
        pass

    def forward(self, X):
        self.save_X = np.copy(X)
        return np.arctan(self.save_X)

    def backward(self, grad_sortie):
        grad_local = None
        if self.save_X is None:
            return grad_local, None
        grad_entree = grad_sortie / (1 + self.save_X**2)
        return grad_local, grad_entree


class Dense(Generic):
    def __init__(self, nb_entree, nb_output):
        self.n_entree = nb_entree
        self.n_sortie = nb_output
        self.nb_params = self.n_entree * self.n_sortie + self.n_sortie
        scale = np.sqrt(2.0 / nb_entree)
        self.A = np.random.randn(self.n_sortie, self.n_entree) * scale
        self.b = np.random.randn(self.n_sortie)

    def set_params(self, params):
        self.A = params[: self.n_entree * self.n_sortie].reshape(
            self.n_sortie, self.n_entree
        )
        self.b = params[self.n_entree * self.n_sortie :]

    def get_params(self):
        return np.concatenate([self.A.ravel(), self.b.ravel()])

    def forward(self, X):
        self.save_X = np.copy(X)
        return self.A.dot(X) + self.b.reshape(-1, 1)

    def backward(self, grad_sortie):
        gA = grad_sortie.dot(self.save_X.T)
        gb = np.sum(grad_sortie, axis=1)
        grad_local = np.concatenate([gA.ravel(), gb.ravel()])
        grad_entree = self.A.T.dot(grad_sortie)
        return grad_local, grad_entree


class Loss_L2(Generic):
    def __init__(self, D):
        self.nb_params = 0
        self.save_D = D
        self.save_X = None

    def set_params(self):
        pass

    def get_params(self):
        pass

    def forward(self, X):
        self.save_X = np.copy(X)
        return 0.5 * np.linalg.norm(X - self.save_D) ** 2

    def backward(self, grad_sortie):
        grad_local = None
        return grad_local, self.save_X - self.save_D


class Network(Generic):
    def __init__(self, list_layers):
        self.list_layers = list_layers
        self.nb_params = sum(
            layer.nb_params
            for layer in self.list_layers
            if layer.nb_params is not None and layer.nb_params > 0
        )

    def set_params(self, params):
        offset = 0
        for layer in self.list_layers:
            if layer.nb_params is not None and layer.nb_params > 0:
                layer.set_params(params[offset : offset + layer.nb_params])
                offset += layer.nb_params

    def get_params(self):
        parts = []
        for layer in self.list_layers:
            p = layer.get_params()
            if p is not None:
                parts.append(p)
        return np.concatenate(parts)

    def forward(self, X):
        Z = np.copy(X)
        for layer in self.list_layers:
            Z = layer.forward(Z)
        return Z

    def backward(self, grad_sortie):
        grad_locals = []
        g = grad_sortie
        for layer in reversed(self.list_layers):
            gl, g = layer.backward(g)
            if gl is not None:
                grad_locals.append(gl)
        return np.concatenate(grad_locals[::-1]), g


class Sigmoid(Generic):
    def __init__(self):
        self.nb_params = 0
        self.save_X = None

    def set_params(self, params):
        pass

    def get_params(self):
        pass

    def forward(self, X):
        self.save_X = np.copy(X)
        return 1 / (1 + np.exp(-self.save_X))

    def backward(self, grad_sortie):
        grad_local = None
        if self.save_X is None:
            return grad_local, None
        grad_entree = grad_sortie * (
            1 / (1 + np.exp(-self.save_X)) * (1 - 1 / (1 + np.exp(-self.save_X)))
        )
        return grad_local, grad_entree


class Ilogit_and_KL(Generic):
    def __init__(self, D):
        self.nb_params = 0
        self.save_D = D
        self.save_X = None

    def set_params(self, params):
        pass

    def get_params(self):
        return None

    def forward(self, X):
        self.save_X = np.copy(X)
        x_max = np.max(self.save_X, axis=0)
        log_sum_exp = x_max + np.log(np.sum(np.exp(self.save_X - x_max), axis=0))
        return np.mean(log_sum_exp - np.sum(self.save_X * self.save_D, axis=0))

    def backward(self, grad_sortie):
        grad_local = None
        x_max = np.max(self.save_X, axis=0)
        ytilde = np.exp(self.save_X - x_max) / np.sum(np.exp(self.save_X - x_max), axis=0)
        batch = self.save_X.shape[1]
        return grad_local, (ytilde - self.save_D) / batch
