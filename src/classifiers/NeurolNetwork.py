from sklearn.neural_network import MLPClassifier
class MLP(ParametricClassifier):

    def __init__(self, data_manager):
        super().__init__(data_manager)
        self.model = MLPClassifier(max_iter=20, warm_start=True, verbose=True ,random_state=0)
        self.param_grid = {'hidden_layer_sizes': [(7, 7), (128,), (128, 7)],
        'tol': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
        'epsilon': [1e-3, 1e-7, 1e-8, 1e-9, 1e-8]}