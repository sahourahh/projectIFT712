from sklearn.ensemble import RandomForestClassifier
class RandamForest(ParametricClassifier):

    def __init__(self, data_manager):
        super().__init__(data_manager)
        self.model = RandomForestClassifier(n_estimators=10)
        self.param_grid = {"max_depth": np.linspace(10, 100, num = 10)}