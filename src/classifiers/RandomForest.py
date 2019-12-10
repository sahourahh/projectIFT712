from sklearn.ensemble import RandomForestClassifier
class RandamForest(ParametricClassifier):

    def __init__(self, data_manager):
        super().__init__(data_manager)
        self.model = RandomForestClassifier(n_estimators=10)
        self.param_grid = {"n_estimators": np.linspace(start = 200, stop = 2000, num = 10),
                           "max_depth": np.linspace(10, 100, num = 10)}