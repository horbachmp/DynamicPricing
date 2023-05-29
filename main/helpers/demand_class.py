import numpy as np
import pandas as pd
import lightgbm as lgb


class DemandModel:
    def __init__(self, filepath) -> None:
        self.model = lgb.Booster(model_file=filepath)

    def predict(self, test_data):
        test_preds = self.model.predict(test_data)
        return test_preds