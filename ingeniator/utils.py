import numpy as np
import pandas as pd


def toy_feature_selection_dataset(as_frame= True, num_features: int = 1000, num_samples: int = 1000, signal_features = 50,
                                   random_state = 42, classification_targets = False):
    np.random.seed(random_state)
    signal = np.random.rand(num_samples,signal_features)
    weight_vector = np.random.rand(signal_features,1)
    y = signal.dot(weight_vector) + np.random.rand()
    X = np.concatenate([signal,np.random.rand(num_samples, num_features-signal_features)], axis=1)
    if classification_targets:
        y = y > y.mean()
    if not as_frame:
        return X,y
    else:
        columns = [f"signal_{i+1}" if i < signal_features else f"noise_{i-signal_features+1}" for i in range(num_features) ]
        df = pd.DataFrame(X,columns=columns)
        return df, pd.Series(y.reshape(-1))
