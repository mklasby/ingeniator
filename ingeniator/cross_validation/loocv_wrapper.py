import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse.construct import rand, random
import seaborn as sns
import os
from sklearn.model_selection import LeaveOneOut, train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, PowerTransformer, LabelEncoder
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.utils import resample
from procan_connectome.config import DATA_PATH, RANDOM_STATE
import logging
import datetime
from procan_connectome.data_processing.linear_svc_importance_filter import LinearSVCImportanceFilter
from procan_connectome.data_processing.rf_importance_filter import RFImportanceFilter
from procan_connectome.data_processing.importance_filter import svc_loovc
from procan_connectome.utils.load_dataset import _get_important_features

class LOOCV_Wrapper(BaseEstimator): 
    """TODO: 
        Test with small df to check final results functions. 
        Run on ARC! 

    """
    def __init__(
        self,
        X:pd.DataFrame,
        y:pd.Series,
        estimator:BaseEstimator,
        pipeline: Pipeline=None,
        perform_grid_search:bool=False,
        param_grid:dict=None,
        log_dir:str = DATA_PATH,
        log_file_name:str= None,
        copy:bool=True, 
        label_col:str=None, 
        balance_classes:bool=False, 
        scoring:str='accuracy',
        verbose:int=2,
        single_label_upsample: str = None,
        n_samples: int = None,
        encode_labels: bool = False,
        cv: int = None, 
        select_features:bool=False,
        feature_threshold:float=0.01,
        random_state = RANDOM_STATE,
        grid_search_feature_selection:bool=False):

        self.X=X
        self.y=y
        self.estimator = estimator
        self.pipeline = pipeline
        self.perform_grid_search=perform_grid_search 
        self.param_grid=param_grid
        self.log_dir=log_dir
        self.log_file_name = log_file_name
        self.copy=copy 
        self.label_col = label_col
        self.balance_classes = balance_classes
        self.scoring=scoring
        self.verbose=verbose
        self.n_samples = n_samples
        self.single_label_upsample = single_label_upsample
        self.cv = cv
        self.encode_labels = encode_labels
        self.select_features = select_features
        self.feature_threshold = feature_threshold
        self.random_state = random_state
        self.grid_search_feature_selection = grid_search_feature_selection

    def _reset(self): 
        if hasattr(self, 'pipe_'): 
            del self.pipe_
        if hasattr(self, 'results_df_'):
            del self.results_df_
        if hasattr(self, 'y_pred_'): 
            del self.y_pred_
            del self.y_true_
            del self.accuracy_scores_
        if hasattr(self, 'importances_'):
            del self.importances_
        if hasattr(self, 'best_params_'):
            del self.best_params_
        if hasattr(self, "le"):
            del self.le

    def _pipeline_transform(self, X_train, X_test, y_train=None) -> tuple:
        self.pipe_ = clone(self.pipeline)
        # cols = X_train.columns
        # train_idx = X_train.index
        # test_idx = X_test.index
        # X_train = self.pipe_.fit_transform(X_train, y_train)
        # X_test = self.pipe_.transform(X_test)
        # return pd.DataFrame(X_train, columns=cols, index=train_idx), pd.DataFrame(X_test, columns=cols, index=test_idx)
        X_train = self.pipe_.fit_transform(X_train, y_train)
        X_test = self.pipe_.transform(X_test)
        return X_train, X_test
        

    def _get_best_grid_search_estimator(self, X_train:pd.DataFrame, y_train:pd.Series,
                                        estimator:BaseEstimator) -> BaseEstimator:
        if self.cv is None: 
            cv = y_train.groupby(y_train).count().min()
            if cv == 1: 
                cv = 5
        else: 
            cv = self.cv
        grid_search = GridSearchCV(
            estimator=estimator, 
            param_grid=self.param_grid,
            scoring=self.scoring, 
            verbose=self.verbose,
            cv=cv,
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        self.best_params_.append(grid_search.best_params_)
        self.accuracy_scores_.append(grid_search.best_score_)
        return grid_search.best_estimator_


    def _upsample_label(self, df: pd.DataFrame, label:str, n_samples:int)->pd.DataFrame: 
        label_subset = df[df[self.label_col] == label]
        if len(label_subset) == n_samples:
            return label_subset
        else: 
            return resample(label_subset,
                            replace=True, 
                            n_samples=n_samples,
                            random_state=RANDOM_STATE,
                ) 

    def _get_balanced_train(self, X_train:pd.DataFrame, y_train:pd.Series, n_samples:int=None,
                            single_label:str=None) -> tuple:
        df = X_train.join(y_train)
        dfs = []
        if n_samples is None: 
            n_samples = df.groupby(self.label_col).count().iloc[:,0].max()
        if single_label is not None: 
            dfs.append(df[df[self.label_col]!=single_label])
            dfs.append(self._upsample_label(df, single_label, n_samples))
        else: 
            labels = df[self.label_col].unique()
            for label in labels:
                dfs.append(self._upsample_label(df, label, n_samples))
        df_upsampled = pd.concat(dfs)
        return df_upsampled.drop(columns=[y_train.name]), df_upsampled[y_train.name]


    def _get_estimator_name(self, estimator:BaseEstimator): 
        estimator_name = str(type(estimator)).split(".")[-1][:-2]
        return estimator_name

    def _save_combined_feature_importances(self):
        labels = self.X.columns  
        feature_importances = np.array(self.importances_).mean(axis=0)
        feature_importances_df = pd.DataFrame(sorted(zip(map(lambda x: round(x,6), feature_importances), labels), reverse=True), columns=["Importance", "Feature"])
        save_path = os.path.join(self.log_dir, self.log_file_name + '_feature_importances.csv')
        feature_importances_df.to_csv(save_path, index=False)
        logging.info(f"Feature importances saved to: {save_path}")
        return 

    def _svc_select_features(self, X_train: pd.DataFrame, y_train:pd.Series) -> pd.DataFrame: 
        svc = LinearSVCImportanceFilter(self.feature_threshold,
                                        random_state=self.random_state, sort=False)
        logging.info(f"X_train before feature selection: {X_train.shape}")
        if self.grid_search_feature_selection: 
            _, feature_importances_df = svc_loovc(X_train, y_train, False, False, threshold=self.feature_threshold)
            important_features = feature_importances_df.loc[feature_importances_df['Importance'] >= self.feature_threshold]['Feature']
            X_train = X_train[important_features.values.tolist()]
            self.importances_.append(feature_importances_df['Importance'].values)
        else: 
            X_train = svc.fit_transform(X_train, y_train)
            self.importances_.append(svc.feature_importances_df_['Importance'].values)
            logging.info(self._check_features(X_train, svc))
        logging.info(f"X_train after feature selection: {X_train.shape}")
        return X_train 

    def _check_features(self, X_train: pd.DataFrame, importance_filter):
        for feature in X_train.columns: 
            if feature not in importance_filter.important_features_.values.tolist(): 
                raise ValueError(f"{feature} missing from X_train!")
        return 'Features OK!'


    def _rf_select_features(self, X_train: pd.DataFrame, y_train:pd.Series) -> pd.DataFrame: 
        rf = RFImportanceFilter(threshold = self.feature_threshold, 
                                random_state=self.random_state, sort=False)
        X_train = rf.fit_transform(X_train, y_train)
        self.importances_.append(rf.feature_importances_df_['Importance'].values)
        print(self._check_features(X_train, rf))
        return X_train 
        
    def _select_features(self, X_train: pd.DataFrame, y_train:pd.Series) -> pd.DataFrame: 
        if type(self.estimator) == LinearSVC: 
            X_train = self._svc_select_features(X_train, y_train)
        elif type(self.estimator) == RandomForestClassifier: 
            X_train = self._rf_select_features(X_train, y_train)
        else: 
            raise TypeError("Select features has only been implemented for LinearSVC and RandomForestClassifier at this time...")

        ## TODO: Implement baseEnsemble feature selection as well. 
        return X_train 

    def fit(self, X=None, y=None): 
        if X is not None: 
            self.X = X
        if y is not None: 
            self.y = y
        self._fit()

    def _fit(self): 
        self._reset()
        loo = LeaveOneOut()
        self.y_true_ = []
        self.y_pred_ = []
        self.importances_ = []
        self.best_params_ = []
        self.accuracy_scores_ = []
        X = self.X.copy(deep=True)
        y = self.y.copy(deep=True)
        counter = 0
        est_name = self._get_estimator_name(self.estimator)
        logging.info(f"Fitting LOOCV models for {est_name}")
        
        if self.encode_labels:
            self.le = LabelEncoder()
            y = pd.Series(self.le.fit_transform(y), name=self.y.name, index=self.y.index)
            
        for train_idx, test_idx in loo.split(X):
            counter +=1
            logging.info(f"Fitting model {counter} of {len(X)}")
            X_train, X_test = X.iloc[train_idx, :], X.iloc[test_idx, :]
            y_train, y_test = y[train_idx], y[test_idx]
            estimator = clone(self.estimator)
            
            if self.balance_classes: 
                logging.debug(f'Balancing classes for iteration {counter}...')
                X_train, y_train = self._get_balanced_train(X_train, y_train, n_samples=self.n_samples,
                                                            single_label=self.single_label_upsample)
                logging.info(f"Upsampled breakdown: {y_train.groupby(y_train).count()}")

            if self.pipeline is not None: 
                logging.debug(f'Fitting pipe to iteration {counter}...')
                X_train, X_test = self._pipeline_transform(X_train, X_test, y_train)

            if self.select_features: 
                logging.debug(f'Selecting top {self.feature_threshold} features for iteration {counter}...')
                X_train = self._select_features(X_train, y_train)
                X_test = X_test[X_train.columns]       
            
            if self.perform_grid_search: 
                logging.debug(f'Performing grid search for iteration {counter}')
                start = datetime.datetime.now()
                estimator = self._get_best_grid_search_estimator(X_train, y_train, estimator)
                end=datetime.datetime.now()
                logging.debug(f'Grid search for iteration {counter} completed in: {end-start}...')

            estimator.fit(X_train,y_train)
            self.y_pred_.append(estimator.predict(X_test)[0])
            self.y_true_.append(y_test.values[0])

        logging.info(f"LOOCV model fit complete for {est_name}")
        acc = accuracy_score(self.y_true_, self.y_pred_)
        logging.info(f"Overall accuracy for {est_name}: {acc}")

        if self.perform_grid_search:
            grid_results = pd.DataFrame({
                "Best_Params": self.best_params_,
                "Scores": self.accuracy_scores_
            })
            fname = os.path.join(self.log_dir, self.log_file_name+'_grid_search' + '.csv')
            grid_results.to_csv(fname)
            logging.info(
                f"Saved grid search results to {fname}"
            )

        if self.select_features: 
            self._save_combined_feature_importances() 
            
        if self.encode_labels:
            self.y_pred_ = self.le.inverse_transform(self.y_pred_)
            self.y_true_ = self.le.inverse_transform(self.y_true_)

        self.results_df_ = pd.DataFrame({
            "y_true": self.y_true_,
            'y_pred': self.y_pred_
        }).set_index(X.index)
        fname = os.path.join(self.log_dir, self.log_file_name+'_results.csv')
        self.results_df_.to_csv(fname)
        logging.info(
            f"Saved results to {fname}"
        )

        
                                                    





