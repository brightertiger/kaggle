import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from .config import Config

class EnsemblePredictor:
    def __init__(self, method='logistic_regression'):
        self.method = method
        self.models = []
        self.scalers = []
        self.is_fitted = False
    
    def add_predictions(self, predictions, name=None):
        if not hasattr(self, 'predictions_df'):
            self.predictions_df = pd.DataFrame()
        
        if name is None:
            name = f'model_{len(self.predictions_df.columns)}'
        
        self.predictions_df[name] = predictions
    
    def fit(self, train_predictions, train_targets, cv_folds=5):
        if self.method == 'logistic_regression':
            self._fit_logistic_regression(train_predictions, train_targets, cv_folds)
        elif self.method == 'random_forest':
            self._fit_random_forest(train_predictions, train_targets, cv_folds)
        elif self.method == 'weighted_average':
            self._fit_weighted_average(train_predictions, train_targets, cv_folds)
        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")
        
        self.is_fitted = True
    
    def _fit_logistic_regression(self, train_predictions, train_targets, cv_folds):
        self.models = []
        self.scalers = []
        
        for fold in range(cv_folds):
            # Create fold splits (assuming you have fold information)
            # For simplicity, we'll use random splits
            n_samples = len(train_predictions)
            fold_size = n_samples // cv_folds
            
            val_start = fold * fold_size
            val_end = (fold + 1) * fold_size if fold < cv_folds - 1 else n_samples
            
            val_indices = np.arange(val_start, val_end)
            train_indices = np.concatenate([
                np.arange(0, val_start),
                np.arange(val_end, n_samples)
            ])
            
            X_train = train_predictions[train_indices]
            y_train = train_targets[train_indices]
            X_val = train_predictions[val_indices]
            y_val = train_targets[val_indices]
            
            # Scale features
            scaler = QuantileTransformer(n_quantiles=100, output_distribution='normal')
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Train model
            model = LogisticRegression(C=1.0, fit_intercept=True, random_state=Config.SEED)
            model.fit(X_train_scaled, y_train)
            
            # Validate
            val_pred = model.predict_proba(X_val_scaled)[:, 1]
            val_auc = roc_auc_score(y_val, val_pred)
            
            print(f'Fold {fold}: AUC = {val_auc:.4f}')
            
            self.models.append(model)
            self.scalers.append(scaler)
    
    def _fit_random_forest(self, train_predictions, train_targets, cv_folds):
        self.models = []
        
        for fold in range(cv_folds):
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=Config.SEED + fold
            )
            model.fit(train_predictions, train_targets)
            self.models.append(model)
    
    def _fit_weighted_average(self, train_predictions, train_targets, cv_folds):
        # Calculate weights based on individual model performance
        weights = []
        
        for i in range(train_predictions.shape[1]):
            auc = roc_auc_score(train_targets, train_predictions[:, i])
            weights.append(auc)
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        self.weights = weights
        print(f'Model weights: {weights}')
    
    def predict(self, test_predictions):
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        if self.method == 'logistic_regression':
            return self._predict_logistic_regression(test_predictions)
        elif self.method == 'random_forest':
            return self._predict_random_forest(test_predictions)
        elif self.method == 'weighted_average':
            return self._predict_weighted_average(test_predictions)
    
    def _predict_logistic_regression(self, test_predictions):
        predictions = []
        
        for model, scaler in zip(self.models, self.scalers):
            test_scaled = scaler.transform(test_predictions)
            pred = model.predict_proba(test_scaled)[:, 1]
            predictions.append(pred)
        
        return np.mean(predictions, axis=0)
    
    def _predict_random_forest(self, test_predictions):
        predictions = []
        
        for model in self.models:
            pred = model.predict_proba(test_predictions)[:, 1]
            predictions.append(pred)
        
        return np.mean(predictions, axis=0)
    
    def _predict_weighted_average(self, test_predictions):
        return np.average(test_predictions, axis=1, weights=self.weights)

class StackingEnsemble:
    def __init__(self, base_models, meta_model=None):
        self.base_models = base_models
        self.meta_model = meta_model or LogisticRegression(random_state=Config.SEED)
        self.is_fitted = False
    
    def fit(self, X_train, y_train, X_val, y_val):
        # Get base model predictions
        base_predictions_train = []
        base_predictions_val = []
        
        for model in self.base_models:
            train_pred = model.predict_proba(X_train)[:, 1]
            val_pred = model.predict_proba(X_val)[:, 1]
            
            base_predictions_train.append(train_pred)
            base_predictions_val.append(val_pred)
        
        # Stack predictions
        stacked_train = np.column_stack(base_predictions_train)
        stacked_val = np.column_stack(base_predictions_val)
        
        # Train meta model
        self.meta_model.fit(stacked_train, y_train)
        
        # Validate
        meta_pred = self.meta_model.predict_proba(stacked_val)[:, 1]
        meta_auc = roc_auc_score(y_val, meta_pred)
        
        print(f'Meta model AUC: {meta_auc:.4f}')
        self.is_fitted = True
    
    def predict(self, X_test):
        if not self.is_fitted:
            raise ValueError("Stacking ensemble must be fitted before making predictions")
        
        # Get base model predictions
        base_predictions = []
        for model in self.base_models:
            pred = model.predict_proba(X_test)[:, 1]
            base_predictions.append(pred)
        
        # Stack predictions
        stacked_test = np.column_stack(base_predictions)
        
        # Meta model prediction
        meta_pred = self.meta_model.predict_proba(stacked_test)[:, 1]
        
        return meta_pred
