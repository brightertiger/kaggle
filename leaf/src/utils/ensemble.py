import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
import pickle

class ModelEnsemble:
    def __init__(self):
        self.models = {}
        self.blend_model = None
    
    def add_model_predictions(self, model_name, predictions_path):
        predictions = pd.read_csv(predictions_path)
        self.models[model_name] = predictions
        print(f"Added predictions from {model_name}")
    
    def create_blend_features(self, data):
        features = []
        for model_name, preds in self.models.items():
            model_features = [col for col in preds.columns if col not in ['image_id', 'label']]
            features.extend([f"{model_name}_{feat}" for feat in model_features])
        
        blend_data = data[['image_id', 'fold', 'label']].copy()
        
        for model_name, preds in self.models.items():
            model_features = [col for col in preds.columns if col not in ['image_id', 'label']]
            preds.columns = ['image_id', 'label'] + [f"{model_name}_{feat}" for feat in model_features]
            blend_data = blend_data.merge(preds, on=['image_id', 'label'])
        
        return blend_data, features
    
    def train_blend_model(self, fold, features, baseline_features=None):
        data = self.create_blend_features(self.data)[0]
        
        train_data = data[data['fold'] != fold].copy().reset_index(drop=True)
        valid_data = data[data['fold'] == fold].copy().reset_index(drop=True)
        
        X_train = np.array(train_data[features])
        y_train = np.array(train_data['label'])
        X_valid = np.array(valid_data[features])
        y_valid = np.array(valid_data['label'])
        
        if baseline_features:
            y_baseline = np.argmax(np.array(valid_data[baseline_features]), axis=1)
        else:
            y_baseline = None
        
        model = LogisticRegression(max_iter=500, multi_class='ovr', C=0.2)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_valid)
        
        baseline_acc = np.mean(y_baseline == y_valid) if y_baseline is not None else 0
        model_acc = np.mean(y_pred == y_valid)
        
        print(f'Fold {fold} - Baseline: {baseline_acc:.4f}, Model: {model_acc:.4f}')
        
        pickle.dump(model, open(f'../../model/blend/blend_{fold}.pkl', 'wb'))
        
        return {
            'fold': fold,
            'baseline_acc': baseline_acc,
            'model_acc': model_acc,
            'predictions': y_pred
        }
    
    def create_ensemble(self, data_path, model_predictions):
        self.data = pd.read_csv(data_path)
        
        for model_name, pred_path in model_predictions.items():
            self.add_model_predictions(model_name, pred_path)
        
        features = []
        for model_name, preds in self.models.items():
            model_features = [col for col in preds.columns if col not in ['image_id', 'label']]
            features.extend([f"{model_name}_{feat}" for feat in model_features])
        
        baseline_features = [feat for feat in features if 'version7' in feat]
        
        results = []
        for fold in range(5):
            result = self.train_blend_model(fold, features, baseline_features)
            results.append(result)
        
        overall_acc = np.mean([r['model_acc'] for r in results])
        print(f'Overall Ensemble Accuracy: {overall_acc:.4f}')
        
        return results
