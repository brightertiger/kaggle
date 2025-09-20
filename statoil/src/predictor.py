import numpy as np
import pandas as pd
from keras import backend as K
from sklearn.metrics import log_loss
from functools import reduce

class ModelPredictor:
    def __init__(self, config):
        self.config = config
        
    def predict_test_set(self, model_class, model_name, source_name):
        test_images = np.load(f'{self.config.DATA_DIR}/{source_name}/score/images.npy')
        test_angles = np.load(f'{self.config.DATA_DIR}/{source_name}/score/angles.npy')
        test_ids = np.load(f'{self.config.DATA_DIR}/{source_name}/score/ids.npy')
        test_generator = [test_images, test_angles]
        
        predictions = []
        for fold_idx in range(1, self.config.FOLDS + 1):
            model = model_class.define_model()
            model.load_weights(f'{self.config.MODEL_DIR}/{model_name}/model_{fold_idx}.hdf5')
            pred = model.predict(test_generator)
            predictions.append(pred[:, 0])
            K.clear_session()
        
        predictions = np.array(predictions).mean(axis=0)
        
        submission = pd.DataFrame({
            'id': test_ids,
            'is_iceberg': predictions
        })
        
        os.makedirs(self.config.SUBMISSION_DIR, exist_ok=True)
        submission.to_csv(f'{self.config.SUBMISSION_DIR}/{model_name}.csv', index=False)
        
        return submission
    
    def predict_cv_set(self, model_class, model_name, source_name):
        cv_predictions = []
        
        for fold_idx in range(1, self.config.FOLDS + 1):
            test_images = np.load(f'{self.config.DATA_DIR}/{source_name}/train/test_images_{fold_idx}.npy')
            test_angles = np.load(f'{self.config.DATA_DIR}/{source_name}/train/test_angles_{fold_idx}.npy')
            test_ids = np.load(f'{self.config.DATA_DIR}/{source_name}/train/test_ids_{fold_idx}.npy')
            test_labels = np.load(f'{self.config.DATA_DIR}/{source_name}/train/test_labels_{fold_idx}.npy')
            test_generator = [test_images, test_angles]
            
            model = model_class.define_model()
            model.load_weights(f'{self.config.MODEL_DIR}/{model_name}/model_{fold_idx}.hdf5')
            pred = model.predict(test_generator)[:, 0]
            K.clear_session()
            
            cv_data = pd.DataFrame({
                'id': test_ids,
                'label': test_labels,
                'score': np.clip(pred, 0.0001, 0.9999)
            })
            
            cv_loss = log_loss(cv_data['label'], cv_data['score'])
            print(f'Fold {fold_idx} Log Loss: {cv_loss:.6f}')
            
            cv_predictions.append(cv_data[['id', 'label', 'score']])
        
        cv_results = reduce(lambda x, y: x.append(y), cv_predictions)
        overall_loss = log_loss(cv_results['label'], cv_results['score'])
        print(f'Overall CV Log Loss: {overall_loss:.6f}')
        
        os.makedirs(f'{self.config.DATA_DIR}/model', exist_ok=True)
        cv_results.to_csv(f'{self.config.DATA_DIR}/model/{model_name}.csv', index=False)
        
        return cv_results
