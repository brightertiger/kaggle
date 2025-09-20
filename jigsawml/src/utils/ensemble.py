import pandas as pd
import numpy as np
from functools import reduce

class ModelEnsemble:
    def __init__(self, model_dir):
        self.model_dir = model_dir
    
    def post_process_version(self, version):
        fold_results = []
        
        for fold in range(5):
            data = pd.read_csv(f'{self.model_dir}/version{version}/score_{fold}.csv')
            data = data.append(data.iloc[:63812, :].copy())
            fold_results.append(data)
        
        print(f'Data shapes: {[df.shape for df in fold_results]}')
        
        combined = pd.concat(fold_results, ignore_index=True)
        ensemble_result = combined.groupby('id').mean().reset_index()
        
        ensemble_result.to_csv(f'{self.model_dir}/version_{version}.csv', index=False)
        return ensemble_result
    
    def create_final_ensemble(self):
        version1 = self.post_process_version('1')
        version2 = self.post_process_version('2')
        
        combined = pd.concat([version1, version2, version2, version2], ignore_index=True)
        final_ensemble = combined.groupby('id').mean().reset_index()
        
        final_ensemble.to_csv(f'{self.model_dir}/combined.csv', index=False)
        return final_ensemble
