import os
import pandas as pd
from .data_preprocessing import preprocess_campaign_data, create_train_validation_split, load_driver_data
from .feature_engineering import (
    create_customer_features, create_coupon_features, create_campaign_features,
    create_coupon_spend_features, create_count_features, create_transaction_brand_features,
    create_transaction_category_features, create_transaction_item_features,
    create_similarity_features, create_time_features
)
from .modeling import train_model_v1, train_model_v2, train_model_v3
from .ensemble import rank_blend_predictions


class AmExpertPipeline:
    """Main pipeline for AmExpert coupon redemption prediction."""
    
    def __init__(self, data_dir='data', feature_dir='data/feature', 
                 model_dir='data/model', score_dir='data/score'):
        self.data_dir = data_dir
        self.feature_dir = feature_dir
        self.model_dir = model_dir
        self.score_dir = score_dir
        
        os.makedirs(self.feature_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.score_dir, exist_ok=True)
    
    def preprocess_data(self):
        """Preprocess raw data."""
        print("Preprocessing campaign data...")
        preprocess_campaign_data(
            f'{self.data_dir}/data/campaign_data.csv',
            f'{self.data_dir}/data/campaign_data_clean.csv'
        )
        
        print("Creating train/validation split...")
        train, valid = create_train_validation_split(f'{self.data_dir}/data/train.csv')
        
        driver = load_driver_data(f'{self.data_dir}/driver.csv')
        test = pd.read_csv(f'{self.data_dir}/data/test.csv')[['id']]
        
        print(f"Data shapes - Driver: {driver.shape}, Train: {train.shape}, Valid: {valid.shape}, Test: {test.shape}")
        
        return driver, train, valid, test
    
    def create_features(self):
        """Create all feature sets."""
        print("Creating customer features...")
        create_customer_features(
            f'{self.data_dir}/data/customer_demographics.csv',
            f'{self.data_dir}/data/customer_transaction_data.csv',
            f'{self.data_dir}/data/item_data.csv',
            f'{self.data_dir}/driver.csv',
            f'{self.feature_dir}/customer_feature.csv'
        )
        
        print("Creating coupon features...")
        create_coupon_features(
            f'{self.data_dir}/data/coupon_item_mapping.csv',
            f'{self.data_dir}/data/item_data.csv',
            f'{self.data_dir}/driver.csv',
            f'{self.feature_dir}/coupon_feature.csv'
        )
        
        print("Creating campaign features...")
        create_campaign_features(
            f'{self.data_dir}/data/campaign_data_clean.csv',
            f'{self.data_dir}/driver.csv',
            f'{self.feature_dir}/campaign_feature.csv'
        )
        
        print("Creating coupon spend features...")
        create_coupon_spend_features(
            f'{self.data_dir}/data/coupon_item_mapping.csv',
            f'{self.data_dir}/data/customer_transaction_data.csv',
            f'{self.data_dir}/driver.csv',
            f'{self.feature_dir}/coupon_spend_profile.csv'
        )
        
        print("Creating count features...")
        create_count_features(
            f'{self.data_dir}/data/customer_transaction_data.csv',
            f'{self.data_dir}/driver.csv',
            f'{self.feature_dir}/count_feature.csv'
        )
        
        print("Creating transaction brand features...")
        create_transaction_brand_features(
            f'{self.data_dir}/data/customer_transaction_data.csv',
            f'{self.data_dir}/data/item_data.csv',
            f'{self.data_dir}/driver.csv',
            f'{self.feature_dir}/tranx_brand_feature.csv'
        )
        
        print("Creating transaction category features...")
        create_transaction_category_features(
            f'{self.data_dir}/data/customer_transaction_data.csv',
            f'{self.data_dir}/data/item_data.csv',
            f'{self.data_dir}/driver.csv',
            f'{self.feature_dir}/tranx_category_feature.csv'
        )
        
        print("Creating transaction item features...")
        create_transaction_item_features(
            f'{self.data_dir}/data/customer_transaction_data.csv',
            f'{self.data_dir}/driver.csv',
            f'{self.feature_dir}/tranx_item_feature.csv'
        )
        
        print("Creating similarity features...")
        create_similarity_features(
            f'{self.data_dir}/data/customer_transaction_data.csv',
            f'{self.data_dir}/data/coupon_item_mapping.csv',
            f'{self.data_dir}/data/item_data.csv',
            f'{self.data_dir}/driver.csv',
            f'{self.feature_dir}/similarity.csv'
        )
        
        print("Creating time features...")
        create_time_features(
            f'{self.data_dir}/data/customer_transaction_data.csv',
            f'{self.data_dir}/data/coupon_item_mapping.csv',
            f'{self.data_dir}/data/campaign_data_clean.csv',
            f'{self.data_dir}/driver.csv',
            f'{self.feature_dir}/tranx_time_feature.csv'
        )
    
    def merge_features(self):
        """Merge all features into final datasets."""
        print("Merging features...")
        
        driver = load_driver_data(f'{self.data_dir}/driver.csv')
        train, valid = create_train_validation_split(f'{self.data_dir}/data/train.csv')
        test = pd.read_csv(f'{self.data_dir}/data/test.csv')[['id']]
        
        feature_files = [
            'customer_feature.csv', 'campaign_feature.csv', 'coupon_feature.csv',
            'coupon_spend_profile.csv', 'count_feature.csv', 'tranx_brand_feature.csv',
            'tranx_category_feature.csv', 'tranx_item_feature.csv', 'similarity.csv',
            'tranx_time_feature.csv'
        ]
        
        for feature_file in feature_files:
            feature_data = pd.read_csv(f'{self.feature_dir}/{feature_file}')
            driver = driver.merge(feature_data, on='id')
        
        train = train.merge(driver, on='id')
        valid = valid.merge(driver, on='id')
        test = test.merge(driver, on='id')
        
        print(f"Final shapes - Driver: {driver.shape}, Train: {train.shape}, Valid: {valid.shape}, Test: {test.shape}")
        
        train.to_csv(f'{self.model_dir}/train.csv', index=False)
        valid.to_csv(f'{self.model_dir}/valid.csv', index=False)
        test.to_csv(f'{self.model_dir}/test.csv', index=False)
        
        full = train.append(valid)
        full.to_csv(f'{self.model_dir}/full.csv', index=False)
        
        print(f"Redemption rates - Train: {train['redemption_status'].mean():.4f}, Valid: {valid['redemption_status'].mean():.4f}")
        
        return train, valid, test
    
    def train_models(self):
        """Train all model versions."""
        print("Training Model V1...")
        train_model_v1(
            f'{self.model_dir}/train.csv',
            f'{self.model_dir}/valid.csv',
            f'{self.model_dir}/test.csv',
            f'{self.model_dir}/lightgbm_v1.model',
            f'{self.score_dir}/score_v1.csv'
        )
        
        print("Training Model V2...")
        train_model_v2(
            f'{self.model_dir}/train.csv',
            f'{self.model_dir}/valid.csv',
            f'{self.model_dir}/test.csv',
            f'{self.model_dir}/lightgbm_v2.model',
            f'{self.score_dir}/score_v2.csv'
        )
        
        print("Training Model V3...")
        train_model_v3(
            f'{self.model_dir}/train.csv',
            f'{self.model_dir}/valid.csv',
            f'{self.model_dir}/test.csv',
            f'{self.model_dir}/lightgbm_v3.model',
            f'{self.score_dir}/score_v3.csv'
        )
    
    def blend_predictions(self):
        """Blend model predictions."""
        print("Blending predictions...")
        score_paths = [
            f'{self.score_dir}/score_v1.csv',
            f'{self.score_dir}/score_v2.csv',
            f'{self.score_dir}/score_v3.csv'
        ]
        
        final_score = rank_blend_predictions(score_paths, 'score.csv')
        return final_score
    
    def run_full_pipeline(self):
        """Run the complete pipeline."""
        print("Starting AmExpert Pipeline...")
        
        self.preprocess_data()
        self.create_features()
        self.merge_features()
        self.train_models()
        final_score = self.blend_predictions()
        
        print("Pipeline completed successfully!")
        return final_score
