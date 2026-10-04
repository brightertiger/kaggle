import pandas as pd

from .data_utils import FeatureEngineer, TalkingDataProcessor

class DataPreprocessor:
    """Data preprocessing pipeline for TalkingData AdTracking dataset."""
    
    def __init__(self, config):
        self.config = config
        self.feature_engineer = FeatureEngineer(config)
        
    def preprocess_all_data(self):
        """Run complete data preprocessing pipeline."""
        print("Starting data preprocessing...")
        self.config._create_directories()
        
        # Load and process training data
        print("Processing training data...")
        train_data = self._load_and_process_train_data()
        
        # Load and process test data
        print("Processing test data...")
        test_data = self._load_and_process_test_data()
        
        # Load supplement data
        print("Processing supplement data...")
        supplement_data = self._load_supplement_data()
        
        # Match complete events before counting so test overlap is counted once.
        id_mapping = self._create_id_mapping(test_data, supplement_data)

        # Create feature mappings
        print("Creating feature mappings...")
        self._create_feature_mappings(train_data, test_data, supplement_data, id_mapping)
        
        print("Data preprocessing completed!")
        
    def _load_and_process_train_data(self) -> pd.DataFrame:
        """Load and process training data."""
        processor = TalkingDataProcessor(self.config)
        
        # Load raw data
        data = processor.load_raw_data('train')
        
        # Create time features
        data = processor.create_time_features(data)
        
        # Filter data
        data = processor.filter_data(data)
        
        # Split into train/validation
        train_data, valid_data = processor.create_train_validation_split(data)
        
        # Prepare final data
        train_data = processor.prepare_final_data(train_data, 'train')
        valid_data = processor.prepare_final_data(valid_data, 'train')
        
        # Save processed data
        processor.save_processed_data(train_data, 'train_data.feather')
        processor.save_processed_data(valid_data, 'valid_data.feather')
        
        return train_data
    
    def _load_and_process_test_data(self) -> pd.DataFrame:
        """Load and process test data."""
        processor = TalkingDataProcessor(self.config)
        
        # Load raw data
        data = processor.load_raw_data('test')
        
        # Create time features
        data = processor.create_time_features(data)
        
        # Prepare final data
        data = processor.prepare_final_data(data, 'test')
        
        # Save processed data
        processor.save_processed_data(data, 'test_data.feather')
        
        return data
    
    def _load_supplement_data(self) -> pd.DataFrame:
        """Load supplement data."""
        processor = TalkingDataProcessor(self.config)
        
        # Load raw data
        data = processor.load_raw_data('supplement')
        
        return processor.create_time_features(data)
    
    def _create_feature_mappings(self, train_data: pd.DataFrame, 
                                test_data: pd.DataFrame, 
                                supplement_data: pd.DataFrame, id_mapping: pd.DataFrame):
        """Create feature mappings for count features."""
        
        # Combine all data for feature creation
        unmatched_test = test_data[~test_data['click_id'].isin(id_mapping['click_id'])]
        feature_cols = ['ip', 'app', 'device', 'os', 'channel', 'day', 'hour']
        combined_data = pd.concat(
            [frame[feature_cols] for frame in (train_data, unmatched_test, supplement_data)],
            ignore_index=True,
        )
        
        # Create count features
        count_features = self.feature_engineer.create_count_features(combined_data)
        
        # Save count features
        for name, feature_df in count_features.items():
            file_path = self.config.FEATURES_DIR / 'count' / f'{name}.feather'
            file_path.parent.mkdir(parents=True, exist_ok=True)
            feature_df.to_feather(file_path)
        
        # Create unique features
        unique_features = self.feature_engineer.create_unique_features(combined_data)
        
        # Save unique features
        for name, feature_df in unique_features.items():
            file_path = self.config.FEATURES_DIR / 'unique' / f'{name}.feather'
            file_path.parent.mkdir(parents=True, exist_ok=True)
            feature_df.to_feather(file_path)
        
        # Create user features
        user_features = self.feature_engineer.create_user_features(combined_data)
        
        # Save user features
        for name, feature_df in user_features.items():
            file_path = self.config.FEATURES_DIR / 'count' / f'{name}.feather'
            file_path.parent.mkdir(parents=True, exist_ok=True)
            feature_df.to_feather(file_path)
        
        # Create ranking features
        ranking_features = self.feature_engineer.create_ranking_features(combined_data)
        
        # Save ranking features
        for name, feature_df in ranking_features.items():
            file_path = self.config.FEATURES_DIR / 'rank' / f'{name}.feather'
            file_path.parent.mkdir(parents=True, exist_ok=True)
            feature_df.to_feather(file_path)
    
    def _create_id_mapping(self, test_data: pd.DataFrame, supplement_data: pd.DataFrame):
        """Create ID mapping between test and supplement data."""
        
        # Prepare data for mapping
        feature_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time']
        old_data = supplement_data[feature_cols + ['click_id']].rename(columns={'click_id': 'old_id'})
        new_data = test_data[feature_cols + ['click_id']].copy()
        for frame in (old_data, new_data):
            frame['_occurrence'] = frame.groupby(feature_cols).cumcount()
        
        # Create mapping based on feature combinations
        id_map = new_data.merge(
            old_data, on=feature_cols + ['_occurrence'], validate='one_to_one'
        )[['click_id', 'old_id']]
        
        # Save mapping
        file_path = self.config.PROCESSED_DATA_DIR / 'mapping.feather'
        id_map.to_feather(file_path)
        
        print(f"Created ID mapping with {len(id_map)} entries")
        return id_map
