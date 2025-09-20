import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class TalkingDataProcessor:
    """Main data processor for TalkingData AdTracking dataset."""
    
    def __init__(self, config):
        self.config = config
        self.dtypes = config.DTYPES
        
    def load_raw_data(self, data_type: str = 'train') -> pd.DataFrame:
        """Load raw data files with optimized data types."""
        if data_type == 'train':
            file_path = self.config.RAW_DATA_DIR / self.config.TRAIN_FILE
            usecols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
        elif data_type == 'test':
            file_path = self.config.RAW_DATA_DIR / self.config.TEST_FILE
            usecols = ['click_id', 'ip', 'app', 'device', 'os', 'channel', 'click_time']
        elif data_type == 'supplement':
            file_path = self.config.RAW_DATA_DIR / self.config.SUPPLEMENT_FILE
            usecols = ['click_id', 'ip', 'app', 'device', 'os', 'channel', 'click_time']
        else:
            raise ValueError("data_type must be 'train', 'test', or 'supplement'")
            
        dtypes = {k: v for k, v in self.dtypes.items() if k in usecols}
        
        data = pd.read_csv(file_path, dtype=dtypes, usecols=usecols)
        data['click_time'] = pd.to_datetime(data['click_time'])
        
        return data
    
    def create_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features from click_time."""
        data = data.copy()
        data['hour'] = data['click_time'].dt.hour
        data['day'] = data['click_time'].dt.day
        
        return data
    
    def filter_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply time-based and hour-based filtering."""
        data = data.copy()
        
        # Filter by start date
        data['keep_row'] = data['click_time'] > self.config.START_DATE
        
        # Filter by specific hours
        data['keep_hour'] = data['hour'].isin(self.config.KEEP_HOURS)
        
        # Combine filters
        data['keep'] = data['keep_row'] + data['keep_hour']
        data = data[data['keep'] > 0]
        
        return data
    
    def create_train_validation_split(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into training and validation sets based on date."""
        data = data.copy()
        
        # Create validation split based on date
        data['valid_date'] = data['click_time'] > self.config.VALID_DATE
        data['is_valid'] = data['valid_date'] * data['keep_hour']
        
        valid_data = data[data['is_valid'] == True].copy()
        train_data = data[data['is_valid'] == False].copy()
        
        # Clean up temporary columns
        drop_cols = ['valid_date', 'is_valid', 'keep_row', 'keep', 'keep_hour']
        train_data = train_data.drop(drop_cols, axis=1, errors='ignore')
        valid_data = valid_data.drop(drop_cols, axis=1, errors='ignore')
        
        # Reset indices
        train_data = train_data.reset_index(drop=True)
        valid_data = valid_data.reset_index(drop=True)
        
        # Add click_id for train data
        if 'click_id' not in train_data.columns:
            train_data['click_id'] = train_data.index
        
        return train_data, valid_data
    
    def prepare_final_data(self, data: pd.DataFrame, data_type: str = 'train') -> pd.DataFrame:
        """Prepare final dataset with selected columns."""
        if data_type == 'train':
            cols = ['click_id', 'is_attributed', 'day', 'hour', 'ip', 'app', 'os', 'device', 'channel']
        else:
            cols = ['click_id', 'day', 'hour', 'ip', 'app', 'os', 'device', 'channel']
            
        return data[cols]
    
    def save_processed_data(self, data: pd.DataFrame, filename: str):
        """Save processed data to feather format."""
        file_path = self.config.PROCESSED_DATA_DIR / filename
        data.to_feather(file_path)
        print(f"Saved {filename}: {data.shape}")

class FeatureEngineer:
    """Feature engineering class for creating various types of features."""
    
    def __init__(self, config):
        self.config = config
    
    def create_count_features(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create count-based features for various combinations."""
        features = {}
        
        # Single feature counts
        for feature in ['ip', 'app', 'os']:
            grouped = data.groupby(feature)['hour'].count().reset_index()
            grouped.columns = [feature, f'{feature}_cnt']
            grouped = self._optimize_dtypes(grouped)
            features[f'{feature}_cnt'] = grouped
        
        # Multi-feature counts
        combinations = [
            (['ip', 'day', 'hour'], 'device'),
            (['ip', 'app'], 'device'),
            (['ip', 'app', 'os'], 'device'),
            (['ip', 'device'], 'os'),
            (['app', 'channel'], 'os'),
            (['ip', 'hour', 'os'], 'channel'),
            (['ip', 'hour', 'app'], 'channel')
        ]
        
        for features_list, count_col in combinations:
            grouped = data.groupby(features_list)[count_col].count().reset_index()
            feature_name = '_'.join(features_list)
            grouped.columns = features_list + [f'{feature_name}_cnt']
            grouped = self._optimize_dtypes(grouped)
            features[f'{feature_name}_cnt'] = grouped
        
        return features
    
    def create_unique_features(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create unique count features."""
        features = {}
        
        # IP-app unique combinations
        ip_app_unq = data.groupby('ip')['app'].nunique().reset_index()
        ip_app_unq.columns = ['ip', 'ip_app_unq']
        ip_app_unq = self._optimize_dtypes(ip_app_unq)
        features['ip_app_unq'] = ip_app_unq
        
        # IP-channel unique combinations
        ip_channel_unq = data.groupby('ip')['channel'].nunique().reset_index()
        ip_channel_unq.columns = ['ip', 'ip_channel_unq']
        ip_channel_unq = self._optimize_dtypes(ip_channel_unq)
        features['ip_channel_unq'] = ip_channel_unq
        
        return features
    
    def create_user_features(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create user-based features."""
        features = {}
        
        # User count features
        user_count = data.groupby(['ip', 'device', 'os'])['channel'].count().reset_index()
        user_count.columns = ['ip', 'device', 'os', 'user_count']
        user_count = self._optimize_dtypes(user_count)
        features['user_count'] = user_count
        
        # User-app count features
        user_app_count = data.groupby(['ip', 'device', 'os', 'app'])['channel'].count().reset_index()
        user_app_count.columns = ['ip', 'device', 'os', 'app', 'user_app_count']
        user_app_count = self._optimize_dtypes(user_app_count)
        features['user_app_count'] = user_app_count
        
        return features
    
    def create_next_click_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create next click time features."""
        data = data.copy()
        data['click_time'] = data['click_time'].astype(np.int64) // 10 ** 9
        data = data.sort_values(by=['ip', 'app', 'device', 'os', 'click_time'])
        
        # Calculate next click time
        data['next_time'] = data.groupby(['ip', 'app', 'device', 'os'])['click_time'].shift(-1)
        data['next_click'] = data['next_time'] - data['click_time']
        data['next_click'] = data['next_click'].fillna(-1.0)
        
        return data[['click_id', 'next_click']]
    
    def create_ranking_features(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create ranking features for categorical variables."""
        features = {}
        
        # IP ranking
        ip_counts = data.groupby('ip').size().reset_index(name='count')
        ip_counts['ip_rank'] = ip_counts['count'].rank(method='dense', ascending=False)
        features['ip_rank'] = ip_counts[['ip', 'ip_rank']]
        
        # App-channel ranking
        app_channel_counts = data.groupby(['app', 'channel']).size().reset_index(name='count')
        app_channel_counts['app_channel_rank'] = app_channel_counts['count'].rank(method='dense', ascending=False)
        features['app_channel_rank'] = app_channel_counts[['app', 'channel', 'app_channel_rank']]
        
        # App-OS ranking
        app_os_counts = data.groupby(['app', 'os']).size().reset_index(name='count')
        app_os_counts['app_os_rank'] = app_os_counts['count'].rank(method='dense', ascending=False)
        features['app_os_rank'] = app_os_counts[['app', 'os', 'app_os_rank']]
        
        # Channel-OS ranking
        channel_os_counts = data.groupby(['channel', 'os']).size().reset_index(name='count')
        channel_os_counts['channel_os_rank'] = channel_os_counts['count'].rank(method='dense', ascending=False)
        features['channel_os_rank'] = channel_os_counts[['channel', 'os', 'channel_os_rank']]
        
        return features
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types to reduce memory usage."""
        for column in df.columns:
            if df[column].dtype == 'int64':
                if df[column].max() <= 250:
                    df[column] = df[column].astype('uint8')
                elif df[column].max() <= 65000 and df[column].min() >= 0:
                    df[column] = df[column].astype('uint16')
                elif df[column].max() <= 4294967295 and df[column].min() >= 0:
                    df[column] = df[column].astype('uint32')
        return df
