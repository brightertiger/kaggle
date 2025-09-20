import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def clean_date_format(date_str):
    """Convert date from DD/MM/YY format to YYYY-MM-DD format."""
    day, month, year = date_str.split('/')
    return f'20{year}-{month}-{day}'


def encode_categorical_features(data, features):
    """Encode categorical features using LabelEncoder."""
    for feature in features:
        encoder = LabelEncoder()
        data[feature] = encoder.fit_transform(data[feature].fillna('none'))
    return data


def preprocess_campaign_data(campaign_path, output_path):
    """Clean and preprocess campaign data."""
    campaign_data = pd.read_csv(campaign_path, parse_dates=False)
    campaign_data['start_date'] = campaign_data['start_date'].map(clean_date_format)
    campaign_data['end_date'] = campaign_data['end_date'].map(clean_date_format)
    campaign_data.to_csv(output_path, index=False)
    return campaign_data


def create_train_validation_split(train_path, validation_campaign_id=13):
    """Split training data into train and validation sets."""
    train = pd.read_csv(train_path)
    valid = train[train['campaign_id'] == validation_campaign_id][['id', 'redemption_status']]
    train = train[train['campaign_id'] != validation_campaign_id][['id', 'redemption_status']]
    return train, valid


def load_driver_data(driver_path):
    """Load and prepare driver data."""
    return pd.read_csv(driver_path).drop(['coupon_id', 'campaign_id'], axis=1)
