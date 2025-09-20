import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from functools import reduce


def encode_categorical_features(data, features):
    """Encode categorical features using LabelEncoder."""
    for feature in features:
        encoder = LabelEncoder()
        data[feature] = encoder.fit_transform(data[feature].fillna('none'))
    return data


def create_customer_features(customer_demographics_path, transaction_path, item_path, driver_path, output_path):
    """Create comprehensive customer features."""
    profile = pd.read_csv(customer_demographics_path)
    profile = encode_categorical_features(profile, ['age_range', 'marital_status', 'no_of_children', 'family_size'])
    
    tranx = pd.read_csv(transaction_path)
    tranx['total_discount'] = tranx['other_discount'] + tranx['coupon_discount']
    tranx['other_perc'] = -1. * tranx['other_discount'] / tranx['selling_price']
    tranx['coupon_perc'] = -1. * tranx['coupon_discount'] / tranx['selling_price']
    
    item = pd.read_csv(item_path)
    item = encode_categorical_features(item, ['brand_type', 'category'])
    
    tranx = tranx.merge(item, on='item_id')
    
    features = []
    
    feat_1 = tranx.groupby('customer_id')['item_id'].agg(['count', pd.Series.nunique])
    feat_1 = feat_1.rename(columns={'count': 'cust_num_tranx', 'nunique': 'cust_unq_item'})
    features.append(feat_1.reset_index())
    
    feat_2 = tranx.groupby('customer_id')['selling_price'].mean().reset_index()
    feat_2 = feat_2.rename(columns={'selling_price': 'cust_avg_price'})
    features.append(feat_2)
    
    feat_3 = tranx.groupby('customer_id')['other_discount'].agg(['sum', np.count_nonzero]).reset_index()
    feat_3 = feat_3.rename(columns={'sum': 'cust_odsc_sum', 'count_nonzero': 'cust_odsc_cnt'})
    features.append(feat_3)
    
    feat_4 = tranx.groupby('customer_id')['brand'].agg([pd.Series.nunique]).reset_index()
    feat_4 = feat_4.rename(columns={'nunique': 'cust_unq_brand'})
    features.append(feat_4)
    
    feat_5 = tranx.groupby('customer_id')['brand_type'].agg([pd.Series.nunique]).reset_index()
    feat_5 = feat_5.rename(columns={'nunique': 'cust_unq_brand_typ'})
    features.append(feat_5)
    
    feat_6 = tranx.groupby('customer_id')['category'].agg([pd.Series.nunique]).reset_index()
    feat_6 = feat_6.rename(columns={'nunique': 'cust_unq_category'})
    features.append(feat_6)
    
    feat_7 = tranx.groupby('customer_id')['coupon_discount'].agg(['sum', np.count_nonzero]).reset_index()
    feat_7 = feat_7.rename(columns={'sum': 'cust_cdsc_sum', 'count_nonzero': 'cust_cdsc_cnt'})
    features.append(feat_7)
    
    feat_8 = tranx.groupby('customer_id')['selling_price'].sum().reset_index()
    feat_8 = feat_8.rename(columns={'selling_price': 'cust_sum_price'})
    features.append(feat_8)
    
    for feat in features[1:]:
        profile = profile.merge(feat, on='customer_id', how='outer')
    
    profile = profile.fillna(-1)
    
    driver = pd.read_csv(driver_path)[['id', 'customer_id']]
    data = driver.merge(profile, on='customer_id', how='left').drop('customer_id', axis=1)
    data = data.fillna(-1)
    
    data.to_csv(output_path, index=False)
    return data


def create_coupon_features(coupon_mapping_path, item_path, driver_path, output_path):
    """Create coupon-based features."""
    driver = pd.read_csv(driver_path)[['id', 'coupon_id']]
    
    coupon = pd.read_csv(coupon_mapping_path)
    item = pd.read_csv(item_path)
    coupon = coupon.merge(item, on='item_id')
    
    coupon = encode_categorical_features(coupon, ['brand', 'brand_type', 'category'])
    
    coupon['cnt_coup_item'] = coupon.groupby('coupon_id')['item_id'].transform(pd.Series.nunique)
    coupon['cnt_coup_brand'] = coupon.groupby('coupon_id')['brand'].transform(pd.Series.nunique)
    coupon['cnt_coup_category'] = coupon.groupby('coupon_id')['category'].transform(pd.Series.nunique)
    coupon['cnt_coup_brand_typ'] = coupon.groupby('coupon_id')['brand_type'].transform(pd.Series.nunique)
    
    coupon = coupon[['coupon_id', 'cnt_coup_item', 'cnt_coup_brand', 'cnt_coup_category', 'cnt_coup_brand_typ']]
    coupon = coupon.drop_duplicates(subset=['coupon_id'])
    
    data = driver.merge(coupon, on='coupon_id').drop('coupon_id', axis=1)
    data.to_csv(output_path, index=False)
    return data


def create_campaign_features(campaign_path, driver_path, output_path):
    """Create campaign-based features."""
    driver = pd.read_csv(driver_path)[['id', 'campaign_id']]
    campaign = pd.read_csv(campaign_path)
    
    campaign['start_date'] = pd.to_datetime(campaign['start_date'])
    campaign['end_date'] = pd.to_datetime(campaign['end_date'])
    campaign.loc[campaign['start_date'] > campaign['end_date'], 'end_date'] = campaign['start_date'] + pd.DateOffset(90)
    
    campaign['campaign_duration'] = (campaign['end_date'] - campaign['start_date']).dt.days
    campaign['campaign_duration'] = campaign['campaign_duration'].fillna(90)
    
    data = driver.merge(campaign[['campaign_id', 'campaign_duration']], on='campaign_id')
    data = data.drop('campaign_id', axis=1)
    
    data.to_csv(output_path, index=False)
    return data


def create_coupon_spend_features(coupon_mapping_path, transaction_path, driver_path, output_path):
    """Create coupon spending profile features."""
    driver = pd.read_csv(driver_path)[['id', 'coupon_id']]
    
    coupon = pd.read_csv(coupon_mapping_path)
    tranx = pd.read_csv(transaction_path)
    
    coupon_tranx = tranx.merge(coupon, on='item_id')
    
    features = coupon_tranx.groupby('coupon_id').agg({
        'quantity': ['sum', 'mean', 'std'],
        'selling_price': ['sum', 'mean', 'std'],
        'coupon_discount': ['sum', 'mean', 'std']
    }).reset_index()
    
    features.columns = ['coupon_id'] + [f'coup_{col[0]}_{col[1]}' for col in features.columns[1:]]
    features = features.fillna(0)
    
    data = driver.merge(features, on='coupon_id').drop('coupon_id', axis=1)
    data.to_csv(output_path, index=False)
    return data


def create_count_features(transaction_path, driver_path, output_path):
    """Create count-based features."""
    driver = pd.read_csv(driver_path)[['id', 'customer_id', 'coupon_id']]
    tranx = pd.read_csv(transaction_path)
    
    customer_counts = tranx.groupby('customer_id').agg({
        'item_id': 'nunique',
        'brand': 'nunique',
        'category': 'nunique'
    }).reset_index()
    customer_counts.columns = ['customer_id', 'cust_item_count', 'cust_brand_count', 'cust_category_count']
    
    data = driver.merge(customer_counts, on='customer_id', how='left')
    data = data.drop(['customer_id', 'coupon_id'], axis=1)
    data = data.fillna(0)
    
    data.to_csv(output_path, index=False)
    return data


def create_transaction_brand_features(transaction_path, item_path, driver_path, output_path):
    """Create transaction brand features."""
    driver = pd.read_csv(driver_path)[['id', 'customer_id']]
    tranx = pd.read_csv(transaction_path)
    item = pd.read_csv(item_path)
    
    tranx = tranx.merge(item, on='item_id')
    
    brand_features = tranx.groupby('customer_id').agg({
        'brand': 'nunique',
        'brand_type': 'nunique',
        'selling_price': ['sum', 'mean'],
        'quantity': ['sum', 'mean']
    }).reset_index()
    
    brand_features.columns = ['customer_id'] + [f'brand_{col[0]}_{col[1]}' if col[1] else col[0] for col in brand_features.columns[1:]]
    
    data = driver.merge(brand_features, on='customer_id', how='left').drop('customer_id', axis=1)
    data = data.fillna(0)
    
    data.to_csv(output_path, index=False)
    return data


def create_transaction_category_features(transaction_path, item_path, driver_path, output_path):
    """Create transaction category features."""
    driver = pd.read_csv(driver_path)[['id', 'customer_id']]
    tranx = pd.read_csv(transaction_path)
    item = pd.read_csv(item_path)
    
    tranx = tranx.merge(item, on='item_id')
    
    category_features = tranx.groupby('customer_id').agg({
        'category': 'nunique',
        'selling_price': ['sum', 'mean'],
        'quantity': ['sum', 'mean']
    }).reset_index()
    
    category_features.columns = ['customer_id'] + [f'cat_{col[0]}_{col[1]}' if col[1] else col[0] for col in category_features.columns[1:]]
    
    data = driver.merge(category_features, on='customer_id', how='left').drop('customer_id', axis=1)
    data = data.fillna(0)
    
    data.to_csv(output_path, index=False)
    return data


def create_transaction_item_features(transaction_path, driver_path, output_path):
    """Create transaction item features."""
    driver = pd.read_csv(driver_path)[['id', 'customer_id']]
    tranx = pd.read_csv(transaction_path)
    
    item_features = tranx.groupby('customer_id').agg({
        'item_id': 'nunique',
        'selling_price': ['sum', 'mean'],
        'quantity': ['sum', 'mean']
    }).reset_index()
    
    item_features.columns = ['customer_id'] + [f'item_{col[0]}_{col[1]}' if col[1] else col[0] for col in item_features.columns[1:]]
    
    data = driver.merge(item_features, on='customer_id', how='left').drop('customer_id', axis=1)
    data = data.fillna(0)
    
    data.to_csv(output_path, index=False)
    return data


def jaccard_similarity(set1, set2):
    """Calculate Jaccard similarity between two sets."""
    set1, set2 = set(set1), set(set2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0


def create_similarity_features(transaction_path, coupon_mapping_path, item_path, driver_path, output_path):
    """Create similarity features between customer preferences and coupon items."""
    driver = pd.read_csv(driver_path)
    
    item = pd.read_csv(item_path)
    item = encode_categorical_features(item, ['brand', 'brand_type', 'category'])
    
    tranx = pd.read_csv(transaction_path)
    tranx = tranx[tranx['coupon_discount'] == 0]
    tranx = tranx.merge(item, on='item_id')
    
    coupon = pd.read_csv(coupon_mapping_path)
    coupon = coupon.merge(item, on='item_id')
    
    cust_item = tranx.groupby('customer_id')['item_id'].apply(list).reset_index().rename(columns={'item_id': 'cm_ilist'})
    coup_item = coupon.groupby('coupon_id')['item_id'].apply(list).reset_index().rename(columns={'item_id': 'cp_ilist'})
    
    cust_brand = tranx.groupby('customer_id')['brand'].apply(list).reset_index().rename(columns={'brand': 'cm_blist'})
    coup_brand = coupon.groupby('coupon_id')['brand'].apply(list).reset_index().rename(columns={'brand': 'cp_blist'})
    
    cust_cat = tranx.groupby('customer_id')['category'].apply(list).reset_index().rename(columns={'category': 'cm_clist'})
    coup_cat = coupon.groupby('coupon_id')['category'].apply(list).reset_index().rename(columns={'category': 'cp_clist'})
    
    driver = driver.merge(cust_item, on=['customer_id'])
    driver = driver.merge(coup_item, on=['coupon_id'])
    driver = driver.merge(cust_brand, on=['customer_id'])
    driver = driver.merge(coup_brand, on=['coupon_id'])
    driver = driver.merge(cust_cat, on=['customer_id'])
    driver = driver.merge(coup_cat, on=['coupon_id'])
    
    driver['over_1'] = driver[['cm_ilist', 'cp_ilist']].apply(lambda x: jaccard_similarity(*x), axis=1)
    driver['over_2'] = driver[['cm_blist', 'cp_blist']].apply(lambda x: jaccard_similarity(*x), axis=1)
    driver['over_3'] = driver[['cm_clist', 'cp_clist']].apply(lambda x: jaccard_similarity(*x), axis=1)
    
    driver = driver[['id', 'over_1', 'over_2', 'over_3']]
    driver.to_csv(output_path, index=False)
    return driver


def create_time_features(transaction_path, coupon_mapping_path, campaign_path, driver_path, output_path):
    """Create time-based transaction features."""
    driver = pd.read_csv(driver_path)
    
    item = pd.read_csv(coupon_mapping_path)
    
    campaign = pd.read_csv(campaign_path)
    campaign['start_date'] = pd.to_datetime(campaign['start_date'])
    campaign['end_date'] = pd.to_datetime(campaign['end_date'])
    campaign.loc[campaign['start_date'] > campaign['end_date'], 'end_date'] = campaign['start_date'] + pd.DateOffset(90)
    campaign = campaign[['campaign_id', 'start_date']]
    campaign = campaign.set_index('campaign_id')['start_date'].to_dict()
    
    tranx = pd.read_csv(transaction_path)
    tranx['date'] = pd.to_datetime(tranx['date'])
    tranx = tranx.merge(item, on='item_id')
    
    def summary(driver_subset, tranx_subset, campaign_id, date):
        sub_driver = driver_subset[driver_subset['campaign_id'] == campaign_id]
        sub_tranx = tranx_subset[tranx_subset['date'] < date]
        
        sub_cust_coup = sub_tranx.groupby(['customer_id', 'coupon_id'])
        sub_cust_coup = sub_cust_coup[['quantity', 'selling_price', 'coupon_discount']].sum()
        sub_cust_coup = sub_cust_coup.reset_index()
        sub_cust_coup.columns = ['customer_id', 'coupon_id'] + ['cust_coup_qty', 'cust_coup_prc', 'cust_coup_cdsc']
        
        sub_cust = sub_tranx.groupby(['customer_id'])
        sub_cust = sub_cust[['quantity', 'selling_price', 'coupon_discount']].sum()
        sub_cust = sub_cust.reset_index()
        sub_cust.columns = ['customer_id'] + ['cust_qty', 'cust_prc', 'cust_cdsc']
        
        sub_coup = sub_tranx.groupby(['coupon_id'])
        sub_coup = sub_coup[['quantity', 'selling_price', 'coupon_discount']].sum()
        sub_coup = sub_coup.reset_index()
        sub_coup.columns = ['coupon_id'] + ['coup_qty', 'coup_prc', 'coup_cdsc']
        
        sub_driver = sub_driver.merge(sub_cust_coup, on=['customer_id', 'coupon_id'], how='left')
        sub_driver = sub_driver.merge(sub_cust, on=['customer_id'], how='left')
        sub_driver = sub_driver.merge(sub_coup, on=['coupon_id'], how='left')
        sub_driver = sub_driver.fillna(0)
        
        return sub_driver
    
    outputs = []
    for camp, date in campaign.items():
        outputs.append(summary(driver, tranx, camp, date))
    
    driver = reduce(lambda x, y: x.append(y), outputs)
    driver = driver.drop(['campaign_id', 'customer_id', 'coupon_id'], axis=1)
    
    driver.to_csv(output_path, index=False)
    return driver
