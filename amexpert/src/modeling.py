import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score


def create_lightgbm_dataset(data, labels=None, categorical_features=None):
    """Create LightGBM dataset."""
    params = {}
    if labels is not None:
        params['label'] = labels
    if categorical_features is not None:
        params['categorical_feature'] = categorical_features
    params['feature_name'] = list(data.columns)
    return lgb.Dataset(data.values, **params)


def get_lightgbm_params(model_version=1):
    """Get LightGBM parameters for different model versions."""
    base_params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'learning_rate': 0.01,
        'max_bin': 256,
        'subsample': 0.5,
        'subsample_freq': 1,
        'colsample_bytree': 0.5,
        'min_split_gain': 0.0,
        'min_sum_hessian_in_leaf': 1,
        'num_threads': 3,
        'device_type': 'cpu',
        'seed': 42,
        'verbose': 0,
        'metric': 'auc'
    }
    
    if model_version == 1:
        base_params.update({
            'num_leaves': 24,
            'max_depth': 4
        })
    elif model_version == 2:
        base_params.update({
            'num_leaves': 48,
            'max_depth': 6
        })
    elif model_version == 3:
        base_params.update({
            'num_leaves': 64,
            'max_depth': 8
        })
    
    else:
        raise ValueError('model_version must be 1, 2, or 3')
    return base_params


def train_lightgbm_model(train_data, train_labels, valid_data, valid_labels, 
                        model_version=1, categorical_features=None, 
                        num_boost_round=2000, early_stopping_rounds=200, params_override=None):
    """Train LightGBM model."""
    train_matrix = create_lightgbm_dataset(train_data, train_labels, categorical_features)
    valid_matrix = create_lightgbm_dataset(valid_data, valid_labels, categorical_features)
    
    if num_boost_round < 1 or early_stopping_rounds < 0:
        raise ValueError('Boosting rounds must be positive; early stopping must be nonnegative')
    params = get_lightgbm_params(model_version)
    params.update(params_override or {})
    callbacks = [lgb.log_evaluation(25)]
    if early_stopping_rounds:
        callbacks.append(lgb.early_stopping(early_stopping_rounds))
    model = lgb.train(params, train_matrix, valid_sets=[train_matrix, valid_matrix],
                      valid_names=['train', 'validation'],
                      num_boost_round=num_boost_round, callbacks=callbacks)
    print(f"Validation ROC AUC: {roc_auc_score(valid_labels, model.predict(valid_data)):.6f}")
    return model


def get_feature_importance(model, feature_names, importance_type='gain'):
    """Get feature importance from trained model."""
    importance = model.feature_importance(importance_type=importance_type)
    importance_df = pd.DataFrame(importance, columns=['importance'])
    importance_df['feature'] = feature_names
    maximum = importance_df['importance'].max()
    if maximum > 0:
        importance_df['importance'] = importance_df['importance'] / maximum
    importance_df = importance_df[['feature', 'importance']]
    importance_df = importance_df.sort_values(by='importance', ascending=False)
    importance_df = importance_df.reset_index(drop=True)
    return importance_df


def predict_and_save(model, test_data, test_ids, output_path):
    """Make predictions and save results."""
    predictions = model.predict(test_data.values)
    
    results = pd.DataFrame({
        'id': test_ids,
        'redemption_status': predictions
    })
    
    results.to_csv(output_path, index=False)
    return results


def train_model_v1(train_path, valid_path, test_path, model_save_path, score_save_path, **training_options):
    """Train Model Version 1 - Basic LightGBM without categorical features."""
    train_data = pd.read_csv(train_path)
    train_labels = train_data['redemption_status'].values
    train_data = train_data.drop(['id', 'customer_id', 'redemption_status'], axis=1)
    
    valid_data = pd.read_csv(valid_path)
    valid_labels = valid_data['redemption_status'].values
    valid_data = valid_data.drop(['id', 'customer_id', 'redemption_status'], axis=1)
    
    test_data = pd.read_csv(test_path)
    test_ids = test_data['id']
    test_data = test_data.drop(['id', 'customer_id'], axis=1)
    
    model = train_lightgbm_model(train_data, train_labels, valid_data, valid_labels, model_version=1, **training_options)
    model.save_model(model_save_path)
    
    importance = get_feature_importance(model, train_data.columns)
    print("Top 10 Most Important Features:")
    print(importance.head(10))
    
    results = predict_and_save(model, test_data, test_ids, score_save_path)
    return model, results


def train_model_v2(train_path, valid_path, test_path, model_save_path, score_save_path, **training_options):
    """Train Model Version 2 - LightGBM with customer_id as categorical feature."""
    train_data = pd.read_csv(train_path)
    train_labels = train_data['redemption_status'].values
    train_data = train_data.drop(['id', 'redemption_status'], axis=1)
    
    valid_data = pd.read_csv(valid_path)
    valid_labels = valid_data['redemption_status'].values
    valid_data = valid_data.drop(['id', 'redemption_status'], axis=1)
    
    test_data = pd.read_csv(test_path)
    test_ids = test_data['id']
    test_data = test_data.drop(['id'], axis=1)
    
    categorical_features = ['customer_id']
    model = train_lightgbm_model(train_data, train_labels, valid_data, valid_labels, 
                               model_version=2, categorical_features=categorical_features, **training_options)
    model.save_model(model_save_path)
    
    importance = get_feature_importance(model, train_data.columns)
    print("Top 10 Most Important Features:")
    print(importance.head(10))
    
    results = predict_and_save(model, test_data, test_ids, score_save_path)
    return model, results


def train_model_v3(train_path, valid_path, test_path, model_save_path, score_save_path, **training_options):
    """Train Model Version 3 - LightGBM with deeper trees."""
    train_data = pd.read_csv(train_path)
    train_labels = train_data['redemption_status'].values
    train_data = train_data.drop(['id', 'redemption_status'], axis=1)
    
    valid_data = pd.read_csv(valid_path)
    valid_labels = valid_data['redemption_status'].values
    valid_data = valid_data.drop(['id', 'redemption_status'], axis=1)
    
    test_data = pd.read_csv(test_path)
    test_ids = test_data['id']
    test_data = test_data.drop(['id'], axis=1)
    
    categorical_features = ['customer_id']
    model = train_lightgbm_model(train_data, train_labels, valid_data, valid_labels, 
                               model_version=3, categorical_features=categorical_features, **training_options)
    model.save_model(model_save_path)
    
    importance = get_feature_importance(model, train_data.columns)
    print("Top 10 Most Important Features:")
    print(importance.head(10))
    
    results = predict_and_save(model, test_data, test_ids, score_save_path)
    return model, results
