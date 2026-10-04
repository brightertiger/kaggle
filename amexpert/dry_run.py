#!/usr/bin/env python3
"""Exercise the original feature families and all models with synthetic CSVs."""
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.ensemble import geometric_mean_blend, rank_blend_predictions, weighted_blend_predictions
from src.pipeline import AmExpertPipeline

ROOT = Path(__file__).resolve().parent


def make_sample(data_dir):
    """Write competition schemas, including sparse and cold-start cases."""
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    customers = np.arange(1, 25)
    pd.DataFrame({
        'customer_id': customers[:-1],
        'age_range': ['26-35', '36-45'] * 11 + ['46-55'],
        'marital_status': ['Married', None] * 11 + ['Single'],
        'rented': customers[:-1] % 2,
        'family_size': ['2', '3', '5+'] * 7 + ['1', '2'],
        'no_of_children': [None, '1', '3+'] * 7 + ['0', '2'],
        'income_bracket': customers[:-1] % 5 + 1,
    }).to_csv(data_dir / 'customer_demographics.csv', index=False)
    pd.DataFrame({
        'item_id': np.arange(1, 13), 'brand': np.tile([1, 2, 3], 4),
        'brand_type': ['Established', 'Local'] * 6,
        'category': ['Grocery', 'Pharmaceutical', 'Packaged Meat'] * 4,
    }).to_csv(data_dir / 'item_data.csv', index=False)
    mapping = [(coupon, item) for coupon in range(1, 7)
               for item in [coupon, coupon + 1]]
    mapping.append((7, 12))  # A mapped item with no transactions.
    pd.DataFrame(mapping, columns=['coupon_id', 'item_id']).to_csv(
        data_dir / 'coupon_item_mapping.csv', index=False)
    transactions = []
    for customer in customers[:-1]:  # Last customer has no history or demographics.
        for day in range(8):
            transactions.append({
                'date': f'2013-01-{day + 1:02d}', 'customer_id': customer,
                'item_id': int(rng.integers(1, 8)), 'quantity': int(rng.integers(1, 4)),
                'selling_price': float(10 * customer + day),
                'other_discount': float(-(customer % 3)),
                'coupon_discount': -2.0 if customer == 23 or day % 3 == 0 else 0.0,
            })
    # Post-campaign transaction: temporal features must exclude this purchase.
    transactions.append(dict(date='2014-01-01', customer_id=1, item_id=1,
                             quantity=1000, selling_price=1000.0,
                             other_discount=0.0, coupon_discount=0.0))
    pd.DataFrame(transactions).to_csv(data_dir / 'customer_transaction_data.csv', index=False)
    pd.DataFrame({
        'campaign_id': [1, 13, 2], 'campaign_type': ['X', 'Y', 'X'],
        'start_date': ['01/02/13', '01/03/13', '01/04/13'],
        'end_date': ['28/02/13', '31/03/13', '01/03/13'],
    }).to_csv(data_dir / 'campaign_data.csv', index=False)
    rows = []
    for campaign in [1, 13]:
        for customer in customers:
            for coupon in [1, 3, 7]:
                rows.append(dict(id=len(rows) + 1, campaign_id=campaign,
                                 coupon_id=coupon, customer_id=customer,
                                 redemption_status=int(customer % 4 == 0 and coupon != 7)))
    pd.DataFrame(rows).to_csv(data_dir / 'train.csv', index=False)
    test = pd.DataFrame({
        'id': [208, 201, 207, 202, 206, 203, 205, 204],
        'campaign_id': 2, 'coupon_id': [1, 3, 7, 99, 1, 3, 7, 1],
        'customer_id': [4, 8, 12, 16, 20, 23, 24, 1],
    })
    test.to_csv(data_dir / 'test.csv', index=False)
    test[['id']].assign(redemption_status=0.0).to_csv(
        data_dir / 'sample_submission.csv', index=False)


def check_outputs(data_dir, output_dir, result):
    test = pd.read_csv(data_dir / 'test.csv')
    assert list(result.columns) == ['id', 'redemption_status']
    assert result['id'].tolist() == test['id'].tolist(), 'Lost or reordered test IDs'
    assert np.isfinite(result['redemption_status']).all()
    assert result['redemption_status'].between(0, 1).all()
    train = pd.read_csv(output_dir / 'model/train.csv')
    valid = pd.read_csv(output_dir / 'model/valid.csv')
    assert len(train) + len(valid) == len(pd.read_csv(data_dir / 'train.csv'))
    assert np.isfinite(train.to_numpy(dtype=float)).all()
    assert {'cust_num_tranx', 'cust_unq_item'}.issubset(train.columns)
    driver = pd.read_csv(output_dir / 'feature/driver.csv')
    temporal = pd.read_csv(output_dir / 'feature/tranx_time_feature.csv').set_index('id')
    raw = pd.read_csv(data_dir / 'customer_transaction_data.csv')
    expected = raw[(raw.customer_id == 1) & (raw.date < '2013-02-01')].quantity.sum()
    first_id = driver[(driver.customer_id == 1) & (driver.campaign_id == 1)].id.iloc[0]
    assert temporal.loc[first_id, 'cust_qty'] == expected, 'Temporal totals duplicated or leaked'
    sim = pd.read_csv(output_dir / 'feature/similarity.csv').set_index('id')
    cold_ids = driver[driver.customer_id.isin([23, 24])].id
    assert (sim.loc[cold_ids].to_numpy() == 0).all()
    for version in [1, 2, 3]:
        model = lgb.Booster(model_file=str(output_dir / f'model/lightgbm_v{version}.model'))
        assert model.num_trees() > 1, 'Sample must exercise actual boosting'
        assert model.feature_importance().sum() > 0
        matrix = pd.read_csv(output_dir / 'model/test.csv')[model.feature_name()]
        saved = pd.read_csv(output_dir / f'score/score_v{version}.csv')
        np.testing.assert_allclose(model.predict(matrix), saved.redemption_status)
    # Verify the alternative ensemble helpers also align shuffled rows by ID.
    paths = [output_dir / f'score/score_v{v}.csv' for v in [1, 2, 3]]
    shuffled = output_dir / 'score/shuffled.csv'
    pd.read_csv(paths[0]).iloc[::-1].to_csv(shuffled, index=False)
    source = pd.read_csv(paths[0]).redemption_status.to_numpy()
    for blend in [weighted_blend_predictions, geometric_mean_blend]:
        args = ([paths[0], shuffled], [1, 1]) if blend == weighted_blend_predictions else ([paths[0], shuffled],)
        out = blend(*args, output_path=output_dir / 'score/check.csv')
        np.testing.assert_allclose(out.redemption_status, source)
    out = rank_blend_predictions([paths[0], shuffled], output_dir / 'score/check.csv')
    np.testing.assert_allclose(out.redemption_status, pd.Series(source).rank(pct=True))
    duplicate = pd.read_csv(paths[0])
    duplicate.loc[1, 'id'] = duplicate.loc[0, 'id']
    duplicate.to_csv(shuffled, index=False)
    try:
        rank_blend_predictions([paths[0], shuffled], output_dir / 'score/check.csv')
    except ValueError:
        pass
    else:
        raise AssertionError('Duplicate prediction IDs were accepted')
    shuffled.unlink()
    (output_dir / 'score/check.csv').unlink()


def main():
    data_dir = ROOT / 'sample_data'
    output_dir = ROOT / 'dry_run_output'
    make_sample(data_dir)
    pipeline = AmExpertPipeline(
        data_dir=data_dir, feature_dir=output_dir / 'feature',
        model_dir=output_dir / 'model', score_dir=output_dir / 'score',
        num_boost_round=12, early_stopping_rounds=4,
        model_params={'device_type': 'cpu', 'num_threads': 1, 'min_data_in_leaf': 2,
                      'min_sum_hessian_in_leaf': 0.01, 'learning_rate': 0.1,
                      'verbosity': -1},
    )
    result = pipeline.run_full_pipeline()
    check_outputs(data_dir, output_dir, result)
    print(f'DRY RUN PASS: all feature families, three CPU LightGBM models, '
          f'and rank blend; {len(result)} submission rows; no stages skipped.')
    print(f'Submission: {output_dir / "score/submission.csv"}')


if __name__ == '__main__':
    main()
