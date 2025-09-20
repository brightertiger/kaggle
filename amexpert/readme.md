# AmExpert Coupon Redemption Prediction

A comprehensive machine learning solution for predicting coupon redemption behavior in the AmExpert competition. This project demonstrates advanced feature engineering, ensemble modeling, and systematic approach to solving a real-world business problem.

## 🎯 Problem Statement

The challenge was to predict whether customers would redeem coupons based on their historical behavior, demographic information, and coupon characteristics. This is a binary classification problem with significant business implications for targeted marketing campaigns.

## 📊 Dataset Overview

The dataset consists of several key components:

- **Customer Demographics**: Age range, marital status, family size, number of children
- **Transaction History**: Purchase patterns, discount usage, brand preferences
- **Coupon Information**: Item mappings, discount amounts, campaign details
- **Campaign Data**: Start/end dates, campaign durations
- **Item Data**: Brand types, categories, pricing information

### Data Statistics
- **Training Set**: 78,369 samples
- **Validation Set**: 22,606 samples (Campaign ID 13)
- **Test Set**: 50,226 samples
- **Redemption Rate**: ~2.5% (highly imbalanced)

## 🏗️ Solution Architecture

### 1. Data Preprocessing
- **Date Format Standardization**: Convert DD/MM/YY to YYYY-MM-DD format
- **Categorical Encoding**: Label encoding for categorical variables
- **Train/Validation Split**: Time-based split using Campaign ID 13 as validation

### 2. Feature Engineering Pipeline

#### Customer Features
- **Transaction Patterns**: Number of transactions, unique items purchased
- **Spending Behavior**: Average price, total spending, discount usage
- **Brand/Category Diversity**: Unique brands, categories, brand types
- **Discount Preferences**: Other discount vs coupon discount patterns

#### Coupon Features
- **Item Diversity**: Number of unique items, brands, categories per coupon
- **Brand Type Distribution**: Distribution across different brand types
- **Category Coverage**: Breadth of categories covered

#### Campaign Features
- **Duration Analysis**: Campaign length and timing
- **Seasonal Patterns**: Time-based campaign characteristics

#### Transaction Features
- **Brand Preferences**: Historical brand interaction patterns
- **Category Preferences**: Category-wise purchase behavior
- **Item-Level Patterns**: Individual item purchase history

#### Advanced Features
- **Similarity Metrics**: Jaccard similarity between customer preferences and coupon items
- **Time-Based Features**: Historical transaction patterns before campaign start
- **Spending Profiles**: Customer-coupon interaction history

### 3. Modeling Strategy

#### Model Architecture
Three LightGBM models with different configurations:

1. **Model V1**: Basic LightGBM without categorical features
   - 24 leaves, depth 4
   - Focus on numerical features only

2. **Model V2**: LightGBM with customer_id as categorical feature
   - 48 leaves, depth 6
   - Leverages customer-specific patterns

3. **Model V3**: Deep LightGBM with enhanced capacity
   - 64 leaves, depth 8
   - Maximum model complexity

#### Hyperparameters
```python
{
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'learning_rate': 0.01,
    'subsample': 0.5,
    'colsample_bytree': 0.5,
    'colsample_bylevel': 0.5,
    'metric': 'auc',
    'num_boost_round': 2000,
    'early_stopping_rounds': 200
}
```

### 4. Ensemble Strategy

**Rank Blending**: Combines predictions using rank averaging
- Converts predictions to ranks to handle scale differences
- Averages ranks across models
- Converts back to probability space

This approach is robust to:
- Different prediction scales
- Model-specific biases
- Outlier predictions

## 🚀 Key Innovations

### 1. Comprehensive Feature Engineering
- **Multi-level Aggregation**: Customer, coupon, and campaign level features
- **Temporal Features**: Time-based transaction patterns
- **Similarity Metrics**: Jaccard similarity for preference matching
- **Interaction Features**: Customer-coupon specific patterns

### 2. Advanced Similarity Computation
```python
def jaccard_similarity(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0
```

### 3. Time-Aware Feature Engineering
- Historical transaction patterns before campaign start
- Customer-coupon interaction history
- Temporal discount usage patterns

### 4. Robust Ensemble Method
- Rank-based blending for scale invariance
- Multiple model architectures for diversity
- Early stopping for generalization

## 📈 Results & Performance

### Model Performance
- **Model V1**: Strong baseline performance
- **Model V2**: Improved with categorical features
- **Model V3**: Best individual model performance

### Ensemble Benefits
- **Correlation Analysis**: Models show moderate correlation (~0.7-0.8)
- **Diversity**: Different architectures capture different patterns
- **Robustness**: Ensemble reduces overfitting risk

### Business Impact
- **Targeted Marketing**: Identify high-probability redemption customers
- **Cost Optimization**: Focus resources on likely converters
- **Campaign Effectiveness**: Improve coupon design based on insights

## 🛠️ Technical Implementation

### Code Structure
```
amexpert/
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py    # Data cleaning and preparation
│   ├── feature_engineering.py  # Feature creation pipeline
│   ├── modeling.py             # Model training and evaluation
│   ├── ensemble.py             # Prediction blending
│   └── pipeline.py             # Main pipeline orchestration
├── main.py                     # Entry point
├── example_usage.py            # Usage example
├── requirements.txt            # Dependencies
└── README.md                   # Documentation
```

### Usage
```bash
# Install dependencies
pip install -r requirements.txt

# Run complete pipeline
python main.py --step all

# Run specific steps
python main.py --step preprocess
python main.py --step features
python main.py --step merge
python main.py --step train
python main.py --step blend

# Simple example
python example_usage.py
```

### Key Functions
- `AmExpertPipeline`: Main pipeline class
- `create_customer_features()`: Customer behavior features
- `create_similarity_features()`: Preference matching
- `train_lightgbm_model()`: Model training
- `rank_blend_predictions()`: Ensemble blending

## 🔍 Feature Importance Analysis

Top features typically include:
1. **Customer Transaction Patterns**: Historical purchase behavior
2. **Similarity Metrics**: Preference alignment with coupon items
3. **Discount Usage**: Historical coupon vs other discount patterns
4. **Brand/Category Preferences**: Alignment with coupon characteristics
5. **Temporal Features**: Time-based transaction patterns

## 💡 Business Insights

### Customer Segmentation
- **High Redemption Probability**: Customers with strong brand/category alignment
- **Low Redemption Probability**: Customers with mismatched preferences
- **Opportunity Customers**: Customers with potential but no historical coupon usage

### Coupon Optimization
- **Item Selection**: Focus on items matching customer preferences
- **Brand Alignment**: Ensure brand consistency with customer history
- **Category Coverage**: Balance breadth vs specificity

### Campaign Timing
- **Seasonal Patterns**: Optimal timing for different customer segments
- **Duration Optimization**: Campaign length vs redemption probability
- **Frequency Analysis**: Optimal campaign frequency per customer

## 🎓 Learning Outcomes

This project demonstrates:

1. **End-to-End ML Pipeline**: From data preprocessing to model deployment
2. **Advanced Feature Engineering**: Multi-level aggregation and similarity metrics
3. **Ensemble Methods**: Robust blending strategies
4. **Business Understanding**: Translating technical solutions to business value
5. **Code Organization**: Modular, maintainable, and scalable code structure

## 🔮 Future Enhancements

### Model Improvements
- **Deep Learning**: Neural networks for complex pattern recognition
- **Time Series Models**: LSTM/GRU for temporal patterns
- **Graph Neural Networks**: Customer-item relationship modeling

### Feature Engineering
- **External Data**: Weather, economic indicators, seasonal trends
- **Advanced Similarity**: Cosine similarity, collaborative filtering
- **Feature Selection**: Automated feature importance and selection

### Business Applications
- **Real-time Scoring**: Online prediction for dynamic campaigns
- **A/B Testing**: Framework for campaign optimization
- **Customer Lifetime Value**: Integration with CLV models

## 📚 References

- LightGBM Documentation: https://lightgbm.readthedocs.io/
- Scikit-learn Documentation: https://scikit-learn.org/
- Pandas Documentation: https://pandas.pydata.org/

## 👨‍💻 Author

This project was developed as part of a data science competition, demonstrating advanced machine learning techniques for business applications.

---

*This README serves as both technical documentation and a portfolio piece showcasing comprehensive data science capabilities.*
