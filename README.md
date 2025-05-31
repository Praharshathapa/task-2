ML Classification Pipeline
A robust, production-ready machine learning classification pipeline designed to handle complex datasets with preprocessing challenges, including infinite values, missing data, and mixed data types.

Features
Comprehensive Data Preprocessing: Handles infinite values, missing data, and mixed data types
Multiple Model Support: Logistic Regression, Random Forest, and Support Vector Machine
Automated Hyperparameter Tuning: GridSearchCV with cross-validation
Robust Error Handling: Multiple fallback mechanisms and detailed logging
Performance Metrics: Accuracy, AUROC, Sensitivity, Specificity, and F1-score
Prediction Generation: Probability predictions for training, test, and blind test sets
CSV Export: Automated saving of predictions and results
Installation
Requirements
bash
pip install pandas numpy scikit-learn
Dependencies
pandas >= 1.3.0
numpy >= 1.21.0
scikit-learn >= 1.0.0
Quick Start
python
from ml_pipeline import MLClassificationPipeline

# Initialize pipeline
pipeline = MLClassificationPipeline()

# Run complete pipeline
results_df, predictions = pipeline.run_complete_pipeline(
    train_path='train_set.csv',
    test_path='test_set.csv', 
    blind_path='blinded_test_set.csv',
    target_column='CLASS',
    id_column='ID',  # Optional
    output_dir='./'
)
Data Requirements
Input Files
Training Set (train_set.csv): Contains features and target variable
Test Set (test_set.csv): Contains features for evaluation
Blind Test Set (blinded_test_set.csv): Contains features for final predictions
Data Format
CSV format with headers
Target column should be named consistently (e.g., 'CLASS')
Optional ID column for tracking samples
Mixed data types supported (numeric and categorical)
Example Data Structure
ID,feature1,feature2,feature3,categorical_feature,CLASS
1,1.5,2.3,inf,A,0
2,2.1,-inf,1.2,B,1
3,3.2,4.5,2.1,A,0
Usage Examples
Basic Usage
python
pipeline = MLClassificationPipeline()

# Load data
train_data, test_data, blind_data = pipeline.load_data(
    'train_set.csv', 
    'test_set.csv', 
    'blinded_test_set.csv'
)

# Preprocess data
X_train, X_test, X_blind, y_train = pipeline.preprocess_data('CLASS')

# Setup and train models
pipeline.setup_models()
pipeline.train_and_tune_models(X_train, y_train)

# Generate predictions
predictions = pipeline.generate_predictions(X_train, X_test, X_blind)

# Save results
pipeline.save_predictions(predictions)
Advanced Configuration
python
# Custom model parameters
pipeline.models['Random Forest'] = (
    RandomForestClassifier(random_state=42),
    {
        'n_estimators': [100, 300, 500],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    }
)

# Custom cross-validation
pipeline.train_and_tune_models(X_train, y_train, cv_folds=5)
Pipeline Components
1. Data Loading
Reads CSV files into pandas DataFrames
Validates data structure and types
Reports data dimensions and column information
2. Data Preprocessing
Infinite Value Handling
Detects positive and negative infinite values
Replaces with percentile-based substitutes
Multiple validation layers with fallback mechanisms
Missing Value Imputation
Median imputation for numeric features
Constant imputation for categorical features
Custom SafeImputer class for robust handling
Feature Scaling
RobustScaler for numeric features (handles outliers)
OneHotEncoder for categorical features
ColumnTransformer for unified preprocessing
3. Model Training
Supported Models
Logistic Regression: Fast, interpretable linear model
Random Forest: Ensemble method, handles non-linearity
Support Vector Machine: Powerful for complex decision boundaries
Hyperparameter Tuning
GridSearchCV with stratified cross-validation
ROC-AUC optimization
Parallel processing for efficiency
4. Model Evaluation
Metrics Calculated
Accuracy: Overall correctness
AUROC: Area Under ROC Curve
Sensitivity: True positive rate (recall)
Specificity: True negative rate
F1-Score: Harmonic mean of precision and recall
Validation Strategy
Training/validation split for model selection
Cross-validation for hyperparameter tuning
Separate test set for final evaluation
5. Prediction Generation
Probability predictions for all datasets
Class probability distributions
Optional ID column preservation
Output Files
Prediction Files
{model_name}_train_predictions.csv: Training set predictions
{model_name}_test_predictions.csv: Test set predictions
{model_name}_blind_predictions.csv: Blind test predictions
File Format
csv
ID,Class_0_Prob,Class_1_Prob
1,0.8234,0.1766
2,0.2145,0.7855
Error Handling
Robust Design
Multiple validation checkpoints
Fallback mechanisms for preprocessing failures
Graceful degradation with informative logging
Common Issues Addressed
Infinite values in numeric data
Memory constraints with large datasets
Convergence issues in optimization
Data type inconsistencies
Performance Considerations
Memory Optimization
Efficient data type handling
Chunked processing for large datasets
Memory-conscious preprocessing
Computational Efficiency
Parallel hyperparameter tuning
Optimized cross-validation
Early stopping mechanisms
Troubleshooting
Common Issues
Infinite Values Error
ValueError: Input X contains infinity or a value too large for dtype('float64')
Solution: The pipeline automatically handles this with enhanced infinite value replacement.
Memory Issues
MemoryError: Unable to allocate array
Solution: Reduce hyperparameter grid size or use data sampling.
Convergence Warnings
ConvergenceWarning: lbfgs failed to converge
Solution: Pipeline automatically falls back to default parameters.
Debug Mode
python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run pipeline with detailed logging
pipeline.run_complete_pipeline(...)
Contributing
Code Style
Follow PEP 8 guidelines
Add docstrings for all methods
Include error handling for all operations
Testing
Unit tests for all components
Integration tests for complete pipeline
Edge case validation
License
This project is licensed under the MIT License - see the LICENSE file for details.

Changelog
Version 1.1.0
Enhanced infinite value handling
Added SafeImputer class
Improved error handling and logging
Added emergency cleanup mechanisms
Version 1.0.0
Initial implementation
Basic preprocessing pipeline
Multi-model support
Automated hyperparameter tuning
Support
For issues and questions:

Check the troubleshooting section
Review error logs for specific issues
Ensure data format matches requirements
Acknowledgments
Built with scikit-learn ecosystem
Inspired by production ML best practices
Designed for research and commercial applications
