import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class MLClassificationPipeline:
    def __init__(self):
        self.models = {}
        self.preprocessor = None
        self.label_encoder = None
        self.results = {}
        self.feature_names = None
        
    def load_data(self, train_path, test_path, blind_test_path):
        """Load the three datasets"""
        self.train_data = pd.read_csv(train_path)
        self.test_data = pd.read_csv(test_path)
        self.blind_test_data = pd.read_csv(blind_test_path)
        
        print("Data loaded successfully!")
        print(f"Training set shape: {self.train_data.shape}")
        print(f"Test set shape: {self.test_data.shape}")
        print(f"Blind test set shape: {self.blind_test_data.shape}")
        
        # Print column data types with existence check
        for col in self.train_data.columns:
            test_dtype = self.test_data[col].dtype if col in self.test_data.columns else 'Not present'
            blind_dtype = self.blind_test_data[col].dtype if col in self.blind_test_data.columns else 'Not present'
            print(f" - {col} (Train: {self.train_data[col].dtype}, Test: {test_dtype}, Blind: {blind_dtype})")
        
        return self.train_data, self.test_data, self.blind_test_data
    
    def replace_infinite_values(self, df):
        """Replace infinite values with appropriate substitutes in numeric columns"""
        df_copy = df.copy()
        numeric_cols = df_copy.select_dtypes(include=['int64', 'float64']).columns
        
        print(f"Processing {len(numeric_cols)} numeric columns for infinite values...")
        
        for col in numeric_cols:
            # Check for infinite values first
            inf_mask = np.isinf(df_copy[col])
            if inf_mask.any():
                print(f"Found {inf_mask.sum()} infinite values in column '{col}'")
                
                # Get finite values for calculation
                finite_values = df_copy[col][~inf_mask & df_copy[col].notna()]
                
                if len(finite_values) > 0:
                    # Use percentiles for more robust replacement
                    q99 = finite_values.quantile(0.99)
                    q01 = finite_values.quantile(0.01)
                    
                    # Replace positive infinity with value beyond 99th percentile
                    pos_inf_mask = df_copy[col] == np.inf
                    if pos_inf_mask.any():
                        replacement_val = q99 * 1.1 if q99 > 0 else 1e6
                        df_copy.loc[pos_inf_mask, col] = replacement_val
                        print(f"  Replaced {pos_inf_mask.sum()} +inf values with {replacement_val}")
                    
                    # Replace negative infinity with value beyond 1st percentile
                    neg_inf_mask = df_copy[col] == -np.inf
                    if neg_inf_mask.any():
                        replacement_val = q01 * 1.1 if q01 < 0 else -1e6
                        df_copy.loc[neg_inf_mask, col] = replacement_val
                        print(f"  Replaced {neg_inf_mask.sum()} -inf values with {replacement_val}")
                else:
                    # If no finite values, use default replacements
                    df_copy.loc[df_copy[col] == np.inf, col] = 1e6
                    df_copy.loc[df_copy[col] == -np.inf, col] = -1e6
                    print(f"  Used default replacements for column '{col}' (no finite values)")
        
        # Final check for any remaining infinite values
        remaining_inf = df_copy.isin([np.inf, -np.inf]).any().any()
        if remaining_inf:
            cols_with_inf = df_copy.columns[df_copy.isin([np.inf, -np.inf]).any()].tolist()
            print(f"WARNING: Columns still contain infinite values: {cols_with_inf}")
            
            # Force replacement of any remaining infinite values
            for col in cols_with_inf:
                df_copy[col] = df_copy[col].replace([np.inf, -np.inf], [1e6, -1e6])
        else:
            print("✓ All infinite values successfully replaced")
        
        # Also check for very large values that might cause issues
        for col in numeric_cols:
            max_val = df_copy[col].max()
            min_val = df_copy[col].min()
            if abs(max_val) > 1e10 or abs(min_val) > 1e10:
                print(f"WARNING: Column '{col}' has very large values (max: {max_val}, min: {min_val})")
                # Cap extremely large values
                df_copy[col] = np.clip(df_copy[col], -1e10, 1e10)
        
        return df_copy
    
    def preprocess_data(self, target_column, id_column=None):
        """Comprehensive data preprocessing"""
        print("\n=== DATA PREPROCESSING ===")
        
        # Handle infinite values in numeric columns
        print("Handling infinite values...")
        self.train_data = self.replace_infinite_values(self.train_data)
        self.test_data = self.replace_infinite_values(self.test_data)
        self.blind_test_data = self.replace_infinite_values(self.blind_test_data)
        
        # Separate features and target
        if id_column:
            X_train = self.train_data.drop([target_column, id_column], axis=1)
            X_test = self.test_data.drop([id_column], axis=1) if id_column in self.test_data.columns else self.test_data.copy()
            X_blind = self.blind_test_data.drop([id_column], axis=1) if id_column in self.blind_test_data.columns else self.blind_test_data.copy()
        else:
            X_train = self.train_data.drop([target_column], axis=1)
            X_test = self.test_data.copy()
            X_blind = self.blind_test_data.copy()
            
        y_train = self.train_data[target_column]
        
        # Store feature names
        self.feature_names = X_train.columns.tolist()
        
        # Identify numeric and categorical columns
        numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
        
        print(f"Numeric features: {len(numeric_features)}")
        print(f"Categorical features: {len(categorical_features)}")
        
        # Create custom imputer that handles infinite values
        class SafeImputer(SimpleImputer):
            def fit(self, X, y=None):
                # Ensure no infinite values before fitting
                X_clean = np.where(np.isinf(X), np.nan, X)
                return super().fit(X_clean, y)
            
            def transform(self, X):
                # Ensure no infinite values before transforming
                X_clean = np.where(np.isinf(X), np.nan, X)
                return super().transform(X_clean)
        
        # Create preprocessing pipelines with safe imputer
        numeric_transformer = Pipeline(steps=[
            ('safe_imputer', SafeImputer(strategy='median')),
            ('scaler', RobustScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine preprocessing steps
        if len(categorical_features) > 0:
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)
                ])
        else:
            # Only numeric features
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features)
                ])
        
        # Encode target variable if it's categorical
        if y_train.dtype == 'object':
            self.label_encoder = LabelEncoder()
            y_train = self.label_encoder.fit_transform(y_train)
            print(f"Target classes: {self.label_encoder.classes_}")
        
        # Final check before preprocessing
        print("Final infinite value check before preprocessing...")
        for df_name, df in [("X_train", X_train), ("X_test", X_test), ("X_blind", X_blind)]:
            inf_check = df.select_dtypes(include=[np.number]).isin([np.inf, -np.inf]).any().any()
            if inf_check:
                print(f"WARNING: {df_name} still contains infinite values!")
            else:
                print(f"✓ {df_name} clean of infinite values")
        
        # Fit preprocessor on training data
        try:
            X_train_processed = self.preprocessor.fit_transform(X_train)
            X_test_processed = self.preprocessor.transform(X_test)
            X_blind_processed = self.preprocessor.transform(X_blind)
            
            print(f"Processed training shape: {X_train_processed.shape}")
            print("✓ Preprocessing completed successfully!")
            
        except Exception as e:
            print(f"ERROR during preprocessing: {e}")
            print("Attempting emergency cleanup...")
            
            # Emergency cleanup - replace any remaining problematic values
            numeric_cols = X_train.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
                X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
                X_blind[col] = pd.to_numeric(X_blind[col], errors='coerce')
                
                # Replace infinite with NaN, then let imputer handle
                X_train[col] = X_train[col].replace([np.inf, -np.inf], np.nan)
                X_test[col] = X_test[col].replace([np.inf, -np.inf], np.nan)
                X_blind[col] = X_blind[col].replace([np.inf, -np.inf], np.nan)
            
            # Retry preprocessing
            X_train_processed = self.preprocessor.fit_transform(X_train)
            X_test_processed = self.preprocessor.transform(X_test)
            X_blind_processed = self.preprocessor.transform(X_blind)
            
            print(f"Processed training shape after cleanup: {X_train_processed.shape}")
            print("✓ Preprocessing completed after emergency cleanup!")
        
        return X_train_processed, X_test_processed, X_blind_processed, y_train
    
    def setup_models(self):
        """Initialize models with hyperparameter grids"""
        print("\n=== MODEL SETUP ===")
        
        # Logistic Regression
        lr_params = {
            'C': [0.01, 0.1, 1, 10],
            'solver': ['liblinear', 'lbfgs'],
            'max_iter': [1000]
        }
        
        # Random Forest
        rf_params = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        # Support Vector Machine
        svm_params = {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']
        }
        
        self.models = {
            'Logistic Regression': (LogisticRegression(random_state=42), lr_params),
            'Random Forest': (RandomForestClassifier(random_state=42), rf_params),
            'SVM': (SVC(probability=True, random_state=42), svm_params)
        }
        
        print("Models initialized:")
        for name in self.models.keys():
            print(f"- {name}")
    
    def train_and_tune_models(self, X_train, y_train, cv_folds=3):
        """Train models with hyperparameter tuning"""
        print(f"\n=== MODEL TRAINING & TUNING (CV={cv_folds}) ===")
        
        self.trained_models = {}
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        for name, (model, params) in self.models.items():
            print(f"\nTraining {name}...")
            
            try:
                # Hyperparameter tuning
                grid_search = GridSearchCV(
                    model, 
                    params, 
                    cv=cv, 
                    scoring='roc_auc',
                    n_jobs=-1,
                    verbose=0,
                    error_score='raise'
                )
                
                grid_search.fit(X_train, y_train)
                
                self.trained_models[name] = grid_search.best_estimator_
                
                print(f"✓ Best parameters for {name}: {grid_search.best_params_}")
                print(f"✓ Best CV AUC score: {grid_search.best_score_:.4f}")
                
            except Exception as e:
                print(f"ERROR training {name}: {e}")
                # Train with default parameters as fallback
                print(f"Training {name} with default parameters...")
                model.fit(X_train, y_train)
                self.trained_models[name] = model
                print(f"✓ {name} trained with default parameters")
    
    def calculate_metrics(self, y_true, y_pred, y_prob):
        """Calculate all required metrics"""
        try:
            # Handle multiclass case for specificity
            if len(np.unique(y_true)) > 2:
                # For multiclass, calculate macro-averaged specificity
                cm = confusion_matrix(y_true, y_pred)
                specificity_scores = []
                for i in range(len(cm)):
                    tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
                    fp = np.sum(cm[:, i]) - cm[i, i]
                    specificity_scores.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
                specificity = np.mean(specificity_scores)
                
                # For multiclass AUC, use macro average
                try:
                    auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
                except:
                    auc = 0.0
            else:
                # Binary classification
                cm = confusion_matrix(y_true, y_pred)
                tn, fp, fn, tp = cm.ravel()
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                auc = roc_auc_score(y_true, y_prob[:, 1] if y_prob.ndim > 1 else y_prob)
            
            metrics = {
                'Accuracy': accuracy_score(y_true, y_pred),
                'AUROC': auc,
                'Sensitivity': recall_score(y_true, y_pred, average='macro'),
                'Specificity': specificity,
                'F1-score': f1_score(y_true, y_pred, average='macro')
            }
            
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            metrics = {
                'Accuracy': 0.0,
                'AUROC': 0.0,
                'Sensitivity': 0.0,
                'Specificity': 0.0,
                'F1-score': 0.0
            }
        
        return metrics
    
    def evaluate_models(self, X_train, y_train, X_test, y_test):
        """Evaluate all models on training and test sets"""
        print("\n=== MODEL EVALUATION ===")
        
        self.results = {
            'Training': {},
            'Test': {}
        }
        
        for name, model in self.trained_models.items():
            print(f"\nEvaluating {name}...")
            
            try:
                # Training set evaluation
                train_pred = model.predict(X_train)
                train_prob = model.predict_proba(X_train)
                train_metrics = self.calculate_metrics(y_train, train_pred, train_prob)
                self.results['Training'][name] = train_metrics
                
                # Test set evaluation
                test_pred = model.predict(X_test)
                test_prob = model.predict_proba(X_test)
                test_metrics = self.calculate_metrics(y_test, test_pred, test_prob)
                self.results['Test'][name] = test_metrics
                
                print(f"✓ Test Accuracy: {test_metrics['Accuracy']:.4f}")
                print(f"✓ Test AUROC: {test_metrics['AUROC']:.4f}")
                
            except Exception as e:
                print(f"ERROR evaluating {name}: {e}")
    
    def generate_predictions(self, X_train, X_test, X_blind, train_id=None, test_id=None, blind_id=None):
        """Generate probability predictions for all datasets"""
        print("\n=== GENERATING PREDICTIONS ===")
        
        predictions = {}
        
        for name, model in self.trained_models.items():
            print(f"Generating predictions for {name}...")
            
            try:
                # Get class labels
                classes = model.classes_
                
                # Training set predictions
                train_probs = model.predict_proba(X_train)
                train_df = pd.DataFrame(train_probs, columns=[f'Class_{c}_Prob' for c in classes])
                if train_id is not None:
                    train_df.insert(0, 'ID', train_id.values if hasattr(train_id, 'values') else train_id)
                
                # Test set predictions
                test_probs = model.predict_proba(X_test)
                test_df = pd.DataFrame(test_probs, columns=[f'Class_{c}_Prob' for c in classes])
                if test_id is not None:
                    test_df.insert(0, 'ID', test_id.values if hasattr(test_id, 'values') else test_id)
                
                # Blind test set predictions
                blind_probs = model.predict_proba(X_blind)
                blind_df = pd.DataFrame(blind_probs, columns=[f'Class_{c}_Prob' for c in classes])
                if blind_id is not None:
                    blind_df.insert(0, 'ID', blind_id.values if hasattr(blind_id, 'values') else blind_id)
                
                predictions[name] = {
                    'train': train_df,
                    'test': test_df,
                    'blind': blind_df
                }
                
                print(f"✓ Predictions generated for {name}")
                
            except Exception as e:
                print(f"ERROR generating predictions for {name}: {e}")
        
        return predictions
    
    def save_predictions(self, predictions, output_dir='./'):
        """Save prediction CSV files"""
        print(f"\n=== SAVING PREDICTIONS to {output_dir} ===")
        
        for model_name, preds in predictions.items():
            model_clean = model_name.replace(' ', '_').lower()
            
            try:
                # Save each dataset's predictions
                preds['train'].to_csv(f"{output_dir}{model_clean}_train_predictions.csv", index=False)
                preds['test'].to_csv(f"{output_dir}{model_clean}_test_predictions.csv", index=False)
                preds['blind'].to_csv(f"{output_dir}{model_clean}_blind_predictions.csv", index=False)
                
                print(f"✓ Saved predictions for {model_name}")
                
            except Exception as e:
                print(f"ERROR saving predictions for {model_name}: {e}")
    
    def print_results_table(self):
        """Print formatted results table"""
        print("\n=== RESULTS TABLE ===")
        
        if not self.results:
            print("No results to display")
            return None
        
        try:
            # Create results DataFrame
            results_data = []
            for dataset in ['Training', 'Test']:
                if dataset in self.results:
                    for model in self.results[dataset]:
                        row = {'Dataset': dataset, 'Model': model}
                        row.update(self.results[dataset][model])
                        results_data.append(row)
            
            if results_data:
                results_df = pd.DataFrame(results_data)
                print(results_df.round(4).to_string(index=False))
                return results_df
            else:
                print("No results data available")
                return None
                
        except Exception as e:
            print(f"ERROR creating results table: {e}")
            return None
    
    def run_complete_pipeline(self, train_path, test_path, blind_path, target_column, 
                            id_column=None, output_dir='./'):
        """Run the complete ML pipeline"""
        print("=" * 50)
        print("STARTING COMPLETE ML CLASSIFICATION PIPELINE")
        print("=" * 50)
        
        try:
            # 1. Load data
            self.load_data(train_path, test_path, blind_path)
            
            # 2. Preprocess data
            X_train, X_test, X_blind, y_train = self.preprocess_data(target_column, id_column)
            
            # 3. Split training data for validation
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
            
            # 4. Setup models
            self.setup_models()
            
            # 5. Train and tune models
            self.train_and_tune_models(X_train_split, y_train_split)
            
            # 6. Evaluate models
            self.evaluate_models(X_train_split, y_train_split, X_val_split, y_val_split)
            
            # 7. Generate predictions
            train_id = self.train_data[id_column] if id_column and id_column in self.train_data.columns else None
            test_id = self.test_data[id_column] if id_column and id_column in self.test_data.columns else None
            blind_id = self.blind_test_data[id_column] if id_column and id_column in self.blind_test_data.columns else None
            
            predictions = self.generate_predictions(X_train, X_test, X_blind, train_id, test_id, blind_id)
            
            # 8. Save predictions
            self.save_predictions(predictions, output_dir)
            
            # 9. Print results
            results_df = self.print_results_table()
            
            print("\n" + "=" * 50)
            print("PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 50)
            
            return results_df, predictions
            
        except Exception as e:
            print(f"\nPIPELINE ERROR: {e}")
            print("=" * 50)
            return None, None

# Example usage
if __name__ == "__main__":
    pipeline = MLClassificationPipeline()
    results_df, predictions = pipeline.run_complete_pipeline(
        'train_set.csv', 
        'test_set.csv', 
        'blinded_test_set.csv', 
        'CLASS'
    )