import matplotlib
matplotlib.use('Agg')  # MUST be before importing pyplot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy.stats import ttest_ind
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import io
import base64
import json
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.cluster import AgglomerativeClustering
from kmodes.kprototypes import KPrototypes
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Store uploaded datasets in memory (for demo) or use proper database in production
uploaded_datasets = {}

def convert_to_serializable(obj):
    """
    Convert NumPy/Pandas data types to native Python types for JSON serialization
    """
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    elif pd.isna(obj):
        return None
    else:
        return obj

def validate_and_convert_treatment_column(df, treatment_col):
    """
    Validate and convert treatment column to binary (0/1)
    Returns: (success, message, converted_df, unique_values)
    """
    if treatment_col not in df.columns:
        return False, f"Treatment column '{treatment_col}' not found in dataset", df, None
    
    unique_values = df[treatment_col].dropna().unique()
    n_unique = len(unique_values)
    
    print(f"Treatment column '{treatment_col}' has {n_unique} unique values: {unique_values}")
    
    # If already binary (0/1), no conversion needed
    if n_unique == 2 and set(unique_values) == {0, 1}:
        return True, "Treatment column is already binary (0/1)", df, unique_values
    
    # If exactly 2 unique values, try to convert to binary
    if n_unique == 2:
        df_converted = df.copy()
        try:
            # Try to convert to numeric first
            df_converted[treatment_col] = pd.to_numeric(df_converted[treatment_col], errors='ignore')
            
            # If still object/string type, use label encoding
            if df_converted[treatment_col].dtype == 'object':
                le = LabelEncoder()
                df_converted[treatment_col] = le.fit_transform(df_converted[treatment_col])
                unique_converted = df_converted[treatment_col].unique()
                mapping = {original: converted for original, converted in zip(unique_values, unique_converted)}
                message = f"Treatment column converted from {unique_values.tolist()} to {unique_converted.tolist()}. Mapping: {mapping}"
            else:
                # For numeric with exactly 2 values, ensure they're 0 and 1
                min_val = df_converted[treatment_col].min()
                max_val = df_converted[treatment_col].max()
                if min_val != 0 or max_val != 1:
                    df_converted[treatment_col] = (df_converted[treatment_col] == max_val).astype(int)
                    message = f"Treatment column converted from [{min_val}, {max_val}] to [0, 1]"
                else:
                    message = "Treatment column is binary (0/1)"
                
            unique_converted = df_converted[treatment_col].unique()
            return True, message, df_converted, unique_converted
            
        except Exception as e:
            return False, f"Failed to convert treatment column: {str(e)}", df, unique_values
    
    # If more than 2 unique values, check if we can identify treatment/control
    elif n_unique > 2:
        # Try to identify the most common values that could represent treatment/control
        value_counts = df[treatment_col].value_counts()
        top_values = value_counts.head(2).index.tolist()
        
        if len(top_values) == 2:
            df_converted = df.copy()
            # Create binary column based on top 2 values
            df_converted[treatment_col] = (df_converted[treatment_col] == top_values[0]).astype(int)
            message = f"Treatment column converted: '{top_values[0]}' -> 1 (treatment), others -> 0 (control). Original had {n_unique} unique values."
            unique_converted = [0, 1]
            return True, message, df_converted, unique_converted
        else:
            return False, f"Treatment column has {n_unique} unique values. Cannot automatically determine treatment/control groups.", df, unique_values
    
    # If only 1 unique value
    else:
        return False, f"Treatment column has only 1 unique value ({unique_values[0]}). Need exactly 2 values for treatment/control groups.", df, unique_values

def generate_eda_visualizations(df, treatment_col, pre_features, outcome_col=None):
    """
    Generate comprehensive EDA visualizations comparing treatment vs control groups
    for various metrics like sasakyan, bahay, tubig, internet, etc.
    """
    try:
        visualizations = {}
        
        # 1. Treatment Group Distribution
        plt.figure(figsize=(8, 6))
        treatment_counts = df[treatment_col].value_counts()
        plt.pie(treatment_counts.values, labels=['Control', 'Treatment'], autopct='%1.1f%%', 
                colors=['lightblue', 'lightcoral'], startangle=90)
        plt.title('Treatment vs Control Group Distribution')
        
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', bbox_inches='tight')
        img_buf.seek(0)
        visualizations['treatment_distribution'] = base64.b64encode(img_buf.read()).decode('utf-8')
        plt.close()
        
        # 2. Numeric Features Comparison (Improved Bar plots instead of Box plots)
        numeric_features = df[pre_features].select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if numeric_features:
            # Select top 6 numeric features for visualization
            top_numeric = numeric_features[:6]
            n_plots = len(top_numeric)
            n_cols = 2
            n_rows = math.ceil(n_plots / n_cols)
            
            plt.figure(figsize=(15, 5 * n_rows))
            for i, feature in enumerate(top_numeric, 1):
                plt.subplot(n_rows, n_cols, i)
                
                # Calculate means and standard deviations for each group
                control_data = df[df[treatment_col] == 0][feature]
                treatment_data = df[df[treatment_col] == 1][feature]
                
                control_mean = control_data.mean()
                treatment_mean = treatment_data.mean()
                control_std = control_data.std()
                treatment_std = treatment_data.std()
                
                # Create bar plot with better styling (similar to cluster analysis)
                groups = ['Control', 'Treatment']
                means = [control_mean, treatment_mean]
                stds = [control_std, treatment_std]
                
                bars = plt.bar(groups, means, yerr=stds, capsize=5,
                              color=['lightblue', 'lightcoral'],
                              edgecolor='black', linewidth=1, alpha=0.8)
                
                plt.title(f'{feature} - Treatment vs Control', fontweight='bold')
                plt.xlabel('Group')
                plt.ylabel(f'Average {feature}')
                
                # Add value labels on bars
                for bar, mean_val in zip(bars, means):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                            f'{mean_val:.2f}', ha='center', va='bottom', fontweight='bold')
                
                # Add grid for better readability
                plt.grid(True, axis='y', alpha=0.3, linestyle='--')
                plt.gca().set_axisbelow(True)
                plt.gca().set_facecolor('#f8f9fa')
            
            plt.tight_layout()
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png', bbox_inches='tight', dpi=100,
                       facecolor='white', edgecolor='white')
            img_buf.seek(0)
            visualizations['numeric_features_comparison'] = base64.b64encode(img_buf.read()).decode('utf-8')
            plt.close()
        
        # 3. Categorical Features Comparison (Bar plots)
        categorical_features = df[pre_features].select_dtypes(include=['object']).columns.tolist()
        
        if categorical_features:
            # Select top 4 categorical features for visualization
            top_categorical = categorical_features[:4]
            n_plots = len(top_categorical)
            n_cols = 2
            n_rows = math.ceil(n_plots / n_cols)
            
            plt.figure(figsize=(15, 5 * n_rows))
            for i, feature in enumerate(top_categorical, 1):
                plt.subplot(n_rows, n_cols, i)
                
                # Calculate percentages
                cross_tab = pd.crosstab(df[feature], df[treatment_col], normalize='index') * 100
                cross_tab.plot(kind='bar', ax=plt.gca(), color=['lightblue', 'lightcoral'])
                plt.title(f'{feature} Distribution by Group', fontweight='bold')
                plt.xlabel(feature)
                plt.ylabel('Percentage (%)')
                plt.legend(['Control', 'Treatment'])
                plt.xticks(rotation=45)
                
                # Add grid for better readability
                plt.grid(True, axis='y', alpha=0.3, linestyle='--')
                plt.gca().set_axisbelow(True)
                plt.gca().set_facecolor('#f8f9fa')
            
            plt.tight_layout()
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png', bbox_inches='tight', dpi=100,
                       facecolor='white', edgecolor='white')
            img_buf.seek(0)
            visualizations['categorical_features_comparison'] = base64.b64encode(img_buf.read()).decode('utf-8')
            plt.close()
        
        # 4. Correlation Heatmap (Numeric features only)
        if len(numeric_features) > 1:
            plt.figure(figsize=(12, 10))
            correlation_matrix = df[numeric_features + [treatment_col]].corr()
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                       square=True, fmt='.2f', cbar_kws={"shrink": .8})
            plt.title('Correlation Heatmap (Numeric Features)', fontweight='bold')
            plt.tight_layout()
            
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png', bbox_inches='tight', dpi=100,
                       facecolor='white', edgecolor='white')
            img_buf.seek(0)
            visualizations['correlation_heatmap'] = base64.b64encode(img_buf.read()).decode('utf-8')
            plt.close()
        
        # 5. Outcome Variable Comparison (if available)
        if outcome_col and outcome_col in df.columns:
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            # Use improved bar plot style for outcome comparison
            control_outcome = df[df[treatment_col] == 0][outcome_col]
            treatment_outcome = df[df[treatment_col] == 1][outcome_col]
            
            control_mean = control_outcome.mean()
            treatment_mean = treatment_outcome.mean()
            control_std = control_outcome.std()
            treatment_std = treatment_outcome.std()
            
            groups = ['Control', 'Treatment']
            means = [control_mean, treatment_mean]
            stds = [control_std, treatment_std]
            
            bars = plt.bar(groups, means, yerr=stds, capsize=5,
                          color=['lightblue', 'lightcoral'],
                          edgecolor='black', linewidth=1, alpha=0.8)
            
            plt.title(f'{outcome_col} - Treatment vs Control', fontweight='bold')
            plt.xlabel('Group')
            plt.ylabel(outcome_col)
            
            # Add value labels on bars
            for bar, mean_val in zip(bars, means):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{mean_val:.2f}', ha='center', va='bottom', fontweight='bold')
            
            plt.grid(True, axis='y', alpha=0.3, linestyle='--')
            plt.gca().set_axisbelow(True)
            plt.gca().set_facecolor('#f8f9fa')
            
            plt.subplot(1, 2, 2)
            outcome_means = df.groupby(treatment_col)[outcome_col].mean()
            bars = plt.bar(['Control', 'Treatment'], outcome_means.values, 
                          color=['lightblue', 'lightcoral'],
                          edgecolor='black', linewidth=1, alpha=0.8)
            plt.title(f'Average {outcome_col} by Group', fontweight='bold')
            plt.ylabel(outcome_col)
            
            for i, (bar, v) in enumerate(zip(bars, outcome_means.values)):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
            
            plt.grid(True, axis='y', alpha=0.3, linestyle='--')
            plt.gca().set_axisbelow(True)
            plt.gca().set_facecolor('#f8f9fa')
            
            plt.tight_layout()
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png', bbox_inches='tight', dpi=100,
                       facecolor='white', edgecolor='white')
            img_buf.seek(0)
            visualizations['outcome_comparison'] = base64.b64encode(img_buf.read()).decode('utf-8')
            plt.close()
        
        # 6. Feature Importance for Treatment Prediction
        try:
            from sklearn.ensemble import RandomForestClassifier
            
            # Prepare data
            X = df[pre_features]
            y = df[treatment_col]
            
            # Handle categorical variables
            X_encoded = pd.get_dummies(X, drop_first=True)
            
            # Fit random forest
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_encoded, y)
            
            # Get feature importance
            feature_importance = pd.DataFrame({
                'feature': X_encoded.columns,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False).head(10)
            
            plt.figure(figsize=(10, 6))
            bars = plt.barh(range(len(feature_importance)), feature_importance['importance'],
                           color=sns.color_palette("viridis", len(feature_importance)),
                           edgecolor='black', linewidth=1, alpha=0.8)
            
            plt.gca().invert_yaxis()
            plt.title('Top 10 Features for Predicting Treatment Group', fontweight='bold')
            plt.xlabel('Feature Importance')
            plt.yticks(range(len(feature_importance)), feature_importance['feature'])
            
            # Add value labels on bars
            for i, (bar, importance) in enumerate(zip(bars, feature_importance['importance'])):
                plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{importance:.3f}', ha='left', va='center', fontweight='bold')
            
            plt.grid(True, axis='x', alpha=0.3, linestyle='--')
            plt.gca().set_axisbelow(True)
            plt.gca().set_facecolor('#f8f9fa')
            plt.gcf().patch.set_facecolor('white')
            
            plt.tight_layout()
            
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png', bbox_inches='tight', dpi=100,
                       facecolor='white', edgecolor='white')
            img_buf.seek(0)
            visualizations['feature_importance'] = base64.b64encode(img_buf.read()).decode('utf-8')
            plt.close()
            
        except Exception as e:
            print(f"Feature importance visualization failed: {str(e)}")
        
        # 7. Missing Values Heatmap
        missing_data = df[pre_features].isnull()
        if missing_data.any().any():
            plt.figure(figsize=(12, 8))
            sns.heatmap(missing_data, cbar=True, cmap='viridis', yticklabels=False)
            plt.title('Missing Values in Pre-Features', fontweight='bold')
            plt.xlabel('Features')
            plt.tight_layout()
            
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png', bbox_inches='tight', dpi=100,
                       facecolor='white', edgecolor='white')
            img_buf.seek(0)
            visualizations['missing_values'] = base64.b64encode(img_buf.read()).decode('utf-8')
            plt.close()
        
        return visualizations
        
    except Exception as e:
        print(f"Error generating EDA visualizations: {str(e)}")
        return {}

@app.route('/api/upload-dataset', methods=['POST'])
def upload_dataset():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'Only CSV files are supported'}), 400
        
        # Read the CSV file
        df = pd.read_csv(file)
        
        # Validate dataset size
        if len(df) < 10:
            return jsonify({'error': f'Dataset too small. Need at least 10 rows, got {len(df)}'}), 400
        
        if len(df.columns) < 3:
            return jsonify({'error': f'Dataset needs at least 3 columns, got {len(df.columns)}'}), 400
        
        # Check for missing values and provide info
        missing_info = {}
        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                missing_info[col] = {
                    'missing_count': int(missing_count),
                    'missing_percentage': float((missing_count / len(df)) * 100)
                }
        
        # Identify potential treatment columns (binary or with few unique values)
        potential_treatment_cols = []
        for col in df.columns:
            unique_values = df[col].dropna().unique()
            n_unique = len(unique_values)
            
            # Consider columns with 2 unique values as potential treatment columns
            if n_unique == 2:
                potential_treatment_cols.append({
                    'column': col,
                    'unique_values': convert_to_serializable(unique_values.tolist()),
                    'value_counts': convert_to_serializable(df[col].value_counts().to_dict())
                })
            # Also consider columns with small number of unique values
            elif 2 < n_unique <= 5:
                potential_treatment_cols.append({
                    'column': col,
                    'unique_values': convert_to_serializable(unique_values.tolist()),
                    'value_counts': convert_to_serializable(df[col].value_counts().to_dict()),
                    'note': f'Has {n_unique} unique values - may need conversion'
                })
        
        # Generate a unique ID for this dataset
        dataset_id = str(hash(file.filename + str(pd.Timestamp.now())))
        
        # Store dataset info
        uploaded_datasets[dataset_id] = {
            'dataframe': df,
            'filename': file.filename,
            'columns': df.columns.tolist(),
            'numeric_columns': df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
            'shape': df.shape,
            'missing_info': missing_info,
            'potential_treatment_cols': potential_treatment_cols
        }
        
        response_data = {
            'dataset_id': dataset_id,
            'filename': file.filename,
            'columns': df.columns.tolist(),
            'numeric_columns': df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
            'shape': [int(df.shape[0]), int(df.shape[1])],
            'missing_info': convert_to_serializable(missing_info),
            'potential_treatment_cols': potential_treatment_cols
        }
        
        return jsonify(response_data)
    
    except Exception as e:
        print(f"Error in upload_dataset: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/dataset-info/<dataset_id>', methods=['GET'])
def get_dataset_info(dataset_id):
    try:
        if dataset_id not in uploaded_datasets:
            return jsonify({'error': 'Dataset not found'}), 404
        
        dataset_info = uploaded_datasets[dataset_id]
        
        response_data = {
            'filename': dataset_info['filename'],
            'columns': dataset_info['columns'],
            'numeric_columns': dataset_info['numeric_columns'],
            'categorical_columns': dataset_info['categorical_columns'],
            'shape': [int(dataset_info['shape'][0]), int(dataset_info['shape'][1])],
            'missing_info': convert_to_serializable(dataset_info.get('missing_info', {})),
            'potential_treatment_cols': dataset_info.get('potential_treatment_cols', [])
        }
        
        return jsonify(response_data)
    
    except Exception as e:
        print(f"Error in get_dataset_info: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Function to calculate Cohen's d for independent samples
def cohen_d(d1, d2):
    n1, n2 = len(d1), len(d2)
    s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
    s = math.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    u1, u2 = np.mean(d1), np.mean(d2)
    return float((u1 - u2) / s)

# Function to check PSM success
def check_psm_success(df_effect_sizes, df_matched, df, treatment_col='treatment', ps_col='ps', effect_size_threshold=0.1, p_value_threshold=0.05):
    results = {
        'passed': True,
        'reasons': [],
        'failed_covariates': [],
        'effect_size_summary': None,
        'p_value_summary': None,
        'overlap_status': None,
        'sample_size_status': None
    }
    
    # 1. Check covariate balance (effect sizes and p-values after matching)
    after_matching = df_effect_sizes[df_effect_sizes['matching'] == 'after']
    
    # Count covariates with effect size > threshold or p-value < threshold
    large_effect_size = after_matching[abs(after_matching['effect_size']) > effect_size_threshold]
    significant_p_value = after_matching[after_matching['p-value'] < p_value_threshold]
    
    # Failure if more than 20% of covariates are unbalanced
    unbalanced_covariates = set(large_effect_size['feature']).union(set(significant_p_value['feature']))
    unbalanced_percentage = len(unbalanced_covariates) / len(after_matching) * 100
    
    if unbalanced_percentage > 20:
        results['passed'] = False
        results['reasons'].append(f"High percentage of unbalanced covariates ({unbalanced_percentage:.1f}%).")
        results['failed_covariates'] = list(unbalanced_covariates)
    
    # Store summaries - convert to native Python types
    effect_size_desc = after_matching['effect_size'].describe()
    p_value_desc = after_matching['p-value'].describe()
    
    results['effect_size_summary'] = {
        'count': float(effect_size_desc['count']),
        'mean': float(effect_size_desc['mean']),
        'std': float(effect_size_desc['std']),
        'min': float(effect_size_desc['min']),
        '25%': float(effect_size_desc['25%']),
        '50%': float(effect_size_desc['50%']),
        '75%': float(effect_size_desc['75%']),
        'max': float(effect_size_desc['max'])
    }
    
    results['p_value_summary'] = {
        'count': float(p_value_desc['count']),
        'mean': float(p_value_desc['mean']),
        'std': float(p_value_desc['std']),
        'min': float(p_value_desc['min']),
        '25%': float(p_value_desc['25%']),
        '50%': float(p_value_desc['50%']),
        '75%': float(p_value_desc['75%']),
        'max': float(p_value_desc['max'])
    }
    
    # 2. Check propensity score overlap
    treatment_ps = df_matched[df_matched[treatment_col] == 1][ps_col]
    control_ps = df_matched[df_matched[treatment_col] == 0][ps_col]
    
    # Compare quantiles to check overlap
    treatment_quantiles = treatment_ps.quantile([0.1, 0.5, 0.9]).to_dict()
    control_quantiles = control_ps.quantile([0.1, 0.5, 0.9]).to_dict()
    
    # Convert quantiles to native Python types
    treatment_quantiles = {str(k): float(v) for k, v in treatment_quantiles.items()}
    control_quantiles = {str(k): float(v) for k, v in control_quantiles.items()}
    
    # Check if quantiles are within a reasonable range (e.g., 0.1 difference)
    quantile_diff = max([abs(treatment_quantiles[q] - control_quantiles[q]) for q in treatment_quantiles.keys()])
    if quantile_diff > 0.2:
        results['passed'] = False
        results['reasons'].append(f"Propensity score distributions do not overlap well (max quantile difference: {quantile_diff:.2f}).")
    
    results['overlap_status'] = {
        'treatment_quantiles': treatment_quantiles,
        'control_quantiles': control_quantiles,
        'max_quantile_diff': float(quantile_diff)
    }
    
    # 3. Check sample size retention
    original_treatment_size = int(len(df[df[treatment_col] == 1]))
    matched_treatment_size = int(len(df_matched[df_matched[treatment_col] == 1]))
    retention_rate = float(matched_treatment_size / original_treatment_size * 100)
    
    if retention_rate < 80:
        results['passed'] = False
        results['reasons'].append(f"Low sample retention rate ({retention_rate:.1f}%).")
    
    results['sample_size_status'] = {
        'original_treatment_size': original_treatment_size,
        'matched_treatment_size': matched_treatment_size,
        'retention_rate': retention_rate
    }
    
    return results

def generate_cluster_feature_visualizations(df_cleaned, final_labels, selected_features=None):
    """Generate seaborn visualizations for each feature in cluster analysis with improved bar graphs"""
    try:
        df_cleaned = df_cleaned.copy()
        df_cleaned['cluster'] = final_labels
        
        # Filter features if specific ones are selected
        if selected_features:
            available_features = [f for f in selected_features if f in df_cleaned.columns]
            features_to_analyze = available_features
        else:
            features_to_analyze = [col for col in df_cleaned.columns if col not in ['cluster', 'final_cluster']]
        
        feature_visualizations = {}
        
        for feature in features_to_analyze:
            if feature == 'cluster':
                continue
            
            try:
                if df_cleaned[feature].dtype in ['int64', 'float64']:
                    # Numeric feature - create bar plot of means by cluster
                    plt.figure(figsize=(10, 6))
                    
                    # Calculate means and standard deviations
                    cluster_stats = df_cleaned.groupby('cluster')[feature].agg(['mean', 'std']).reset_index()
                    
                    # Create bar plot
                    bars = plt.bar(cluster_stats['cluster'].astype(str), cluster_stats['mean'],
                                  yerr=cluster_stats['std'], capsize=5,
                                  color=sns.color_palette("viridis", len(cluster_stats)),
                                  edgecolor='black', linewidth=1, alpha=0.8)
                    
                    plt.title(f'{feature} - Average by Cluster', fontweight='bold')
                    plt.xlabel('Cluster')
                    plt.ylabel(f'Average {feature}')
                    
                    # Add value labels on bars
                    for bar, mean_val in zip(bars, cluster_stats['mean']):
                        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                                f'{mean_val:.2f}', ha='center', va='bottom', fontweight='bold')
                    
                    plt.grid(True, axis='y', alpha=0.3, linestyle='--')
                    plt.gca().set_axisbelow(True)
                    plt.gca().set_facecolor('#f8f9fa')
                    plt.gcf().patch.set_facecolor('white')
                    
                else:
                    # Categorical feature - create stacked bar plot
                    plt.figure(figsize=(12, 6))
                    
                    # Get value counts by cluster
                    cross_tab = pd.crosstab(df_cleaned['cluster'], df_cleaned[feature], normalize='index') * 100
                    
                    # Limit to top 5 categories for readability
                    top_categories = cross_tab.sum().nlargest(5).index
                    cross_tab = cross_tab[top_categories]
                    
                    # Create stacked bar plot
                    ax = cross_tab.plot(kind='bar', stacked=True, 
                                       color=sns.color_palette("Set3", len(cross_tab.columns)),
                                       edgecolor='black', linewidth=0.5, figsize=(12, 6))
                    
                    plt.title(f'{feature} Distribution by Cluster (%)', fontweight='bold')
                    plt.xlabel('Cluster')
                    plt.ylabel('Percentage (%)')
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    plt.xticks(rotation=0)
                    
                    # Add value labels for the largest segment in each bar
                    for i, cluster in enumerate(cross_tab.index):
                        total_height = 0
                        for category in cross_tab.columns:
                            value = cross_tab.loc[cluster, category]
                            if value > 15:  # Only label segments larger than 15%
                                plt.text(i, total_height + value/2, f'{value:.1f}%', 
                                        ha='center', va='center', fontweight='bold', fontsize=8)
                            total_height += value
                    
                    plt.grid(True, axis='y', alpha=0.3, linestyle='--')
                    plt.gca().set_axisbelow(True)
                    plt.gca().set_facecolor('#f8f9fa')
                    plt.gcf().patch.set_facecolor('white')
                
                plt.tight_layout()
                
                img_buf = io.BytesIO()
                plt.savefig(img_buf, format='png', bbox_inches='tight', dpi=100,
                           facecolor='white', edgecolor='white')
                img_buf.seek(0)
                feature_visualizations[feature] = base64.b64encode(img_buf.read()).decode('utf-8')
                plt.close()
                
            except Exception as e:
                print(f"Error creating visualization for {feature}: {str(e)}")
                # Create a simple error visualization
                plt.figure(figsize=(8, 4))
                plt.text(0.5, 0.5, f'Error visualizing {feature}', 
                        ha='center', va='center', transform=plt.gca().transAxes)
                plt.axis('off')
                
                img_buf = io.BytesIO()
                plt.savefig(img_buf, format='png', bbox_inches='tight')
                img_buf.seek(0)
                feature_visualizations[feature] = base64.b64encode(img_buf.read()).decode('utf-8')
                plt.close()
        
        return feature_visualizations
        
    except Exception as e:
        print(f"Error generating cluster feature visualizations: {str(e)}")
        return {}

def generate_cluster_summaries(df_cleaned, final_labels, selected_features=None):
    """Generate comprehensive cluster summaries for selected features only"""
    df_cleaned = df_cleaned.copy()
    df_cleaned['final_cluster'] = final_labels
    
    # Filter features if specific ones are selected
    if selected_features:
        available_features = [f for f in selected_features if f in df_cleaned.columns]
        features_to_analyze = available_features
    else:
        features_to_analyze = [col for col in df_cleaned.columns if col != 'final_cluster']
    
    cluster_summaries = {}
    
    for feature in features_to_analyze:
        if feature == 'final_cluster':
            continue
        
        if df_cleaned[feature].dtype in ['int64', 'float64']:
            # For numeric features, calculate statistics
            feature_summary = df_cleaned.groupby('final_cluster')[feature].agg([
                'count', 'mean', 'std', 'min', 'max', 'median'
            ]).round(4)
            
            # Convert to dictionary with native Python types
            feature_summary_dict = {}
            for stat in feature_summary.columns:
                feature_summary_dict[stat] = {
                    str(cluster): float(value) for cluster, value in feature_summary[stat].items()
                }
        else:
            # For categorical features, calculate value distributions
            value_counts = df_cleaned.groupby('final_cluster')[feature].value_counts(normalize=True)
            value_counts = value_counts.unstack(fill_value=0).round(4)
            
            # Convert to dictionary format with native Python types
            feature_summary_dict = {}
            for cluster in value_counts.index:
                cluster_dict = {}
                for val in value_counts.columns:
                    if value_counts.loc[cluster, val] > 0:  # Only include values that exist
                        cluster_dict[str(val)] = float(value_counts.loc[cluster, val])
                feature_summary_dict[str(cluster)] = cluster_dict
        
        cluster_summaries[feature] = feature_summary_dict
    
    return cluster_summaries

def generate_additional_clustering_visualizations(df_cleaned, final_labels, X_scaled):
    """Generate additional clustering visualizations with improved bar graphs"""
    try:
        visualizations = {}
        
        # 1. Cluster distribution - IMPROVED BAR GRAPH
        plt.figure(figsize=(10, 6))
        cluster_counts = pd.Series(final_labels).value_counts().sort_index()
        
        # Create bar plot with better styling
        bars = plt.bar(range(len(cluster_counts)), cluster_counts.values, 
                      color=sns.color_palette("Set2", len(cluster_counts)),
                      edgecolor='black', linewidth=1.2, alpha=0.8)
        
        plt.title('Cluster Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Cluster', fontsize=12)
        plt.ylabel('Number of Points', fontsize=12)
        plt.xticks(range(len(cluster_counts)), [f'Cluster {i}' for i in cluster_counts.index])
        
        # Add value labels on top of bars
        for i, (bar, count) in enumerate(zip(bars, cluster_counts.values)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    str(count), ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # Add grid for better readability
        plt.grid(True, axis='y', alpha=0.3, linestyle='--')
        plt.gca().set_axisbelow(True)  # Grid behind bars
        
        # Set background color
        plt.gca().set_facecolor('#f8f9fa')
        plt.gcf().patch.set_facecolor('white')
        
        plt.tight_layout()
        
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', bbox_inches='tight', dpi=100, 
                   facecolor='white', edgecolor='white')
        img_buf.seek(0)
        visualizations['cluster_distribution'] = base64.b64encode(img_buf.read()).decode('utf-8')
        plt.close()
        
        # 2. Feature importance bar graph - IMPROVED
        if hasattr(X_scaled, 'shape') and X_scaled.shape[1] > 0:
            try:
                # Calculate feature importance using variance
                feature_importance = np.std(X_scaled, axis=0)
                feature_names = [f'Feature {i+1}' for i in range(len(feature_importance))]
                
                # Sort features by importance
                sorted_idx = np.argsort(feature_importance)[::-1]
                sorted_importance = feature_importance[sorted_idx]
                sorted_names = [feature_names[i] for i in sorted_idx]
                
                # Take top 10 features
                top_n = min(10, len(sorted_importance))
                
                plt.figure(figsize=(12, 6))
                bars = plt.bar(range(top_n), sorted_importance[:top_n], 
                              color=sns.color_palette("viridis", top_n),
                              edgecolor='black', linewidth=1, alpha=0.8)
                
                plt.title('Top Feature Importance (Standard Deviation)', fontsize=14, fontweight='bold')
                plt.xlabel('Features', fontsize=12)
                plt.ylabel('Standard Deviation', fontsize=12)
                plt.xticks(range(top_n), sorted_names[:top_n], rotation=45, ha='right')
                
                # Add value labels
                for i, (bar, value) in enumerate(zip(bars, sorted_importance[:top_n])):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                            f'{value:.3f}', ha='center', va='bottom', fontsize=9)
                
                plt.grid(True, axis='y', alpha=0.3, linestyle='--')
                plt.gca().set_axisbelow(True)
                plt.gca().set_facecolor('#f8f9fa')
                plt.gcf().patch.set_facecolor('white')
                
                plt.tight_layout()
                
                img_buf = io.BytesIO()
                plt.savefig(img_buf, format='png', bbox_inches='tight', dpi=100,
                           facecolor='white', edgecolor='white')
                img_buf.seek(0)
                visualizations['feature_importance'] = base64.b64encode(img_buf.read()).decode('utf-8')
                plt.close()
                
            except Exception as e:
                print(f"Feature importance visualization failed: {str(e)}")
        
        # 3. Cluster means for top features - IMPROVED BAR GRAPHS
        numeric_features = df_cleaned.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_features) > 0 and len(np.unique(final_labels)) > 1:
            # Take first 4 numeric features
            features_to_plot = numeric_features[:min(4, len(numeric_features))]
            
            n_features = len(features_to_plot)
            n_cols = 2
            n_rows = math.ceil(n_features / n_cols)
            
            plt.figure(figsize=(15, 5 * n_rows))
            
            for i, feature in enumerate(features_to_plot, 1):
                plt.subplot(n_rows, n_cols, i)
                
                # Calculate cluster means
                cluster_means = []
                cluster_names = []
                unique_clusters = np.unique(final_labels)
                
                for cluster in unique_clusters:
                    cluster_data = df_cleaned[final_labels == cluster][feature]
                    if len(cluster_data) > 0:
                        cluster_means.append(cluster_data.mean())
                        cluster_names.append(f'Cluster {cluster}')
                
                if cluster_means:
                    # Create bar plot for cluster means
                    bars = plt.bar(range(len(cluster_means)), cluster_means,
                                  color=sns.color_palette("Set3", len(cluster_means)),
                                  edgecolor='black', linewidth=1, alpha=0.8)
                    
                    plt.title(f'{feature} - Average by Cluster', fontweight='bold')
                    plt.xlabel('Cluster')
                    plt.ylabel(f'Average {feature}')
                    plt.xticks(range(len(cluster_means)), cluster_names, rotation=45)
                    
                    # Add value labels on bars
                    for j, (bar, mean_val) in enumerate(zip(bars, cluster_means)):
                        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                                f'{mean_val:.2f}', ha='center', va='bottom', fontweight='bold')
                    
                    plt.grid(True, axis='y', alpha=0.3, linestyle='--')
                    plt.gca().set_axisbelow(True)
                    plt.gca().set_facecolor('#f8f9fa')
            
            plt.tight_layout()
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png', bbox_inches='tight', dpi=100,
                       facecolor='white', edgecolor='white')
            img_buf.seek(0)
            visualizations['cluster_means'] = base64.b64encode(img_buf.read()).decode('utf-8')
            plt.close()
        
        return visualizations
        
    except Exception as e:
        print(f"Error generating additional clustering visualizations: {str(e)}")
        return {}

def perform_clustering_analysis(df, selected_features=None):
    """Perform clustering analysis on the dataset with optional feature selection"""
    try:
        # Data cleaning and preprocessing
        df_cleaned = df.copy()
        
        # Drop duplicates
        rows_before = int(len(df_cleaned))
        df_cleaned = df_cleaned.drop_duplicates()
        rows_after = int(len(df_cleaned))
        
        # Handle missing values
        num_cols = df_cleaned.select_dtypes(include='number').columns
        for col in num_cols:
            median_value = df_cleaned[col].median()
            df_cleaned[col] = df_cleaned[col].fillna(median_value)
        
        # Drop columns with 50% or more missing values
        threshold_col = len(df_cleaned) * 0.5
        df_cleaned = df_cleaned.dropna(axis=1, thresh=threshold_col)
        
        # Convert all strings to lowercase
        text_cols = df_cleaned.select_dtypes(include='object').columns
        for col in text_cols:
            df_cleaned[col] = df_cleaned[col].astype(str).str.lower().str.strip()
        
        # Check if we have enough data after cleaning
        if len(df_cleaned) < 5:
            return {'error': 'Not enough data after cleaning (need at least 5 rows)'}
        
        # Filter features if specific ones are selected
        if selected_features:
            available_features = [f for f in selected_features if f in df_cleaned.columns]
            if len(available_features) == 0:
                return {'error': 'No selected features available in the dataset after cleaning'}
            df_num = df_cleaned[available_features].select_dtypes(include='number').copy()
        else:
            # Get numeric columns for clustering
            df_num = df_cleaned.select_dtypes(include='number').copy()
        
        if len(df_num.columns) == 0:
            return {'error': 'No numeric columns available for clustering'}
        
        if len(df_num) < 5:
            return {'error': 'Not enough numeric data for clustering (need at least 5 rows)'}
        
        # Adjust number of clusters based on data size
        K = min(3, len(df_num) - 1)
        if K < 2:
            return {'error': 'Not enough data for meaningful clustering'}
        
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(df_num)
        
        metrics_summary = []
        
        def evaluate_clustering(X, labels, name):
            try:
                if len(np.unique(labels)) < 2:
                    return {
                        'Model': name,
                        'Silhouette': 0.0,
                        'Calinski-Harabasz': 0.0,
                        'Davies-Bouldin': float('inf')
                    }
                
                silhouette = float(silhouette_score(X, labels)) if len(np.unique(labels)) > 1 else 0.0
                calinski = float(calinski_harabasz_score(X, labels)) if len(np.unique(labels)) > 1 else 0.0
                davies = float(davies_bouldin_score(X, labels)) if len(np.unique(labels)) > 1 else float('inf')
                
                metrics_summary.append({
                    'Model': name,
                    'Silhouette': silhouette,
                    'Calinski-Harabasz': calinski,
                    'Davies-Bouldin': davies
                })
                
                return {
                    'Model': name,
                    'Silhouette': f"{silhouette:.4f}",
                    'Calinski-Harabasz': f"{calinski:.4f}",
                    'Davies-Bouldin': f"{davies:.4f}"
                }
            except Exception as e:
                print(f"Error evaluating {name}: {str(e)}")
                return {
                    'Model': name,
                    'Silhouette': 0.0,
                    'Calinski-Harabasz': 0.0,
                    'Davies-Bouldin': float('inf')
                }
        
        # PCA
        try:
            pca_model = PCA(n_components=min(2, X_scaled.shape[1]))
            X_pca = pca_model.fit_transform(X_scaled)
            kmeans_pca = KMeans(n_clusters=K, random_state=42, n_init=10)
            labels_pca = kmeans_pca.fit_predict(X_pca)
            pca_results = evaluate_clustering(X_pca, labels_pca, "PCA")
        except Exception as e:
            print(f"PCA failed: {str(e)}")
            pca_results = evaluate_clustering(X_scaled, np.zeros(len(X_scaled)), "PCA")
        
        # Entropy Weighted K-Means
        def entropy_weights(X):
            try:
                X_norm = X / (X.sum(axis=0) + 1e-9)
                X_norm = np.where(X_norm == 0, 1e-9, X_norm)
                entropy = -np.sum(X_norm * np.log(X_norm), axis=0) / np.log(len(X))
                D = 1 - entropy
                weights = D / D.sum()
                return weights
            except:
                return np.ones(X.shape[1]) / X.shape[1]
        
        try:
            weights = entropy_weights(X_scaled)
            X_entropy = X_scaled * weights
            kmeans_entropy = KMeans(n_clusters=K, random_state=42, n_init=10)
            labels_entropy = kmeans_entropy.fit_predict(X_entropy)
            entropy_results = evaluate_clustering(X_entropy, labels_entropy, "Entropy-Weighted")
        except Exception as e:
            print(f"Entropy weighted failed: {str(e)}")
            entropy_results = evaluate_clustering(X_scaled, np.zeros(len(X_scaled)), "Entropy-Weighted")
            weights = np.ones(X_scaled.shape[1]) / X_scaled.shape[1]
        
        # Select best model
        metrics_df = pd.DataFrame(metrics_summary)
        if len(metrics_df) > 0:
            scoring_df = metrics_df.copy()
            scoring_df['Inverse Davies-Bouldin'] = 1 / (scoring_df['Davies-Bouldin'] + 1e-6)
            scaler = MinMaxScaler()
            scaled_metrics = scaler.fit_transform(scoring_df[['Silhouette', 'Calinski-Harabasz', 'Inverse Davies-Bouldin']])
            scoring_df['Composite Score'] = scaled_metrics.mean(axis=1)
            best_model = scoring_df.loc[scoring_df['Composite Score'].idxmax()].to_dict()
        else:
            best_model = {'Model': 'PCA', 'Composite Score': 0.0}
        
        # Extract feature weights from best model
        if best_model['Model'] == "PCA":
            try:
                pca_weights = np.abs(pca_model.components_[:2]).mean(axis=0)
                feature_weights = pd.Series(pca_weights, index=df_num.columns).sort_values(ascending=False)
            except:
                feature_weights = pd.Series(np.ones(len(df_num.columns)) / len(df_num.columns), index=df_num.columns)
        elif best_model['Model'] == "Entropy-Weighted":
            feature_weights = pd.Series(weights, index=df_num.columns).sort_values(ascending=False)
        else:
            feature_weights = pd.Series(np.ones(len(df_num.columns)) / len(df_num.columns), index=df_num.columns)
        
        # Convert feature weights to native Python types
        feature_weights_dict = {str(k): float(v) for k, v in feature_weights.to_dict().items()}
        top_features_dict = {str(k): float(v) for k, v in feature_weights.head(10).to_dict().items()}
        
        # Ensemble clustering
        metrics_summary_2 = []
        
        def evaluate_model(X, labels, name):
            try:
                if len(np.unique(labels)) < 2:
                    return {
                        'Model': name,
                        'Silhouette': 0.0,
                        'Calinski-Harabasz': 0.0,
                        'Davies-Bouldin': float('inf')
                    }
                
                silhouette = float(silhouette_score(X, labels)) if len(np.unique(labels)) > 1 else 0.0
                calinski = float(calinski_harabasz_score(X, labels)) if len(np.unique(labels)) > 1 else 0.0
                davies = float(davies_bouldin_score(X, labels)) if len(np.unique(labels)) > 1 else float('inf')
                
                metrics_summary_2.append({
                    'Model': name,
                    'Silhouette': silhouette,
                    'Calinski-Harabasz': calinski,
                    'Davies-Bouldin': davies
                })
                
                return {
                    'Model': name,
                    'Silhouette': f"{silhouette:.4f}",
                    'Calinski-Harabasz': f"{calinski:.4f}",
                    'Davies-Bouldin': f"{davies:.4f}"
                }
            except Exception as e:
                print(f"Error evaluating {name}: {str(e)}")
                return {
                    'Model': name,
                    'Silhouette': 0.0,
                    'Calinski-Harabasz': 0.0,
                    'Davies-Bouldin': float('inf')
                }
        
        # Separate numeric and categorical
        df_num_ensemble = df_num.copy()  # Use the filtered numeric data
        
        # Normalize numeric data
        scaler = MinMaxScaler()
        X_num_scaled = scaler.fit_transform(df_num_ensemble)
        
        # Weighted KMeans
        try:
            weights_ensemble = entropy_weights(X_num_scaled)
            X_weighted = X_num_scaled * weights_ensemble
            kmeans_weighted = KMeans(n_clusters=K, random_state=42, n_init=10)
            labels_wkmeans = kmeans_weighted.fit_predict(X_weighted)
            wkmeans_results = evaluate_model(X_weighted, labels_wkmeans, "Weighted KMeans")
        except Exception as e:
            print(f"Weighted KMeans failed: {str(e)}")
            wkmeans_results = evaluate_model(X_num_scaled, np.zeros(len(X_num_scaled)), "Weighted KMeans")
            labels_wkmeans = np.zeros(len(X_num_scaled))
        
        # Hierarchical Clustering
        try:
            hierarchical = AgglomerativeClustering(n_clusters=K, linkage='ward')
            labels_hier = hierarchical.fit_predict(X_num_scaled)
            hier_results = evaluate_model(X_num_scaled, labels_hier, "Hierarchical Clustering")
        except Exception as e:
            print(f"Hierarchical clustering failed: {str(e)}")
            hier_results = evaluate_model(X_num_scaled, np.zeros(len(X_num_scaled)), "Hierarchical Clustering")
            labels_hier = np.zeros(len(X_num_scaled))
        
        # Select best ensemble model
        if len(metrics_summary_2) > 0:
            metrics_df2 = pd.DataFrame(metrics_summary_2)
            scoring_df2 = metrics_df2.copy()
            scoring_df2['Inverse Davies-Bouldin'] = 1 / (scoring_df2['Davies-Bouldin'] + 1e-6)
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(scoring_df2[['Silhouette', 'Calinski-Harabasz', 'Inverse Davies-Bouldin']])
            scoring_df2['Composite Score'] = scaled.mean(axis=1)
            best_model2 = scoring_df2.loc[scoring_df2['Composite Score'].idxmax()].to_dict()
        else:
            best_model2 = {'Model': 'Weighted KMeans', 'Composite Score': 0.0}
        
        # Apply final clustering
        final_labels = labels_wkmeans  # Default to weighted kmeans
        
        # Generate cluster summaries with selected features only
        cluster_summaries = generate_cluster_summaries(df_cleaned, final_labels, selected_features)
        
        # Generate feature visualizations
        feature_visualizations = generate_cluster_feature_visualizations(df_cleaned, final_labels, selected_features)
        
        # Generate visualizations - FIXED CLUSTER VISUALIZATION
        feature_weights_base64 = ""
        try:
            plt.figure(figsize=(12, 6))
            
            # Take top features for display
            top_features = feature_weights.head(min(10, len(feature_weights)))
            
            # Create horizontal bar plot with better styling
            bars = plt.barh(range(len(top_features)), top_features.values,
                           color=sns.color_palette("viridis", len(top_features)),
                           edgecolor='black', linewidth=1, alpha=0.8)
            
            plt.gca().invert_yaxis()
            plt.title(f"Top Feature Weights - {best_model['Model']}", fontsize=14, fontweight='bold')
            plt.xlabel("Weight", fontsize=12)
            plt.yticks(range(len(top_features)), top_features.index, fontsize=10)
            
            # Add value labels on bars
            for i, (bar, weight) in enumerate(zip(bars, top_features.values)):
                plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{weight:.3f}', ha='left', va='center', fontweight='bold', fontsize=9)
            
            # Add grid and background
            plt.grid(True, axis='x', alpha=0.3, linestyle='--')
            plt.gca().set_axisbelow(True)
            plt.gca().set_facecolor('#f8f9fa')
            plt.gcf().patch.set_facecolor('white')
            
            plt.tight_layout()
            
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png', bbox_inches='tight', dpi=100, 
                       facecolor='white', edgecolor='white')
            img_buf.seek(0)
            feature_weights_base64 = base64.b64encode(img_buf.read()).decode('utf-8')
            plt.close()
        except Exception as e:
            print(f"Feature weights visualization failed: {str(e)}")
            feature_weights_base64 = ""
        
        # FIXED: Cluster visualization with better error handling
        clusters_base64 = ""
        try:
            # Use PCA for visualization
            pca_vis = PCA(n_components=2)
            X_vis = pca_vis.fit_transform(X_num_scaled)
            
            plt.figure(figsize=(10, 8))
            
            # Create scatter plot with clusters
            scatter = plt.scatter(X_vis[:, 0], X_vis[:, 1], c=final_labels, cmap='Set2', s=60, alpha=0.7)
            
            # Add labels and title
            plt.title(f"Cluster Visualization - {best_model2['Model']}\n(PCA Projection)", fontsize=14, fontweight='bold')
            plt.xlabel("Principal Component 1")
            plt.ylabel("Principal Component 2")
            
            # Add colorbar
            cbar = plt.colorbar(scatter)
            cbar.set_label('Cluster', rotation=270, labelpad=15)
            
            # Add grid for better readability
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save to buffer
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png', bbox_inches='tight', dpi=100, facecolor='white')
            img_buf.seek(0)
            clusters_base64 = base64.b64encode(img_buf.read()).decode('utf-8')
            plt.close()
            
        except Exception as e:
            print(f"Cluster visualization failed: {str(e)}")
            # Create a simple error visualization
            try:
                plt.figure(figsize=(8, 6))
                plt.text(0.5, 0.5, f'Cluster visualization unavailable\n{str(e)}', 
                        ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
                plt.axis('off')
                
                img_buf = io.BytesIO()
                plt.savefig(img_buf, format='png', bbox_inches='tight', dpi=100)
                img_buf.seek(0)
                clusters_base64 = base64.b64encode(img_buf.read()).decode('utf-8')
                plt.close()
            except:
                clusters_base64 = ""
        
        # Generate additional clustering visualizations
        clustering_visualizations = generate_additional_clustering_visualizations(df_cleaned, final_labels, X_num_scaled)
        
        return {
            'dimensionality_reduction': {
                'best_model': str(best_model['Model']),
                'metrics': convert_to_serializable(metrics_summary),
                'feature_weights': feature_weights_dict,
                'top_features': top_features_dict,
                'visualization': feature_weights_base64
            },
            'ensemble_clustering': {
                'best_model': str(best_model2['Model']),
                'metrics': convert_to_serializable(metrics_summary_2),
                'cluster_visualization': clusters_base64,
                'cluster_profiles': "",
                'cluster_summaries': cluster_summaries,
                'feature_visualizations': feature_visualizations,
                'clustering_visualizations': clustering_visualizations  # Add additional visualizations
            },
            'data_cleaning': {
                'rows_before': rows_before,
                'rows_after': rows_after,
                'duplicates_removed': int(rows_before - rows_after)
            },
            'selected_features': selected_features if selected_features else 'all'
        }
    
    except Exception as e:
        print(f"Error in perform_clustering_analysis: {str(e)}")
        return {'error': f'Clustering analysis failed: {str(e)}'}

def handle_missing_values(df, pre_features, strategy='median'):
    """
    Handle missing values in the dataset
    strategy: 'median', 'mean', 'mode', 'drop', or 'iterative'
    """
    df_clean = df.copy()
    
    # Separate numeric and categorical columns from pre_features
    numeric_features = [col for col in pre_features if col in df_clean.select_dtypes(include=['int64', 'float64']).columns]
    categorical_features = [col for col in pre_features if col in df_clean.select_dtypes(include=['object']).columns]
    
    missing_info = {}
    
    # Handle numeric features
    for col in numeric_features:
        missing_count = df_clean[col].isna().sum()
        if missing_count > 0:
            missing_info[col] = {
                'type': 'numeric',
                'missing_count': int(missing_count),
                'missing_percentage': float((missing_count / len(df_clean)) * 100)
            }
            
            if strategy == 'drop':
                # Drop rows with missing values in this column
                df_clean = df_clean.dropna(subset=[col])
            elif strategy == 'median':
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            elif strategy == 'mean':
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
            elif strategy == 'mode':
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 0)
            elif strategy == 'iterative':
                # Use iterative imputer for more sophisticated imputation
                imputer = IterativeImputer(max_iter=10, random_state=42)
                df_clean[col] = imputer.fit_transform(df_clean[[col]]).ravel()
    
    # Handle categorical features
    for col in categorical_features:
        missing_count = df_clean[col].isna().sum()
        if missing_count > 0:
            missing_info[col] = {
                'type': 'categorical',
                'missing_count': int(missing_count),
                'missing_percentage': float((missing_count / len(df_clean)) * 100)
            }
            
            if strategy == 'drop':
                df_clean = df_clean.dropna(subset=[col])
            else:
                # For categorical, use mode (most frequent) or 'Missing' category
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Missing')
    
    return df_clean, missing_info

@app.route('/api/psm-analysis', methods=['POST'])
def psm_analysis():
    try:
        # Get request data
        data = request.get_json()
        dataset_id = data.get('dataset_id')
        pre_features = data.get('pre_features', [])
        treatment_col = data.get('treatment_col')
        outcome_col = data.get('outcome_col')
        missing_strategy = data.get('missing_strategy', 'median')  # Default to median imputation
        clustering_features = data.get('clustering_features', [])  # New parameter for clustering feature selection
        
        if not dataset_id or dataset_id not in uploaded_datasets:
            return jsonify({'error': 'Dataset not found'}), 404
        
        # Load data from stored dataset
        df = uploaded_datasets[dataset_id]['dataframe']
        
        # Validate required columns
        if not pre_features:
            return jsonify({'error': 'No pre-features selected'}), 400
        
        if not treatment_col:
            return jsonify({'error': 'Treatment column not specified'}), 400
        
        if treatment_col not in df.columns:
            return jsonify({'error': f'Treatment column "{treatment_col}" not found in dataset'}), 400
        
        for feature in pre_features:
            if feature not in df.columns:
                return jsonify({'error': f'Pre-feature "{feature}" not found in dataset'}), 400
        
        # Validate and convert treatment column
        success, message, df_converted, unique_values = validate_and_convert_treatment_column(df, treatment_col)
        
        if not success:
            return jsonify({
                'error': message,
                'unique_values': convert_to_serializable(unique_values.tolist() if unique_values is not None else []),
                'suggestion': 'Please select a column with exactly 2 unique values (e.g., Yes/No, 0/1, True/False) or a column that can be converted to binary.'
            }), 400
        
        df = df_converted
        treatment_conversion_message = message
        
        # Generate EDA visualizations BEFORE handling missing values
        eda_visualizations = generate_eda_visualizations(df, treatment_col, pre_features, outcome_col)
        
        # Check for missing values in pre_features
        missing_columns = []
        for feature in pre_features:
            if df[feature].isna().any():
                missing_columns.append(feature)
        
        # Handle missing values
        if missing_columns:
            print(f"Handling missing values in columns: {missing_columns} using strategy: {missing_strategy}")
            df, missing_info = handle_missing_values(df, pre_features, strategy=missing_strategy)
            
            # Check if we still have enough data after handling missing values
            if len(df) < 10:
                return jsonify({
                    'error': f'Too many missing values. After handling missing values, only {len(df)} rows remain. Need at least 10 rows.',
                    'missing_info': convert_to_serializable(missing_info)
                }), 400
        else:
            missing_info = {}
        
        # Check group sizes
        treatment_counts = df[treatment_col].value_counts()
        if any(treatment_counts < 5):
            return jsonify({
                'error': f'Both treatment and control groups need at least 5 observations. Current counts: {treatment_counts.to_dict()}',
                'treatment_counts': convert_to_serializable(treatment_counts.to_dict())
            }), 400
        
        # Convert treatment to binary if needed (should already be done by validate_and_convert_treatment_column)
        df['treatment'] = df[treatment_col].astype(int)
        
        # Separate control and treatment
        df_control = df[df.treatment == 0]
        df_treatment = df[df.treatment == 1]
        
        # T-test before matching (use outcome column if provided)
        if outcome_col and outcome_col in df.columns:
            control_outcome_before = float(df_control[outcome_col].mean())
            treatment_outcome_before = float(df_treatment[outcome_col].mean())
            _, p_before = ttest_ind(df_control[outcome_col], df_treatment[outcome_col])
            p_before = float(p_before)
        else:
            # If no outcome column, use first numeric column as example
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            if len(numeric_cols) > 0:
                outcome_col = numeric_cols[0]
                control_outcome_before = float(df_control[outcome_col].mean())
                treatment_outcome_before = float(df_treatment[outcome_col].mean())
                _, p_before = ttest_ind(df_control[outcome_col], df_treatment[outcome_col])
                p_before = float(p_before)
            else:
                return jsonify({'error': 'No numeric columns found for outcome analysis'}), 400
        
        # Prepare data for propensity score calculation
        X = df[pre_features]
        y = df['treatment']
        
        # Ensure no missing values remain
        if X.isna().any().any():
            return jsonify({
                'error': 'Missing values still present after imputation. Try a different missing value strategy.',
                'missing_columns': X.columns[X.isna().any()].tolist()
            }), 400
        
        # Calculate propensity scores
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X, y)
        pred_prob = lr.predict_proba(X)
        df['ps'] = pred_prob[:, 1]
        
        # Calculate logit of propensity score
        df['ps_logit'] = df['ps'].apply(lambda x: math.log(x / (1-x)) if x not in [0, 1] else math.log((x+1e-6)/(1-x+1e-6)))
        
        # Create treatment and control groups for matching
        df_treatment = df[df['treatment'] == 1].copy()
        df_control = df[df['treatment'] == 0].copy()
        
        # Set caliper and fit NearestNeighbors
        caliper = float(np.std(df.ps) * .75)
        knn = NearestNeighbors(n_neighbors=1, radius=caliper)
        knn.fit(df_control[['ps']])
        
        # Match each treatment unit to a control unit
        matched_treatment_indexes = []
        matched_control_indexes = []
        
        for idx, row in df_treatment.iterrows():
            ps_value = row['ps']
            distances, indices = knn.kneighbors([[ps_value]])
            
            if distances[0][0] <= caliper:
                control_idx = df_control.index[indices[0][0]]
                matched_treatment_indexes.append(idx)
                matched_control_indexes.append(control_idx)
        
        # Check if we got any matches
        if len(matched_treatment_indexes) == 0:
            return jsonify({'error': 'No matches found. Try adjusting the caliper or check your data.'}), 400
        
        # Retrieve matched observations
        matched_treatment = df.loc[matched_treatment_indexes].copy()
        matched_control = df.loc[matched_control_indexes].copy()
        
        # Combine matched pairs
        df_matched = pd.concat([matched_treatment, matched_control])
        df_matched_control = df_matched[df_matched.treatment == 0]
        df_matched_treatment = df_matched[df_matched.treatment == 1]
        
        # T-test after matching
        control_outcome_after = float(df_matched_control[outcome_col].mean())
        treatment_outcome_after = float(df_matched_treatment[outcome_col].mean())
        _, p_after = ttest_ind(df_matched_control[outcome_col], df_matched_treatment[outcome_col])
        p_after = float(p_after)
        
        # Calculate effect sizes
        effect_sizes = []
        for cl in pre_features:
            _, p_before_feat = ttest_ind(df_control[cl], df_treatment[cl])
            _, p_after_feat = ttest_ind(df_matched_control[cl], df_matched_treatment[cl])
            cohen_d_before = cohen_d(df_treatment[cl], df_control[cl])
            cohen_d_after = cohen_d(df_matched_treatment[cl], df_matched_control[cl])
            effect_sizes.append([cl, 'before', cohen_d_before, float(p_before_feat)])
            effect_sizes.append([cl, 'after', cohen_d_after, float(p_after_feat)])
        
        df_effect_sizes = pd.DataFrame(effect_sizes, columns=['feature', 'matching', 'effect_size', 'p-value'])
        
        # Check PSM success
        psm_results = check_psm_success(df_effect_sizes, df_matched, df)
        
        # Generate visualizations
        def create_ps_plot(data, title):
            try:
                plt.figure(figsize=(10, 5))
                sns.histplot(data=data, x='ps', hue='treatment', kde=True, bins=30)
                plt.title(title)
                plt.xlabel("Propensity Score")
                plt.ylabel("Count")
                
                img_buf = io.BytesIO()
                plt.savefig(img_buf, format='png', bbox_inches='tight')
                img_buf.seek(0)
                img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')
                plt.close()
                return img_base64
            except Exception as e:
                print(f"Plot creation failed: {str(e)}")
                return ""
        
        ps_before_base64 = create_ps_plot(df, "Propensity Score Distribution Before Matching")
        ps_after_base64 = create_ps_plot(df_matched, "Propensity Score Distribution After Matching")
        
        # Effect sizes plot
        def create_effect_sizes_plot(df_effect_sizes_sorted):
            try:
                plt.figure(figsize=(15, max(10, len(df_effect_sizes_sorted['feature'].unique()) * 0.25)))
                sns.barplot(data=df_effect_sizes_sorted, x='effect_size', y='feature', hue='matching', orient='h')
                plt.title("Effect Sizes of Covariates Before and After Matching")
                plt.axvline(0.1, color='gray', linestyle='--', label='Small Effect')
                plt.axvline(0.25, color='orange', linestyle='--', label='Medium Effect')
                plt.axvline(0.5, color='red', linestyle='--', label='Large Effect')
                plt.xlabel("Cohen's d Effect Size")
                plt.ylabel("Feature")
                plt.legend()
                
                img_buf = io.BytesIO()
                plt.savefig(img_buf, format='png', bbox_inches='tight')
                img_buf.seek(0)
                img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')
                plt.close()
                return img_base64
            except Exception as e:
                print(f"Effect sizes plot failed: {str(e)}")
                return ""
        
        df_effect_sizes_sorted = df_effect_sizes.sort_values(by='effect_size', ascending=False)
        effect_sizes_base64 = create_effect_sizes_plot(df_effect_sizes_sorted)
        
        # Prepare response - convert everything to serializable types
        response = {
            'treatment_counts': {
                'control': int(len(df_control)),
                'treatment': int(len(df_treatment))
            },
            'outcome_comparison': {
                'outcome_column': str(outcome_col),
                'before': {
                    'control_mean': control_outcome_before,
                    'treatment_mean': treatment_outcome_before,
                    'p_value': p_before
                },
                'after': {
                    'control_mean': control_outcome_after,
                    'treatment_mean': treatment_outcome_after,
                    'p_value': p_after
                }
            },
            'matching_results': {
                'matched_control': int(len(matched_control)),
                'matched_treatment': int(len(matched_treatment)),
                'caliper': caliper
            },
            'psm_evaluation': convert_to_serializable(psm_results),
            'visualizations': {
                'ps_before': ps_before_base64,
                'ps_after': ps_after_base64,
                'effect_sizes': effect_sizes_base64,
                'eda_visualizations': eda_visualizations  # Add EDA visualizations to response
            },
            'missing_values_handled': {
                'strategy_used': str(missing_strategy),
                'missing_info': convert_to_serializable(missing_info)
            },
            'treatment_column_info': {
                'original_column': treatment_col,
                'conversion_message': treatment_conversion_message,
                'final_values': {
                    'control': 0,
                    'treatment': 1
                }
            }
        }
        
        # Only run clustering if PSM passed and we have enough data
        if psm_results['passed'] and len(df) >= 10:
            clustering_results = perform_clustering_analysis(df, clustering_features)
            if 'error' not in clustering_results:
                response['clustering_results'] = convert_to_serializable(clustering_results)
            else:
                response['clustering_warning'] = str(clustering_results['error'])
        
        return jsonify(convert_to_serializable(response))
    
    except Exception as e:
        print(f"Error in psm_analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)