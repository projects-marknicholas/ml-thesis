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

app = Flask(__name__)
CORS(app)

# Function to calculate Cohen's d for independent samples
def cohen_d(d1, d2):
    n1, n2 = len(d1), len(d2)
    s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
    s = math.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    u1, u2 = np.mean(d1), np.mean(d2)
    return (u1 - u2) / s

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
    
    # Store summaries
    results['effect_size_summary'] = after_matching['effect_size'].describe().to_dict()
    results['p_value_summary'] = after_matching['p-value'].describe().to_dict()
    
    # 2. Check propensity score overlap
    treatment_ps = df_matched[df_matched[treatment_col] == 1][ps_col]
    control_ps = df_matched[df_matched[treatment_col] == 0][ps_col]
    
    # Compare quantiles to check overlap
    treatment_quantiles = treatment_ps.quantile([0.1, 0.5, 0.9]).to_dict()
    control_quantiles = control_ps.quantile([0.1, 0.5, 0.9]).to_dict()
    
    # Check if quantiles are within a reasonable range (e.g., 0.1 difference)
    quantile_diff = max([abs(treatment_quantiles[q] - control_quantiles[q]) for q in treatment_quantiles.keys()])
    if quantile_diff > 0.2:
        results['passed'] = False
        results['reasons'].append(f"Propensity score distributions do not overlap well (max quantile difference: {quantile_diff:.2f}).")
    
    results['overlap_status'] = {
        'treatment_quantiles': treatment_quantiles,
        'control_quantiles': control_quantiles,
        'max_quantile_diff': quantile_diff
    }
    
    # 3. Check sample size retention
    original_treatment_size = len(df[df[treatment_col] == 1])
    matched_treatment_size = len(df_matched[df_matched[treatment_col] == 1])
    retention_rate = matched_treatment_size / original_treatment_size * 100
    
    if retention_rate < 80:
        results['passed'] = False
        results['reasons'].append(f"Low sample retention rate ({retention_rate:.1f}%).")
    
    results['sample_size_status'] = {
        'original_treatment_size': original_treatment_size,
        'matched_treatment_size': matched_treatment_size,
        'retention_rate': retention_rate
    }
    
    return results

@app.route('/api/psm-analysis', methods=['POST'])
def psm_analysis():
    # Load data
    df = pd.read_csv('../../datasets/bfar.csv')
    
    # Pre-Program Features
    pre_features = [
        'D1.1:A_BIKE', 'D1.1-A_QTY', 'D1.2:A_MOTORC', 'D1.2-A_QTY', 'D1.3:A_TRICYCLE', 'D1.3-A_QTY',
        'D1.4:A_CAR', 'D1.4-A_QTY', 'D1.5:A_JEEP', 'D1.5-A_QTY', 'D1.6:A_TRUCK', 'D1.6-A_QTY',
        'D1.7:A_OTHERS', 'D1.7-A_QTY', 'D2.1:A_TV', 'D2.1-A_QTY', 'D2.2:A_DVD', 'D2.2-A_QTY',
        'D2.3:A_WASH-M', 'D2.3-A_QTY', 'D2.4:A_AC', 'D2.4-A_QTY', 'D2.5:A_E-FAN', 'D2.5-A_QTY',
        'D2.6:A_FRIDGE', 'D2.6-A_QTY', 'D2.7:A_STOVE', 'D2.7-A_QTY', 'D2.8:A_E-HEATER', 'D2.8-A_QTY',
        'D2.9:A_FURNITURE', 'D2.9-A_QTY', 'D2.10:A_OTHERS', 'D2.10-A_QTY', 'D3.1:A_CP', 'D3.1-A_QTY',
        'D3.2:A_LANDLINE', 'D3.2-A_QTY', 'D3.3:A_COMPUTER', 'D3.3-A_QTY', 'D3.4:A_OTHERS', 'D3.4-A_QTY',
        'E1:A_DRINK-H2O', 'E2:A_DOMESTIC-H2O', 'E3:A_POWER-SUP', 'E4:A_COOK-FUEL', 'E5:A_NET-SUBS',
        'F1:A_HOUSE-OWN', 'F2:A_HOUSE-ACQ', 'F3:A_HOUSE-BUILT', 'F4:A_OTHER-RP', 'G1:A_SSS', 'G2:A_GSIS',
        'G3:A_PhilHealth', 'G4:A_PN-IN', 'G5:A_LIFE-IN', 'G6:A_HEALTH-IN'
    ]
    
    # Create treatment variable
    df['treatment'] = df['Y_BOAT-RE'].notna().astype(int)
    
    # Separate control and treatment
    df_control = df[df.treatment == 0]
    df_treatment = df[df.treatment == 1]
    
    # T-test before matching
    control_income_before = df_control['C5:TOT_INCOME/B'].mean()
    treatment_income_before = df_treatment['C5:TOT_INCOME/B'].mean()
    _, p_before = ttest_ind(df_control['C5:TOT_INCOME/B'], df_treatment['C5:TOT_INCOME/B'])
    
    # Prepare data for propensity score calculation
    X = df[pre_features]
    y = df['treatment']
    
    # Calculate propensity scores
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X, y)
    pred_prob = lr.predict_proba(X)
    df['ps'] = pred_prob[:, 1]
    
    # Calculate logit of propensity score
    df['ps_logit'] = df['ps'].apply(lambda x: math.log(x / (1-x)) if x not in [0, 1] else math.log((x+1e-6)/(1-x+1e-6)))
    
    # Create treatment and control groups for matching
    df_treatment = df[df['treatment'] == 1].copy()
    df_control = df[df['treatment'] == 0].copy()
    
    # Set caliper and fit NearestNeighbors
    caliper = np.std(df.ps) * .75
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
    
    # Retrieve matched observations
    matched_treatment = df.loc[matched_treatment_indexes].copy()
    matched_control = df.loc[matched_control_indexes].copy()
    
    # Combine matched pairs
    df_matched = pd.concat([matched_treatment, matched_control])
    df_matched_control = df_matched[df_matched.treatment == 0]
    df_matched_treatment = df_matched[df_matched.treatment == 1]
    
    # T-test after matching
    control_income_after = df_matched_control['C5:TOT_INCOME/B'].mean()
    treatment_income_after = df_matched_treatment['C5:TOT_INCOME/B'].mean()
    _, p_after = ttest_ind(df_matched_control['C5:TOT_INCOME/B'], df_matched_treatment['C5:TOT_INCOME/B'])
    
    # Calculate effect sizes
    effect_sizes = []
    for cl in pre_features:
        _, p_before_feat = ttest_ind(df_control[cl], df_treatment[cl])
        _, p_after_feat = ttest_ind(df_matched_control[cl], df_matched_treatment[cl])
        cohen_d_before = cohen_d(df_treatment[cl], df_control[cl])
        cohen_d_after = cohen_d(df_matched_treatment[cl], df_matched_control[cl])
        effect_sizes.append([cl, 'before', cohen_d_before, p_before_feat])
        effect_sizes.append([cl, 'after', cohen_d_after, p_after_feat])
    
    df_effect_sizes = pd.DataFrame(effect_sizes, columns=['feature', 'matching', 'effect_size', 'p-value'])
    
    # Check PSM success
    psm_results = check_psm_success(df_effect_sizes, df_matched, df)
    
    # Generate visualizations
    plt.figure(figsize=(10, 5))
    sns.histplot(data=df, x='ps', hue='treatment', kde=True, bins=30)
    plt.title("Propensity Score Distribution Before Matching")
    plt.xlabel("Propensity Score")
    plt.ylabel("Count")
    
    # Save plot to base64
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    ps_before_base64 = base64.b64encode(img_buf.read()).decode('utf-8')
    plt.close()
    
    # After matching plot
    plt.figure(figsize=(10, 5))
    sns.histplot(data=df_matched, x='ps', hue='treatment', kde=True, bins=30)
    plt.title("Propensity Score Distribution After Matching")
    plt.xlabel("Propensity Score")
    plt.ylabel("Count")
    
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    ps_after_base64 = base64.b64encode(img_buf.read()).decode('utf-8')
    plt.close()
    
    # Effect sizes plot
    df_effect_sizes_sorted = df_effect_sizes.sort_values(by='effect_size', ascending=False)
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
    effect_sizes_base64 = base64.b64encode(img_buf.read()).decode('utf-8')
    plt.close()
    
    # Prepare response
    response = {
        'treatment_counts': {
            'control': len(df_control),
            'treatment': len(df_treatment)
        },
        'income_comparison': {
            'before': {
                'control_mean': control_income_before,
                'treatment_mean': treatment_income_before,
                'p_value': p_before
            },
            'after': {
                'control_mean': control_income_after,
                'treatment_mean': treatment_income_after,
                'p_value': p_after
            }
        },
        'matching_results': {
            'matched_control': len(matched_control),
            'matched_treatment': len(matched_treatment),
            'caliper': caliper
        },
        'psm_evaluation': psm_results,
        'visualizations': {
            'ps_before': ps_before_base64,
            'ps_after': ps_after_base64,
            'effect_sizes': effect_sizes_base64
        }
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)