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
import warnings
warnings.filterwarnings('ignore')

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

def generate_cluster_summaries(df_cleaned, final_labels):
    """Generate comprehensive cluster summaries"""
    df_cleaned = df_cleaned.copy()
    df_cleaned['final_cluster'] = final_labels
    
    cluster_summaries = {}
    
    for feature in df_cleaned.columns:
        if feature == 'final_cluster':
            continue
        
        if df_cleaned[feature].dtype in ['int64', 'float64']:
            # For numeric features, calculate statistics
            feature_summary = df_cleaned.groupby('final_cluster')[feature].agg([
                'count', 'mean', 'std', 'min', 'max', 'median'
            ]).round(4).to_dict()
        else:
            # For categorical features, calculate value distributions
            value_counts = df_cleaned.groupby('final_cluster')[feature].value_counts(normalize=True)
            value_counts = value_counts.unstack(fill_value=0).round(4)
            
            # Convert to dictionary format
            feature_summary = {}
            for cluster in value_counts.index:
                cluster_dict = {}
                for val in value_counts.columns:
                    if value_counts.loc[cluster, val] > 0:  # Only include values that exist
                        cluster_dict[str(val)] = float(value_counts.loc[cluster, val])
                feature_summary[str(cluster)] = cluster_dict
        
        cluster_summaries[feature] = feature_summary
    
    return cluster_summaries

def perform_clustering_analysis(df):
    """Perform clustering analysis on the dataset"""
    # Data cleaning and preprocessing
    df_cleaned = df.copy()
    
    # Drop duplicates
    rows_before = len(df_cleaned)
    df_cleaned = df_cleaned.drop_duplicates()
    rows_after = len(df_cleaned)
    
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
    
    # Dimensionality reduction models
    K = 3
    df_num = df_cleaned.select_dtypes(include='number').copy()
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df_num)
    
    metrics_summary = []
    
    def evaluate_clustering(X, labels, name):
        silhouette = silhouette_score(X, labels)
        calinski = calinski_harabasz_score(X, labels)
        davies = davies_bouldin_score(X, labels)
        
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
    
    # PCA
    pca_model = PCA(n_components=2)
    X_pca = pca_model.fit_transform(X_scaled)
    kmeans_pca = KMeans(n_clusters=K, random_state=42, n_init=10)
    labels_pca = kmeans_pca.fit_predict(X_pca)
    pca_results = evaluate_clustering(X_pca, labels_pca, "PCA")
    
    # Entropy Weighted K-Means
    def entropy_weights(X):
        X_norm = X / (X.sum(axis=0) + 1e-9)
        X_norm = np.where(X_norm == 0, 1e-9, X_norm)
        entropy = -np.sum(X_norm * np.log(X_norm), axis=0) / np.log(len(X))
        D = 1 - entropy
        weights = D / D.sum()
        return weights
    
    weights = entropy_weights(X_scaled)
    X_entropy = X_scaled * weights
    kmeans_entropy = KMeans(n_clusters=K, random_state=42, n_init=10)
    labels_entropy = kmeans_entropy.fit_predict(X_entropy)
    entropy_results = evaluate_clustering(X_entropy, labels_entropy, "Entropy-Weighted")
    
    # Select best model
    metrics_df = pd.DataFrame(metrics_summary)
    scoring_df = metrics_df.copy()
    scoring_df['Inverse Davies-Bouldin'] = 1 / (scoring_df['Davies-Bouldin'] + 1e-6)
    scaler = MinMaxScaler()
    scaled_metrics = scaler.fit_transform(scoring_df[['Silhouette', 'Calinski-Harabasz', 'Inverse Davies-Bouldin']])
    scoring_df['Composite Score'] = scaled_metrics.mean(axis=1)
    best_model = scoring_df.loc[scoring_df['Composite Score'].idxmax()]
    
    # Extract feature weights from best model
    if best_model['Model'] == "PCA":
        pca_weights = np.abs(pca_model.components_[:2]).mean(axis=0)
        feature_weights = pd.Series(pca_weights, index=df_num.columns).sort_values(ascending=False)
    elif best_model['Model'] == "Entropy-Weighted":
        feature_weights = pd.Series(weights, index=df_num.columns).sort_values(ascending=False)
    
    # Ensemble clustering
    metrics_summary_2 = []
    
    def evaluate_model(X, labels, name):
        silhouette = silhouette_score(X, labels)
        calinski = calinski_harabasz_score(X, labels)
        davies = davies_bouldin_score(X, labels)
        
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
    
    # Separate numeric and categorical
    df_num_ensemble = df_cleaned.select_dtypes(include='number')
    df_cat = df_cleaned.select_dtypes(include='object')
    
    # Normalize numeric data
    scaler = MinMaxScaler()
    X_num_scaled = scaler.fit_transform(df_num_ensemble)
    
    # Label encode categorical data
    label_encoders = {}
    df_cat_encoded = df_cat.copy()
    for col in df_cat.columns:
        le = LabelEncoder()
        df_cat_encoded[col] = le.fit_transform(df_cat[col])
        label_encoders[col] = le
    
    X_cat_encoded = df_cat_encoded.values
    
    # Combine numeric and categorical
    X_mixed = np.concatenate([X_num_scaled, X_cat_encoded], axis=1)
    
    # Weighted KMeans
    weights_ensemble = entropy_weights(X_num_scaled)
    X_weighted = X_num_scaled * weights_ensemble
    kmeans_weighted = KMeans(n_clusters=K, random_state=42, n_init=10)
    labels_wkmeans = kmeans_weighted.fit_predict(X_weighted)
    wkmeans_results = evaluate_model(X_weighted, labels_wkmeans, "Weighted KMeans")
    
    # K-Prototypes
    kproto = KPrototypes(n_clusters=K, init='Cao', n_init=5, verbose=0)
    labels_kproto = kproto.fit_predict(X_mixed, categorical=list(range(X_num_scaled.shape[1], X_mixed.shape[1])))
    kproto_results = evaluate_model(X_mixed, labels_kproto, "K-Prototypes")
    
    # Hierarchical Clustering
    hierarchical = AgglomerativeClustering(n_clusters=K, linkage='ward')
    labels_hier = hierarchical.fit_predict(X_num_scaled)
    hier_results = evaluate_model(X_num_scaled, labels_hier, "Hierarchical Clustering")
    
    # Select best ensemble model
    metrics_df2 = pd.DataFrame(metrics_summary_2)
    scoring_df2 = metrics_df2.copy()
    scoring_df2['Inverse Davies-Bouldin'] = 1 / (scoring_df2['Davies-Bouldin'] + 1e-6)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(scoring_df2[['Silhouette', 'Calinski-Harabasz', 'Inverse Davies-Bouldin']])
    scoring_df2['Composite Score'] = scaled.mean(axis=1)
    best_model2 = scoring_df2.loc[scoring_df2['Composite Score'].idxmax()]
    
    # Apply final clustering
    final_labels = None
    if best_model2['Model'] == "Weighted KMeans":
        final_model = KMeans(n_clusters=K, random_state=42, n_init=10)
        final_labels = final_model.fit_predict(X_weighted)
    elif best_model2['Model'] == "K-Prototypes":
        final_model = KPrototypes(n_clusters=K, init='Cao', n_init=5, verbose=0)
        final_labels = final_model.fit_predict(X_mixed, categorical=list(range(X_num_scaled.shape[1], X_mixed.shape[1])))
    elif best_model2['Model'] == "Hierarchical Clustering":
        final_model = AgglomerativeClustering(n_clusters=K, linkage='ward')
        final_labels = final_model.fit_predict(X_num_scaled)
    
    # Generate cluster summaries
    cluster_summaries = generate_cluster_summaries(df_cleaned, final_labels)
    
    # Generate visualizations
    plt.figure(figsize=(10, 5))
    feature_weights.head(10).plot(kind='barh')
    plt.gca().invert_yaxis()
    plt.title(f"Top 10 Feature Weights - {best_model['Model']}")
    plt.xlabel("Weight")
    plt.tight_layout()
    
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    feature_weights_base64 = base64.b64encode(img_buf.read()).decode('utf-8')
    plt.close()
    
    # Cluster visualization
    pca_vis = PCA(n_components=2)
    if best_model2['Model'] == "Weighted KMeans":
        X_vis = pca_vis.fit_transform(X_weighted)
    else:
        X_vis = pca_vis.fit_transform(X_num_scaled)
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_vis[:, 0], y=X_vis[:, 1], hue=final_labels, palette="Set2", s=60)
    plt.title(f"Clusters by {best_model2['Model']}")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend(title="Cluster")
    plt.tight_layout()
    
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    clusters_base64 = base64.b64encode(img_buf.read()).decode('utf-8')
    plt.close()
    
    # Generate cluster profile visualization
    # Select top 5 most important features for profiling
    top_features = list(feature_weights.head(5).index)
    
    # Create a radar chart for cluster profiles
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    
    # Add final_cluster to df_cleaned for grouping
    df_cleaned_with_clusters = df_cleaned.copy()
    df_cleaned_with_clusters['final_cluster'] = final_labels
    
    # Normalize feature values for radar chart
    cluster_means = df_cleaned_with_clusters.groupby('final_cluster')[top_features].mean()
    cluster_means_normalized = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())
    
    # Number of variables
    categories = list(cluster_means_normalized.columns)
    N = len(categories)
    
    # Calculate angle for each category
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    # Plot each cluster
    for cluster in cluster_means_normalized.index:
        values = cluster_means_normalized.loc[cluster].values.tolist()
        values += values[:1]  # Complete the circle
        ax.plot(angles, values, linewidth=2, label=f'Cluster {cluster}')
        ax.fill(angles, values, alpha=0.25)
    
    # Add labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title('Cluster Profiles (Top 5 Features)', size=16, y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches='tight')
    img_buf.seek(0)
    cluster_profiles_base64 = base64.b64encode(img_buf.read()).decode('utf-8')
    plt.close()
    
    return {
        'dimensionality_reduction': {
            'best_model': best_model['Model'],
            'metrics': metrics_summary,
            'feature_weights': feature_weights.to_dict(),
            'top_features': feature_weights.head(10).to_dict(),
            'visualization': feature_weights_base64
        },
        'ensemble_clustering': {
            'best_model': best_model2['Model'],
            'metrics': metrics_summary_2,
            'cluster_visualization': clusters_base64,
            'cluster_profiles': cluster_profiles_base64,
            'cluster_summaries': cluster_summaries
        },
        'data_cleaning': {
            'rows_before': rows_before,
            'rows_after': rows_after,
            'duplicates_removed': rows_before - rows_after
        }
    }

@app.route('/api/psm-analysis', methods=['POST'])
def psm_analysis():
    try:
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
        
        # Only run clustering if PSM passed
        if psm_results['passed']:
            clustering_results = perform_clustering_analysis(df)
            response['clustering_results'] = clustering_results
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error in psm_analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)