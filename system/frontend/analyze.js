// Enhanced scatter plot with better interactivity and styling
function createEnhancedScatterPlot(feature, summary, containerId) {
    const canvas = document.createElement('canvas');
    canvas.id = `chart-${feature.replace(/[^a-zA-Z0-9]/g, '-')}`;
    canvas.className = 'chart-container';
    
    const container = document.createElement('div');
    container.className = 'mb-6 bg-white rounded-xl p-6 border border-gray-200 card-hover';
    container.innerHTML = `
        <div class="flex items-center justify-between mb-4">
            <h4 class="font-semibold text-gray-900 flex items-center">
                <div class="w-3 h-3 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full mr-2"></div>
                ${feature}
            </h4>
            <div class="text-sm text-gray-500">Interactive Scatter Plot</div>
        </div>
    `;
    container.appendChild(canvas);
    
    document.getElementById(containerId).appendChild(container);
    
    // Extract data for enhanced scatter plot
    const clusters = Object.keys(summary.mean);
    const colorPalette = [
        'rgba(59, 130, 246, 0.8)',   // Blue
        'rgba(16, 185, 129, 0.8)',   // Green
        'rgba(245, 158, 11, 0.8)',   // Orange
        'rgba(139, 92, 246, 0.8)',   // Purple
        'rgba(236, 72, 153, 0.8)',   // Pink
        'rgba(6, 182, 212, 0.8)'     // Cyan
    ];
    
    const scatterData = {
        datasets: clusters.map((cluster, index) => {
            const points = [];
            const pointCount = Math.min(100, summary.count[cluster]); // Increased points for better visualization
            const mean = summary.mean[cluster];
            const std = summary.std[cluster];
            
            for (let i = 0; i < pointCount; i++) {
                // Use normal distribution for more realistic scatter
                const u1 = Math.random();
                const u2 = Math.random();
                const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
                const value = mean + z0 * std;
                
                points.push({
                    x: value,
                    y: index + (Math.random() - 0.5) * 0.6, // Jitter for better separation
                    cluster: cluster,
                    mean: mean,
                    std: std
                });
            }
            
            return {
                label: `Cluster ${cluster}`,
                data: points,
                backgroundColor: colorPalette[index % colorPalette.length],
                borderColor: colorPalette[index % colorPalette.length].replace('0.8', '1'),
                pointRadius: 4,
                pointHoverRadius: 6,
                pointBorderWidth: 1,
                pointHoverBorderWidth: 2
            };
        })
    };
    
    // Create enhanced scatter plot
    new Chart(canvas, {
        type: 'scatter',
        data: scatterData,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'point'
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: feature,
                        font: {
                            size: 14,
                            weight: 'bold'
                        }
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Cluster Groups',
                        font: {
                            size: 14,
                            weight: 'bold'
                        }
                    },
                    ticks: {
                        callback: function(value) {
                            return `Cluster ${Math.round(value)}`;
                        }
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: `Distribution of ${feature} across clusters`,
                    font: {
                        size: 16,
                        weight: 'bold'
                    },
                    padding: 20
                },
                legend: {
                    position: 'top',
                    labels: {
                        usePointStyle: true,
                        padding: 20
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: 'white',
                    bodyColor: 'white',
                    borderColor: 'rgba(255, 255, 255, 0.2)',
                    borderWidth: 1,
                    callbacks: {
                        title: function(context) {
                            return `Cluster ${context[0].raw.cluster}`;
                        },
                        label: function(context) {
                            return [
                                `Value: ${context.raw.x.toFixed(3)}`,
                                `Mean: ${context.raw.mean.toFixed(3)}`,
                                `Std Dev: ${context.raw.std.toFixed(3)}`
                            ];
                        }
                    }
                }
            },
            animation: {
                duration: 1000,
                easing: 'easeOutQuart'
            }
        }
    });
}



// Enhanced bar chart with better styling
function createEnhancedBarChart(feature, summary, containerId) {
    const canvas = document.createElement('canvas');
    canvas.id = `chart-${feature.replace(/[^a-zA-Z0-9]/g, '-')}`;
    canvas.className = 'chart-container';
    
    const container = document.createElement('div');
    container.className = 'mb-6 bg-white rounded-xl p-6 border border-gray-200 card-hover';
    container.innerHTML = `
        <div class="flex items-center justify-between mb-4">
            <h4 class="font-semibold text-gray-900 flex items-center">
                <div class="w-3 h-3 bg-gradient-to-r from-green-500 to-blue-600 rounded-full mr-2"></div>
                ${feature}
            </h4>
            <div class="text-sm text-gray-500">Categorical Distribution</div>
        </div>
    `;
    container.appendChild(canvas);
    
    document.getElementById(containerId).appendChild(container);
    
    // Extract data for enhanced bar chart
    const clusters = Object.keys(summary);
    const categories = new Set();
    
    clusters.forEach(cluster => {
        Object.keys(summary[cluster]).forEach(category => {
            categories.add(category);
        });
    });
    
    const categoryArray = Array.from(categories);
    const colorPalette = [
        'rgba(59, 130, 246, 0.8)',
        'rgba(16, 185, 129, 0.8)',
        'rgba(245, 158, 11, 0.8)',
        'rgba(139, 92, 246, 0.8)',
        'rgba(236, 72, 153, 0.8)',
        'rgba(6, 182, 212, 0.8)'
    ];
    
    const barData = {
        labels: categoryArray,
        datasets: clusters.map((cluster, index) => ({
            label: `Cluster ${cluster}`,
            data: categoryArray.map(category => summary[cluster][category] || 0),
            backgroundColor: colorPalette[index % colorPalette.length],
            borderColor: colorPalette[index % colorPalette.length].replace('0.8', '1'),
            borderWidth: 2,
            borderRadius: 4,
            borderSkipped: false
        }))
    };
    
    // Create enhanced bar chart
    new Chart(canvas, {
        type: 'bar',
        data: barData,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Categories',
                        font: {
                            size: 14,
                            weight: 'bold'
                        }
                    },
                    grid: {
                        display: false
                    }
                },
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Proportion',
                        font: {
                            size: 14,
                            weight: 'bold'
                        }
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: `Distribution of ${feature} across clusters`,
                    font: {
                        size: 16,
                        weight: 'bold'
                    },
                    padding: 20
                },
                legend: {
                    position: 'top',
                    labels: {
                        usePointStyle: true,
                        padding: 20
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: 'white',
                    bodyColor: 'white',
                    borderColor: 'rgba(255, 255, 255, 0.2)',
                    borderWidth: 1
                }
            },
            animation: {
                duration: 1000,
                easing: 'easeOutQuart'
            }
        }
    });
}

// Enhanced progress simulation
function simulateProgress() {
    const progressContainer = document.getElementById('progressContainer');
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');
    
    progressContainer.classList.remove('hidden');
    
    let progress = 0;
    const interval = setInterval(() => {
        progress += Math.random() * 15;
        if (progress > 100) progress = 100;
        
        progressBar.style.width = `${progress}%`;
        progressText.textContent = `${Math.round(progress)}%`;
        
        if (progress >= 100) {
            clearInterval(interval);
            setTimeout(() => {
                progressContainer.classList.add('hidden');
            }, 500);
        }
    }, 200);
}

// Main analysis function with enhanced UI feedback
document.getElementById('runAnalysis').addEventListener('click', async function() {
    try {
        // Enhanced loading state
        const button = document.getElementById('runAnalysis');
        const buttonText = document.getElementById('buttonText');
        const loadingSpinner = document.getElementById('loadingSpinner');
        
        button.disabled = true;
        button.classList.add('opacity-75', 'cursor-not-allowed');
        buttonText.textContent = 'Processing Analysis...';
        loadingSpinner.classList.remove('hidden');
        
        // Start progress simulation
        simulateProgress();
        
        const response = await fetch('http://127.0.0.1:5000/api/psm-analysis', {
            method: 'POST'
        });
        
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        
        const data = await response.json();
        
        // Reset button state
        button.disabled = false;
        button.classList.remove('opacity-75', 'cursor-not-allowed');
        buttonText.textContent = 'Analysis Complete ✓';
        loadingSpinner.classList.add('hidden');
        
        // Animate results appearance
        setTimeout(() => {
            buttonText.textContent = 'Run PSM Analysis';
        }, 2000);
        
        // Display results with enhanced formatting
        document.getElementById('controlCount').textContent = data.treatment_counts.control.toLocaleString();
        document.getElementById('treatmentCount').textContent = data.treatment_counts.treatment.toLocaleString();
        
        document.getElementById('controlBefore').textContent = `$${data.income_comparison.before.control_mean.toLocaleString(undefined, {minimumFractionDigits: 2})}`;
        document.getElementById('treatmentBefore').textContent = `$${data.income_comparison.before.treatment_mean.toLocaleString(undefined, {minimumFractionDigits: 2})}`;
        document.getElementById('pBefore').textContent = data.income_comparison.before.p_value.toFixed(4);
        
        document.getElementById('controlAfter').textContent = `$${data.income_comparison.after.control_mean.toLocaleString(undefined, {minimumFractionDigits: 2})}`;
        document.getElementById('treatmentAfter').textContent = `$${data.income_comparison.after.treatment_mean.toLocaleString(undefined, {minimumFractionDigits: 2})}`;
        document.getElementById('pAfter').textContent = data.income_comparison.after.p_value.toFixed(4);
        
        document.getElementById('matchedControl').textContent = data.matching_results.matched_control.toLocaleString();
        document.getElementById('matchedTreatment').textContent = data.matching_results.matched_treatment.toLocaleString();
        document.getElementById('caliper').textContent = data.matching_results.caliper.toFixed(4);
        
        // Enhanced PSM Evaluation with status indicators
        const psmEval = data.psm_evaluation;
        let evaluationHTML = `
            <div class="flex items-center mb-4">
                <div class="status-indicator ${psmEval.passed ? 'status-success' : 'status-danger'}"></div>
                <span class="font-semibold ${psmEval.passed ? 'text-green-700' : 'text-red-700'}">
                    Overall Status: ${psmEval.passed ? 'PASSED' : 'FAILED'}
                </span>
            </div>
        `;
        
        if (!psmEval.passed) {
            evaluationHTML += '<div class="bg-red-50 border border-red-200 rounded-lg p-4 mb-4">';
            evaluationHTML += '<h4 class="font-semibold text-red-800 mb-2">Issues Identified:</h4>';
            evaluationHTML += '<ul class="list-disc list-inside text-red-700 space-y-1">';
            psmEval.reasons.forEach(reason => {
                evaluationHTML += `<li>${reason}</li>`;
            });
            evaluationHTML += '</ul>';
            
            if (psmEval.failed_covariates.length > 0) {
                evaluationHTML += `<p class="mt-3 text-red-700"><strong>Unbalanced Covariates:</strong> ${psmEval.failed_covariates.join(', ')}</p>`;
            }
            evaluationHTML += '</div>';
        }
        document.getElementById('psmEvaluation').innerHTML = evaluationHTML;
        
        // Enhanced summaries with better formatting
        let effectSizeHTML = '<div class="space-y-2">';
        for (const [key, value] of Object.entries(psmEval.effect_size_summary)) {
            const status = Math.abs(value) < 0.1 ? 'text-green-600' : Math.abs(value) < 0.25 ? 'text-yellow-600' : 'text-red-600';
            effectSizeHTML += `
                <div class="flex justify-between items-center py-2 border-b border-gray-100">
                    <span class="font-medium text-gray-700">${key}:</span>
                    <span class="font-semibold ${status}">${value.toFixed(4)}</span>
                </div>
            `;
        }
        effectSizeHTML += '</div>';
        document.getElementById('effectSizeSummary').innerHTML = effectSizeHTML;
        
        let pValueHTML = '<div class="space-y-2">';
        for (const [key, value] of Object.entries(psmEval.p_value_summary)) {
            const status = value > 0.05 ? 'text-green-600' : 'text-red-600';
            pValueHTML += `
                <div class="flex justify-between items-center py-2 border-b border-gray-100">
                    <span class="font-medium text-gray-700">${key}:</span>
                    <span class="font-semibold ${status}">${value.toFixed(4)}</span>
                </div>
            `;
        }
        pValueHTML += '</div>';
        document.getElementById('pValueSummary').innerHTML = pValueHTML;
        
        // Enhanced overlap and sample size status
        const overlap = psmEval.overlap_status;
        let overlapHTML = `
            <div class="flex justify-between items-center py-2">
                <span class="font-medium text-gray-700">Max Quantile Difference:</span>
                <span class="font-semibold text-blue-600">${overlap.max_quantile_diff.toFixed(4)}</span>
            </div>
        `;
        document.getElementById('overlapStatus').innerHTML = overlapHTML;
        
        const sampleSize = psmEval.sample_size_status;
        let sampleSizeHTML = `
            <div class="space-y-2">
                <div class="flex justify-between items-center py-2 border-b border-gray-100">
                    <span class="font-medium text-gray-700">Original Treatment Size:</span>
                    <span class="font-semibold text-gray-900">${sampleSize.original_treatment_size.toLocaleString()}</span>
                </div>
                <div class="flex justify-between items-center py-2 border-b border-gray-100">
                    <span class="font-medium text-gray-700">Matched Treatment Size:</span>
                    <span class="font-semibold text-gray-900">${sampleSize.matched_treatment_size.toLocaleString()}</span>
                </div>
                <div class="flex justify-between items-center py-2">
                    <span class="font-medium text-gray-700">Retention Rate:</span>
                    <span class="font-semibold text-blue-600">${sampleSize.retention_rate.toFixed(1)}%</span>
                </div>
            </div>
        `;
        document.getElementById('sampleSizeStatus').innerHTML = sampleSizeHTML;
        
        // Load visualizations
        document.getElementById('psBeforeImg').src = `data:image/png;base64,${data.visualizations.ps_before}`;
        document.getElementById('psAfterImg').src = `data:image/png;base64,${data.visualizations.ps_after}`;
        document.getElementById('effectSizesImg').src = `data:image/png;base64,${data.visualizations.effect_sizes}`;
        
        // Enhanced clustering results
        if (psmEval.passed && data.clustering_results) {
            document.getElementById('clusteringResults').classList.remove('hidden');
            
            // Enhanced data cleaning summary
            const cleaning = data.clustering_results.data_cleaning;
            let cleaningHTML = `
                <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div class="text-center p-4 bg-blue-50 rounded-lg">
                        <div class="text-2xl font-bold text-blue-600">${cleaning.rows_before.toLocaleString()}</div>
                        <div class="text-sm text-blue-700">Rows Before</div>
                    </div>
                    <div class="text-center p-4 bg-green-50 rounded-lg">
                        <div class="text-2xl font-bold text-green-600">${cleaning.rows_after.toLocaleString()}</div>
                        <div class="text-sm text-green-700">Rows After</div>
                    </div>
                    <div class="text-center p-4 bg-orange-50 rounded-lg">
                        <div class="text-2xl font-bold text-orange-600">${cleaning.duplicates_removed.toLocaleString()}</div>
                        <div class="text-sm text-orange-700">Duplicates Removed</div>
                    </div>
                </div>
            `;
            document.getElementById('dataCleaningSummary').innerHTML = cleaningHTML;
            
            // Enhanced dimensionality reduction results
            const dimReduction = data.clustering_results.dimensionality_reduction;
            let dimReductionHTML = `
                <div class="mb-4 p-4 bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg">
                    <div class="flex items-center">
                        <div class="w-3 h-3 bg-blue-500 rounded-full mr-2"></div>
                        <span class="font-semibold text-gray-900">Best Model: ${dimReduction.best_model}</span>
                    </div>
                </div>
                <div class="overflow-hidden rounded-lg border border-gray-200">
                    <table class="min-w-full bg-white">
                        <thead class="bg-gradient-to-r from-gray-50 to-gray-100">
                            <tr>
                                <th class="py-3 px-4 text-left text-sm font-semibold text-gray-700">Model</th>
                                <th class="py-3 px-4 text-left text-sm font-semibold text-gray-700">Silhouette</th>
                                <th class="py-3 px-4 text-left text-sm font-semibold text-gray-700">Calinski-Harabasz</th>
                                <th class="py-3 px-4 text-left text-sm font-semibold text-gray-700">Davies-Bouldin</th>
                            </tr>
                        </thead>
                        <tbody class="divide-y divide-gray-200">
            `;
            
            dimReduction.metrics.forEach((metric, index) => {
                const isSelected = metric.Model === dimReduction.best_model;
                dimReductionHTML += `
                    <tr class="hover:bg-gray-50 transition-colors duration-200 ${isSelected ? 'bg-blue-50' : ''}">
                        <td class="py-3 px-4 font-medium ${isSelected ? 'text-blue-900' : 'text-gray-900'}">${metric.Model}</td>
                        <td class="py-3 px-4 text-gray-700">${metric.Silhouette.toFixed(4)}</td>
                        <td class="py-3 px-4 text-gray-700">${metric['Calinski-Harabasz'].toFixed(4)}</td>
                        <td class="py-3 px-4 text-gray-700">${metric['Davies-Bouldin'].toFixed(4)}</td>
                    </tr>
                `;
            });
            
            dimReductionHTML += `</tbody></table></div>`;
            document.getElementById('dimReductionResults').innerHTML = dimReductionHTML;
            
            // Load feature weights visualization
            document.getElementById('featureWeightsImg').src = `data:image/png;base64,${dimReduction.visualization}`;
            
            // Enhanced ensemble clustering results
            const ensemble = data.clustering_results.ensemble_clustering;
            let ensembleHTML = `
                <div class="mb-4 p-4 bg-gradient-to-r from-green-50 to-blue-50 rounded-lg">
                    <div class="flex items-center">
                        <div class="w-3 h-3 bg-green-500 rounded-full mr-2"></div>
                        <span class="font-semibold text-gray-900">Best Model: ${ensemble.best_model}</span>
                    </div>
                </div>
                <div class="overflow-hidden rounded-lg border border-gray-200">
                    <table class="min-w-full bg-white">
                        <thead class="bg-gradient-to-r from-gray-50 to-gray-100">
                            <tr>
                                <th class="py-3 px-4 text-left text-sm font-semibold text-gray-700">Model</th>
                                <th class="py-3 px-4 text-left text-sm font-semibold text-gray-700">Silhouette</th>
                                <th class="py-3 px-4 text-left text-sm font-semibold text-gray-700">Calinski-Harabasz</th>
                                <th class="py-3 px-4 text-left text-sm font-semibold text-gray-700">Davies-Bouldin</th>
                            </tr>
                        </thead>
                        <tbody class="divide-y divide-gray-200">
            `;
            
            ensemble.metrics.forEach((metric, index) => {
                const isSelected = metric.Model === ensemble.best_model;
                ensembleHTML += `
                    <tr class="hover:bg-gray-50 transition-colors duration-200 ${isSelected ? 'bg-green-50' : ''}">
                        <td class="py-3 px-4 font-medium ${isSelected ? 'text-green-900' : 'text-gray-900'}">${metric.Model}</td>
                        <td class="py-3 px-4 text-gray-700">${metric.Silhouette.toFixed(4)}</td>
                        <td class="py-3 px-4 text-gray-700">${metric['Calinski-Harabasz'].toFixed(4)}</td>
                        <td class="py-3 px-4 text-gray-700">${metric['Davies-Bouldin'].toFixed(4)}</td>
                    </tr>
                `;
            });
            
            ensembleHTML += `</tbody></table></div>`;
            document.getElementById('ensembleClusteringResults').innerHTML = ensembleHTML;
            
            // Load cluster visualizations
            document.getElementById('clustersImg').src = `data:image/png;base64,${ensemble.cluster_visualization}`;
            document.getElementById('clusterProfilesImg').src = `data:image/png;base64,${ensemble.cluster_profiles}`;
            
            // Enhanced cluster summaries with improved visualizations
            const clusterSummariesContainer = document.getElementById('clusterSummaries');
            clusterSummariesContainer.innerHTML = '';
            
            for (const [feature, summary] of Object.entries(ensemble.cluster_summaries)) {
                if (summary.mean && summary.std && summary.count) {
                    createEnhancedScatterPlot(feature, summary, 'clusterSummaries');
                } else {
                    createEnhancedBarChart(feature, summary, 'clusterSummaries');
                }
            }
        }
        
        // Show results with staggered animation
        document.getElementById('results').classList.remove('hidden');
        
    } catch (error) {
        console.error('Error:', error);
        
        // Enhanced error handling
        const errorMessage = document.createElement('div');
        errorMessage.className = 'fixed top-4 right-4 bg-red-500 text-white px-6 py-4 rounded-lg shadow-lg z-50 animate-slide-down';
        errorMessage.innerHTML = `
            <div class="flex items-center">
                <svg class="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                </svg>
                <span>Analysis failed. Please try again.</span>
            </div>
        `;
        document.body.appendChild(errorMessage);
        
        setTimeout(() => {
            errorMessage.remove();
        }, 5000);
        
        // Reset button state
        const button = document.getElementById('runAnalysis');
        const buttonText = document.getElementById('buttonText');
        const loadingSpinner = document.getElementById('loadingSpinner');
        const progressContainer = document.getElementById('progressContainer');
        
        button.disabled = false;
        button.classList.remove('opacity-75', 'cursor-not-allowed');
        buttonText.textContent = 'Run PSM Analysis';
        loadingSpinner.classList.add('hidden');
        progressContainer.classList.add('hidden');
    }
});
