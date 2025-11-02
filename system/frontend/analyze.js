// Global variables
let currentDatasetId = null;
let selectedPreFeatures = [];
let selectedClusteringFeatures = [];
let allClusterFeatures = [];

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

// File upload handling
document.getElementById('csvFile').addEventListener('change', async function(event) {
    try {
        const file = event.target.files[0];
        if (!file) return;

        // Show loading state
        const fileInfo = document.getElementById('fileInfo');
        fileInfo.classList.remove('hidden');
        fileInfo.innerHTML = `
            <div class="flex items-center justify-center p-4">
                <div class="loading-spinner mr-3"></div>
                <span class="text-blue-700">Uploading and analyzing dataset...</span>
            </div>
        `;

        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch('http://127.0.0.1:5000/api/upload-dataset', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Upload failed');
        }

        const data = await response.json();
        currentDatasetId = data.dataset_id;

        // Update file info
        fileInfo.innerHTML = `
            <div class="flex items-center justify-between">
                <div>
                    <p class="font-medium text-blue-900">${data.filename}</p>
                    <p class="text-sm text-blue-700">${data.shape[0]} rows × ${data.shape[1]} columns</p>
                </div>
                <button id="removeFile" class="text-red-600 hover:text-red-800">
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                    </svg>
                </button>
            </div>
        `;

        // Populate configuration panel
        populateConfigurationPanel(data);
        document.getElementById('configPanel').classList.remove('hidden');

        // Add remove file event listener
        document.getElementById('removeFile').addEventListener('click', removeFile);

    } catch (error) {
        console.error('Upload error:', error);
        showError(error.message || 'Failed to upload file. Please try again.');
        document.getElementById('fileInfo').classList.add('hidden');
    }
});

// Remove file function
function removeFile() {
    currentDatasetId = null;
    selectedPreFeatures = [];
    selectedClusteringFeatures = [];
    allClusterFeatures = [];
    document.getElementById('csvFile').value = '';
    document.getElementById('fileInfo').classList.add('hidden');
    document.getElementById('configPanel').classList.add('hidden');
    document.getElementById('results').classList.add('hidden');
}

// Select All functionality
function setupSelectAll() {
    const selectAllBtn = document.getElementById('selectAllBtn');
    const deselectAllBtn = document.getElementById('deselectAllBtn');
    const selectAllClusteringBtn = document.getElementById('selectAllClusteringBtn');
    const deselectAllClusteringBtn = document.getElementById('deselectAllClusteringBtn');
    
    if (selectAllBtn && deselectAllBtn) {
        selectAllBtn.addEventListener('click', function() {
            selectedPreFeatures = [];
            document.querySelectorAll('.feature-checkbox').forEach(checkbox => {
                checkbox.checked = true;
                if (!selectedPreFeatures.includes(checkbox.value)) {
                    selectedPreFeatures.push(checkbox.value);
                }
            });
            updateRunButtonState();
            
            // Show success feedback
            showSuccess('All features selected!');
        });
        
        deselectAllBtn.addEventListener('click', function() {
            selectedPreFeatures = [];
            document.querySelectorAll('.feature-checkbox').forEach(checkbox => {
                checkbox.checked = false;
            });
            updateRunButtonState();
            
            // Show feedback
            showInfo('All features deselected');
        });
    }
    
    if (selectAllClusteringBtn && deselectAllClusteringBtn) {
        selectAllClusteringBtn.addEventListener('click', function() {
            selectedClusteringFeatures = [...allClusterFeatures];
            document.querySelectorAll('.clustering-feature-checkbox').forEach(checkbox => {
                checkbox.checked = true;
            });
            
            // Show success feedback
            showSuccess('All clustering features selected!');
        });
        
        deselectAllClusteringBtn.addEventListener('click', function() {
            selectedClusteringFeatures = [];
            document.querySelectorAll('.clustering-feature-checkbox').forEach(checkbox => {
                checkbox.checked = false;
            });
            
            // Show feedback
            showInfo('All clustering features deselected');
        });
    }
}

// Populate configuration panel with dataset columns
function populateConfigurationPanel(datasetInfo) {
    // Populate treatment column dropdown
    const treatmentSelect = document.getElementById('treatmentCol');
    treatmentSelect.innerHTML = '<option value="">Select treatment column</option>';
    datasetInfo.columns.forEach(col => {
        const option = document.createElement('option');
        option.value = col;
        option.textContent = col;
        treatmentSelect.appendChild(option);
    });

    // Populate outcome column dropdown
    const outcomeSelect = document.getElementById('outcomeCol');
    outcomeSelect.innerHTML = '<option value="">Select outcome column</option>';
    datasetInfo.numeric_columns.forEach(col => {
        const option = document.createElement('option');
        option.value = col;
        option.textContent = col;
        outcomeSelect.appendChild(option);
    });

    // Populate pre-features checkboxes
    const preFeaturesContainer = document.getElementById('preFeaturesContainer');
    preFeaturesContainer.innerHTML = '';
    datasetInfo.columns.forEach(col => {
        const div = document.createElement('div');
        div.className = 'flex items-center space-x-2';
        div.innerHTML = `
            <input type="checkbox" id="feature-${col}" value="${col}" class="feature-checkbox rounded border-gray-300 text-blue-600 focus:ring-blue-500">
            <label for="feature-${col}" class="text-sm text-gray-700">${col}</label>
        `;
        preFeaturesContainer.appendChild(div);
    });

    // Populate clustering features checkboxes
    const clusteringFeaturesContainer = document.getElementById('clusteringFeaturesContainer');
    clusteringFeaturesContainer.innerHTML = '';
    allClusterFeatures = [...datasetInfo.columns]; // Store all available features
    
    datasetInfo.columns.forEach(col => {
        const div = document.createElement('div');
        div.className = 'flex items-center space-x-2';
        div.innerHTML = `
            <input type="checkbox" id="clustering-feature-${col}" value="${col}" class="clustering-feature-checkbox rounded border-gray-300 text-purple-600 focus:ring-purple-500">
            <label for="clustering-feature-${col}" class="text-sm text-gray-700">${col}</label>
        `;
        clusteringFeaturesContainer.appendChild(div);
    });

    // Add event listeners to checkboxes
    document.querySelectorAll('.feature-checkbox').forEach(checkbox => {
        checkbox.addEventListener('change', function() {
            if (this.checked) {
                selectedPreFeatures.push(this.value);
            } else {
                selectedPreFeatures = selectedPreFeatures.filter(f => f !== this.value);
            }
            updateRunButtonState();
        });
    });

    // Add event listeners to clustering checkboxes
    document.querySelectorAll('.clustering-feature-checkbox').forEach(checkbox => {
        checkbox.addEventListener('change', function() {
            if (this.checked) {
                selectedClusteringFeatures.push(this.value);
            } else {
                selectedClusteringFeatures = selectedClusteringFeatures.filter(f => f !== this.value);
            }
        });
    });

    // Setup select all functionality
    setupSelectAll();

    // Add event listeners to dropdowns
    treatmentSelect.addEventListener('change', updateRunButtonState);
    outcomeSelect.addEventListener('change', updateRunButtonState);
}

// Update run button state based on configuration
function updateRunButtonState() {
    const runButton = document.getElementById('runAnalysis');
    const treatmentCol = document.getElementById('treatmentCol').value;
    const hasPreFeatures = selectedPreFeatures.length > 0;

    runButton.disabled = !treatmentCol || !hasPreFeatures;
    
    if (runButton.disabled) {
        runButton.classList.add('opacity-50', 'cursor-not-allowed');
    } else {
        runButton.classList.remove('opacity-50', 'cursor-not-allowed');
    }
}

// Show error message
function showError(message) {
    const errorMessage = document.createElement('div');
    errorMessage.className = 'fixed top-4 right-4 bg-red-500 text-white px-6 py-4 rounded-lg shadow-lg z-50 animate-slide-down';
    errorMessage.innerHTML = `
        <div class="flex items-center">
            <svg class="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
            </svg>
            <span>${message}</span>
        </div>
    `;
    document.body.appendChild(errorMessage);
    
    setTimeout(() => {
        errorMessage.remove();
    }, 5000);
}

// Show success message
function showSuccess(message) {
    const successMessage = document.createElement('div');
    successMessage.className = 'fixed top-4 right-4 bg-green-500 text-white px-6 py-4 rounded-lg shadow-lg z-50 animate-slide-down';
    successMessage.innerHTML = `
        <div class="flex items-center">
            <svg class="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
            </svg>
            <span>${message}</span>
        </div>
    `;
    document.body.appendChild(successMessage);
    
    setTimeout(() => {
        successMessage.remove();
    }, 3000);
}

// Show info message
function showInfo(message) {
    const infoMessage = document.createElement('div');
    infoMessage.className = 'fixed top-4 right-4 bg-blue-500 text-white px-6 py-4 rounded-lg shadow-lg z-50 animate-slide-down';
    infoMessage.innerHTML = `
        <div class="flex items-center">
            <svg class="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
            </svg>
            <span>${message}</span>
        </div>
    `;
    document.body.appendChild(infoMessage);
    
    setTimeout(() => {
        infoMessage.remove();
    }, 3000);
}

// Setup clustering feature filter
function setupClusteringFeatureFilter(clusterSummaries) {
    const featureFilterButtons = document.getElementById('featureFilterButtons');
    const featureSearch = document.getElementById('featureSearch');
    const selectAllFeatures = document.getElementById('selectAllFeatures');
    const clearFeatures = document.getElementById('clearFeatures');
    
    // Clear existing buttons
    featureFilterButtons.innerHTML = '';
    
    // Get all available features from cluster summaries
    const availableFeatures = Object.keys(clusterSummaries);
    
    // Create filter buttons for each feature
    availableFeatures.forEach(feature => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'feature-filter-btn px-3 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors';
        button.textContent = feature;
        button.dataset.feature = feature;
        
        button.addEventListener('click', function() {
            this.classList.toggle('bg-gray-200');
            this.classList.toggle('bg-blue-500');
            this.classList.toggle('text-gray-700');
            this.classList.toggle('text-white');
            
            // Filter cluster summaries display
            filterClusterSummaries(clusterSummaries);
        });
        
        featureFilterButtons.appendChild(button);
    });
    
    // Setup search functionality
    featureSearch.addEventListener('input', function() {
        const searchTerm = this.value.toLowerCase();
        document.querySelectorAll('.feature-filter-btn').forEach(btn => {
            const feature = btn.dataset.feature.toLowerCase();
            if (feature.includes(searchTerm)) {
                btn.style.display = 'inline-block';
            } else {
                btn.style.display = 'none';
            }
        });
    });
    
    // Setup select all functionality
    selectAllFeatures.addEventListener('click', function() {
        document.querySelectorAll('.feature-filter-btn').forEach(btn => {
            btn.classList.add('bg-blue-500', 'text-white');
            btn.classList.remove('bg-gray-200', 'text-gray-700');
        });
        filterClusterSummaries(clusterSummaries);
    });
    
    // Setup clear functionality
    clearFeatures.addEventListener('click', function() {
        document.querySelectorAll('.feature-filter-btn').forEach(btn => {
            btn.classList.remove('bg-blue-500', 'text-white');
            btn.classList.add('bg-gray-200', 'text-gray-700');
        });
        filterClusterSummaries(clusterSummaries);
    });
    
    // Initially select all features
    selectAllFeatures.click();
}

// Filter cluster summaries based on selected features
function filterClusterSummaries(clusterSummaries) {
    const clusterSummariesContainer = document.getElementById('clusterSummaries');
    clusterSummariesContainer.innerHTML = '';
    
    // Get selected features
    const selectedFeatures = Array.from(document.querySelectorAll('.feature-filter-btn.bg-blue-500'))
        .map(btn => btn.dataset.feature);
    
    // If no features selected, show all
    const featuresToShow = selectedFeatures.length > 0 ? selectedFeatures : Object.keys(clusterSummaries);
    
    // Create visualizations for selected features only
    featuresToShow.forEach(feature => {
        if (clusterSummaries[feature]) {
            const summary = clusterSummaries[feature];
            if (summary.mean && summary.std && summary.count) {
                createEnhancedScatterPlot(feature, summary, 'clusterSummaries');
            } else {
                createEnhancedBarChart(feature, summary, 'clusterSummaries');
            }
        }
    });
    
    // Show message if no features selected
    if (featuresToShow.length === 0) {
        clusterSummariesContainer.innerHTML = `
            <div class="text-center py-8 text-gray-500">
                <svg class="w-12 h-12 mx-auto mb-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                </svg>
                <p>No features selected. Use the filter above to select features to display.</p>
            </div>
        `;
    }
}

// Main analysis function with enhanced UI feedback
document.getElementById('runAnalysis').addEventListener('click', async function() {
    try {
        if (!currentDatasetId) {
            showError('Please upload a dataset first');
            return;
        }

        // Enhanced loading state
        const button = document.getElementById('runAnalysis');
        const buttonText = document.getElementById('buttonText');
        const loadingSpinner = document.getElementById('loadingSpinner');
        const missingStrategy = document.getElementById('missingStrategy').value;
        
        button.disabled = true;
        button.classList.add('opacity-75', 'cursor-not-allowed');
        buttonText.textContent = 'Processing Analysis...';
        loadingSpinner.classList.remove('hidden');
        
        // Start progress simulation
        simulateProgress();

        // Get configuration
        const treatmentCol = document.getElementById('treatmentCol').value;
        const outcomeCol = document.getElementById('outcomeCol').value;

        const requestData = {
            dataset_id: currentDatasetId,
            pre_features: selectedPreFeatures,
            treatment_col: treatmentCol,
            outcome_col: outcomeCol || null,
            missing_strategy: missingStrategy,
            clustering_features: selectedClusteringFeatures.length > 0 ? selectedClusteringFeatures : null
        };
        
        const response = await fetch('http://127.0.0.1:5000/api/psm-analysis', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });
        
        const responseData = await response.json();
        
        if (!response.ok) {
            // Handle specific error messages
            let errorMessage = responseData.error || 'Analysis failed';
            
            if (errorMessage.includes('multiple elements') || errorMessage.includes('too small')) {
                errorMessage = 'Your dataset is too small for analysis. Please ensure your CSV has:\n- At least 10-20 rows of data\n- Multiple columns with different values\n- A balanced treatment/control group';
            } else if (errorMessage.includes('No matches found')) {
                errorMessage = 'No matching pairs found. This usually happens when:\n- Treatment and control groups are too different\n- The caliper (matching distance) is too small\n- There are not enough similar observations between groups';
            } else if (errorMessage.includes('exactly 2 unique values')) {
                errorMessage = 'Treatment column must have exactly 2 unique values (e.g., 0/1, Yes/No, True/False). Please select a binary column.';
            } else if (errorMessage.includes('at least 5 observations')) {
                errorMessage = 'Both treatment and control groups need at least 5 observations each. Your dataset might be too small or imbalanced.';
            }
            
            throw new Error(errorMessage);
        }
        
        const data = responseData;
        
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
        displayResults(data, outcomeCol);

    } catch (error) {
        console.error('Error:', error);
        
        // Enhanced error display
        let userMessage = error.message;
        if (userMessage.includes('\n')) {
            // Format multi-line errors
            userMessage = userMessage.split('\n').map((line, index) => 
                index === 0 ? line : `• ${line}`
            ).join('\n');
        }
        
        showError(userMessage);
        
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

// Add this function to display EDA visualizations
function displayEDAVisualizations(edaVisualizations) {
    const edaSection = document.getElementById('edaSection');
    
    if (edaVisualizations && Object.keys(edaVisualizations).length > 0) {
        edaSection.classList.remove('hidden');
        
        // Treatment Distribution
        if (edaVisualizations.treatment_distribution) {
            document.getElementById('treatmentDistributionImg').src = `data:image/png;base64,${edaVisualizations.treatment_distribution}`;
        }
        
        // Numeric Features Comparison
        if (edaVisualizations.numeric_features_comparison) {
            document.getElementById('numericFeaturesSection').classList.remove('hidden');
            document.getElementById('numericFeaturesImg').src = `data:image/png;base64,${edaVisualizations.numeric_features_comparison}`;
        }
        
        // Categorical Features Comparison
        if (edaVisualizations.categorical_features_comparison) {
            document.getElementById('categoricalFeaturesSection').classList.remove('hidden');
            document.getElementById('categoricalFeaturesImg').src = `data:image/png;base64,${edaVisualizations.categorical_features_comparison}`;
        }
        
        // Correlation Heatmap
        if (edaVisualizations.correlation_heatmap) {
            document.getElementById('correlationSection').classList.remove('hidden');
            document.getElementById('correlationHeatmapImg').src = `data:image/png;base64,${edaVisualizations.correlation_heatmap}`;
        }
        
        // Outcome Comparison
        if (edaVisualizations.outcome_comparison) {
            document.getElementById('outcomeEdaSection').classList.remove('hidden');
            document.getElementById('outcomeComparisonImg').src = `data:image/png;base64,${edaVisualizations.outcome_comparison}`;
        }
        
        // Feature Importance
        if (edaVisualizations.feature_importance) {
            document.getElementById('featureImportanceSection').classList.remove('hidden');
            document.getElementById('featureImportanceImg').src = `data:image/png;base64,${edaVisualizations.feature_importance}`;
        }
        
        // Missing Values
        if (edaVisualizations.missing_values) {
            document.getElementById('missingValuesSection').classList.remove('hidden');
            document.getElementById('missingValuesImg').src = `data:image/png;base64,${edaVisualizations.missing_values}`;
        }
    } else {
        edaSection.classList.add('hidden');
    }
}

// Display analysis results
function displayResults(data, outcomeCol) {
    // Reset results container
    document.getElementById('results').classList.remove('hidden');
    
    // Scroll to results
    document.getElementById('results').scrollIntoView({ behavior: 'smooth' });

    // Display EDA visualizations first
    if (data.visualizations && data.visualizations.eda_visualizations) {
        displayEDAVisualizations(data.visualizations.eda_visualizations);
    }

    // Basic statistics
    document.getElementById('controlCount').textContent = data.treatment_counts.control.toLocaleString();
    document.getElementById('treatmentCount').textContent = data.treatment_counts.treatment.toLocaleString();
    
    // Update outcome comparison title
    const outcomeTitle = document.getElementById('outcomeComparisonTitle');
    outcomeTitle.textContent = `${data.outcome_comparison.outcome_column} Comparison Analysis`;
    
    // Outcome comparison
    document.getElementById('controlBefore').textContent = `${data.outcome_comparison.before.control_mean.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
    document.getElementById('treatmentBefore').textContent = `${data.outcome_comparison.before.treatment_mean.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
    document.getElementById('pBefore').textContent = data.outcome_comparison.before.p_value.toFixed(4);
    
    document.getElementById('controlAfter').textContent = `${data.outcome_comparison.after.control_mean.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
    document.getElementById('treatmentAfter').textContent = `${data.outcome_comparison.after.treatment_mean.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
    document.getElementById('pAfter').textContent = data.outcome_comparison.after.p_value.toFixed(4);
    
    // Matching results
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
    } else {
        evaluationHTML += '<div class="bg-green-50 border border-green-200 rounded-lg p-4">';
        evaluationHTML += '<p class="text-green-700 font-medium">✓ Propensity Score Matching was successful!</p>';
        evaluationHTML += '<p class="text-green-600 text-sm mt-1">All quality checks passed. You can proceed with confidence.</p>';
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
                <span class="font-semibold ${status}">${typeof value === 'number' ? value.toFixed(4) : value}</span>
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
                <span class="font-semibold ${status}">${typeof value === 'number' ? value.toFixed(4) : value}</span>
            </div>
        `;
    }
    pValueHTML += '</div>';
    document.getElementById('pValueSummary').innerHTML = pValueHTML;
    
    // Enhanced overlap and sample size status
    const overlap = psmEval.overlap_status;
    let overlapHTML = `
        <div class="space-y-2">
            <div class="flex justify-between items-center py-2 border-b border-gray-100">
                <span class="font-medium text-gray-700">Max Quantile Difference:</span>
                <span class="font-semibold ${overlap.max_quantile_diff <= 0.2 ? 'text-green-600' : 'text-red-600'}">${overlap.max_quantile_diff.toFixed(4)}</span>
            </div>
            <div class="text-xs text-gray-500 mt-1">
                ${overlap.max_quantile_diff <= 0.2 ? '✓ Good overlap between groups' : '⚠ Poor overlap between groups'}
            </div>
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
                <span class="font-semibold ${sampleSize.retention_rate >= 80 ? 'text-green-600' : 'text-red-600'}">${sampleSize.retention_rate.toFixed(1)}%</span>
            </div>
        </div>
    `;
    document.getElementById('sampleSizeStatus').innerHTML = sampleSizeHTML;
    
    // Load visualizations
    document.getElementById('psBeforeImg').src = `data:image/png;base64,${data.visualizations.ps_before}`;
    document.getElementById('psAfterImg').src = `data:image/png;base64,${data.visualizations.ps_after}`;
    document.getElementById('effectSizesImg').src = `data:image/png;base64,${data.visualizations.effect_sizes}`;
    
    // Enhanced clustering results
    const clusteringSection = document.getElementById('clusteringResults');
    if (psmEval.passed && data.clustering_results) {
        clusteringSection.classList.remove('hidden');
        
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
        `;
        
        if (dimReduction.metrics && dimReduction.metrics.length > 0) {
            dimReductionHTML += `
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
                        <td class="py-3 px-4 text-gray-700">${typeof metric.Silhouette === 'number' ? metric.Silhouette.toFixed(4) : metric.Silhouette}</td>
                        <td class="py-3 px-4 text-gray-700">${typeof metric['Calinski-Harabasz'] === 'number' ? metric['Calinski-Harabasz'].toFixed(4) : metric['Calinski-Harabasz']}</td>
                        <td class="py-3 px-4 text-gray-700">${typeof metric['Davies-Bouldin'] === 'number' ? metric['Davies-Bouldin'].toFixed(4) : metric['Davies-Bouldin']}</td>
                    </tr>
                `;
            });
            
            dimReductionHTML += `</tbody></table></div>`;
        } else {
            dimReductionHTML += `<p class="text-gray-500 text-center py-4">No dimensionality reduction metrics available</p>`;
        }
        
        document.getElementById('dimReductionResults').innerHTML = dimReductionHTML;
        
        // Load feature weights visualization
        if (dimReduction.visualization) {
            document.getElementById('featureWeightsImg').src = `data:image/png;base64,${dimReduction.visualization}`;
        }
        
        // Enhanced ensemble clustering results
        const ensemble = data.clustering_results.ensemble_clustering;
        let ensembleHTML = `
            <div class="mb-4 p-4 bg-gradient-to-r from-green-50 to-blue-50 rounded-lg">
                <div class="flex items-center">
                    <div class="w-3 h-3 bg-green-500 rounded-full mr-2"></div>
                    <span class="font-semibold text-gray-900">Best Model: ${ensemble.best_model}</span>
                </div>
            </div>
        `;
        
        if (ensemble.metrics && ensemble.metrics.length > 0) {
            ensembleHTML += `
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
                        <td class="py-3 px-4 text-gray-700">${typeof metric.Silhouette === 'number' ? metric.Silhouette.toFixed(4) : metric.Silhouette}</td>
                        <td class="py-3 px-4 text-gray-700">${typeof metric['Calinski-Harabasz'] === 'number' ? metric['Calinski-Harabasz'].toFixed(4) : metric['Calinski-Harabasz']}</td>
                        <td class="py-3 px-4 text-gray-700">${typeof metric['Davies-Bouldin'] === 'number' ? metric['Davies-Bouldin'].toFixed(4) : metric['Davies-Bouldin']}</td>
                    </tr>
                `;
            });
            
            ensembleHTML += `</tbody></table></div>`;
        } else {
            ensembleHTML += `<p class="text-gray-500 text-center py-4">No ensemble clustering metrics available</p>`;
        }
        
        document.getElementById('ensembleClusteringResults').innerHTML = ensembleHTML;
        
        // Load cluster visualizations
        if (ensemble.cluster_visualization) {
            document.getElementById('clustersImg').src = `data:image/png;base64,${ensemble.cluster_visualization}`;
        }
        if (ensemble.cluster_profiles) {
            document.getElementById('clusterProfilesImg').src = `data:image/png;base64,${ensemble.cluster_profiles}`;
        }
        
        // Setup feature filter for cluster summaries
        if (ensemble.cluster_summaries) {
            setupClusteringFeatureFilter(ensemble.cluster_summaries);
        }
        
    } else {
        clusteringSection.classList.add('hidden');
    }
    
    // Show clustering warning if present
    if (data.clustering_warning) {
        const warningHTML = `
            <div class="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mt-4">
                <div class="flex items-center">
                    <svg class="w-5 h-5 text-yellow-600 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.35 16.5c-.77.833.192 2.5 1.732 2.5z"></path>
                    </svg>
                    <span class="text-yellow-700">Clustering skipped: ${data.clustering_warning}</span>
                </div>
            </div>
        `;
        clusteringSection.innerHTML = warningHTML;
        clusteringSection.classList.remove('hidden');
    }
}

// Add drag and drop functionality
document.addEventListener('DOMContentLoaded', function() {
    const dropArea = document.querySelector('label[for="csvFile"]');
    
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, unhighlight, false);
    });
    
    function highlight() {
        dropArea.classList.add('border-blue-500', 'bg-blue-50');
    }
    
    function unhighlight() {
        dropArea.classList.remove('border-blue-500', 'bg-blue-50');
    }
    
    dropArea.addEventListener('drop', handleDrop, false);
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        document.getElementById('csvFile').files = files;
        
        // Trigger change event
        const event = new Event('change');
        document.getElementById('csvFile').dispatchEvent(event);
    }
});

// Add keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl+A to select all features (when config panel is visible)
    if (e.ctrlKey && e.key === 'a' && document.getElementById('configPanel').classList.contains('hidden') === false) {
        e.preventDefault();
        document.getElementById('selectAllBtn')?.click();
    }
    
    // Ctrl+D to deselect all features
    if (e.ctrlKey && e.key === 'd' && document.getElementById('configPanel').classList.contains('hidden') === false) {
        e.preventDefault();
        document.getElementById('deselectAllBtn')?.click();
    }
});

// Add tooltip for keyboard shortcuts
function addKeyboardShortcutTooltips() {
    const selectAllBtn = document.getElementById('selectAllBtn');
    const deselectAllBtn = document.getElementById('deselectAllBtn');
    
    if (selectAllBtn) {
        selectAllBtn.title = 'Ctrl+A to select all';
    }
    if (deselectAllBtn) {
        deselectAllBtn.title = 'Ctrl+D to deselect all';
    }
}

// Initialize tooltips when DOM is loaded
document.addEventListener('DOMContentLoaded', addKeyboardShortcutTooltips);