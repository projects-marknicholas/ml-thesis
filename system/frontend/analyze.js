// Global variables
let currentDatasetId = null;
let selectedPreFeatures = [];
let selectedClusteringFeatures = [];
let selectedDemographicsFeatures = [];
let allClusterFeatures = [];

// Create feature visualization for cluster analysis instead of table
function createFeatureVisualization(feature, visualizationData, container) {
    const featureContainer = document.createElement('div');
    featureContainer.className = 'mb-6 bg-white rounded-xl p-6 border border-gray-200 card-hover';
    
    let visualizationHTML = `
        <div class="flex items-center justify-between mb-4">
            <h4 class="font-semibold text-gray-900 flex items-center">
                <div class="w-3 h-3 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full mr-2"></div>
                ${feature} - Cluster Analysis
            </h4>
            <div class="text-sm text-gray-500">Visual Distribution</div>
        </div>
        <div class="flex justify-center items-center p-4 bg-gray-50 rounded-lg">
    `;
    
    if (visualizationData) {
        visualizationHTML += `
            <img src="data:image/png;base64,${visualizationData}" 
                 class="w-full h-auto rounded-lg shadow-lg max-w-2xl"
                 alt="${feature} cluster visualization"
                 onerror="this.src=''; this.alt='Visualization failed to load'">
        `;
    } else {
        visualizationHTML += `
            <div class="text-center">
                <div class="loading-spinner mx-auto mb-2"></div>
                <p class="text-gray-600">Visualization not available</p>
            </div>
        `;
    }
    
    visualizationHTML += `</div>`;
    featureContainer.innerHTML = visualizationHTML;
    container.appendChild(featureContainer);
    
    return featureContainer;
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
    selectedDemographicsFeatures = [];
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
    const selectAllDemographicsBtn = document.getElementById('selectAllDemographicsBtn');
    const deselectAllDemographicsBtn = document.getElementById('deselectAllDemographicsBtn');
    
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

    if (selectAllDemographicsBtn && deselectAllDemographicsBtn) {
        selectAllDemographicsBtn.addEventListener('click', function() {
            selectedDemographicsFeatures = [...allClusterFeatures]; // Use all available columns
            document.querySelectorAll('.demographics-checkbox').forEach(checkbox => {
                checkbox.checked = true;
            });
            
            // Show success feedback
            showSuccess('All demographics features selected!');
        });
        
        deselectAllDemographicsBtn.addEventListener('click', function() {
            selectedDemographicsFeatures = [];
            document.querySelectorAll('.demographics-checkbox').forEach(checkbox => {
                checkbox.checked = false;
            });
            
            // Show feedback
            showInfo('All demographics features deselected');
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

    // Populate demographics features checkboxes
    const demographicsContainer = document.getElementById('demographicsContainer');
    demographicsContainer.innerHTML = '';
    datasetInfo.columns.forEach(col => {
        const div = document.createElement('div');
        div.className = 'flex items-center space-x-2';
        div.innerHTML = `
            <input type="checkbox" id="demographics-${col}" value="${col}" class="demographics-checkbox rounded border-gray-300 text-indigo-600 focus:ring-indigo-500">
            <label for="demographics-${col}" class="text-sm text-gray-700">${col}</label>
        `;
        demographicsContainer.appendChild(div);
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

    // Add event listeners to demographics checkboxes
    document.querySelectorAll('.demographics-checkbox').forEach(checkbox => {
        checkbox.addEventListener('change', function() {
            if (this.checked) {
                selectedDemographicsFeatures.push(this.value);
            } else {
                selectedDemographicsFeatures = selectedDemographicsFeatures.filter(f => f !== this.value);
            }
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
function setupClusteringFeatureFilter(clusterVisualizations) {
    const featureFilterButtons = document.getElementById('featureFilterButtons');
    const featureSearch = document.getElementById('featureSearch');
    const selectAllFeatures = document.getElementById('selectAllFeatures');
    const clearFeatures = document.getElementById('clearFeatures');
    
    // Clear existing buttons
    featureFilterButtons.innerHTML = '';
    
    // Get all available features from cluster visualizations
    const availableFeatures = Object.keys(clusterVisualizations);
    
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
            
            // Filter cluster visualizations display
            filterClusterVisualizations(clusterVisualizations);
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
        filterClusterVisualizations(clusterVisualizations);
    });
    
    // Setup clear functionality
    clearFeatures.addEventListener('click', function() {
        document.querySelectorAll('.feature-filter-btn').forEach(btn => {
            btn.classList.remove('bg-blue-500', 'text-white');
            btn.classList.add('bg-gray-200', 'text-gray-700');
        });
        filterClusterVisualizations(clusterVisualizations);
    });
    
    // Initially select all features
    selectAllFeatures.click();
}

// Filter cluster visualizations based on selected features
function filterClusterVisualizations(clusterVisualizations) {
    const clusterVisualizationsContainer = document.getElementById('clusterVisualizations');
    if (!clusterVisualizationsContainer) {
        console.error('clusterVisualizations container not found');
        return;
    }
    
    clusterVisualizationsContainer.innerHTML = '';
    
    // Get selected features
    const selectedFeatures = Array.from(document.querySelectorAll('.feature-filter-btn.bg-blue-500'))
        .map(btn => btn.dataset.feature);
    
    // If no features selected, show all
    const featuresToShow = selectedFeatures.length > 0 ? selectedFeatures : Object.keys(clusterVisualizations);
    
    // Create visualizations for selected features
    featuresToShow.forEach(feature => {
        if (clusterVisualizations[feature]) {
            createFeatureVisualization(feature, clusterVisualizations[feature], clusterVisualizationsContainer);
        }
    });
    
    // Show message if no features selected or no visualizations available
    if (featuresToShow.length === 0 || Object.keys(clusterVisualizations).length === 0) {
        clusterVisualizationsContainer.innerHTML = `
            <div class="text-center py-8 text-gray-500">
                <svg class="w-12 h-12 mx-auto mb-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                </svg>
                <p>No features selected or visualizations available. Use the filter above to select features to display.</p>
            </div>
        `;
    }
}

// Function to display additional clustering visualizations
function displayAdditionalClusteringVisualizations(visualizations) {
    const vizContainer = document.getElementById('clusteringVizContainer');
    if (!vizContainer) return;
    
    vizContainer.innerHTML = ''; // Clear existing content
    
    // Cluster Distribution
    if (visualizations.cluster_distribution) {
        vizContainer.innerHTML += `
            <div class="bg-white rounded-lg p-4 border border-gray-200">
                <h4 class="font-semibold text-gray-800 mb-3">Cluster Distribution</h4>
                <img src="data:image/png;base64,${visualizations.cluster_distribution}" 
                     class="w-full h-auto rounded-lg shadow-sm"
                     onerror="this.style.display='none'">
            </div>
        `;
    }
    
    // Feature Distributions
    if (visualizations.feature_distributions) {
        vizContainer.innerHTML += `
            <div class="bg-white rounded-lg p-4 border border-gray-200">
                <h4 class="font-semibold text-gray-800 mb-3">Feature Distributions by Cluster</h4>
                <img src="data:image/png;base64,${visualizations.feature_distributions}" 
                     class="w-full h-auto rounded-lg shadow-sm"
                     onerror="this.style.display='none'">
            </div>
        `;
    }
}

// Enhanced function to display clustering results
function displayClusteringResults(data) {
    const clusteringSection = document.getElementById('clusteringResults');
    
    if (data.clustering_results) {
        clusteringSection.classList.remove('hidden');
        
        const clustering = data.clustering_results;
        
        // Data cleaning summary
        const cleaning = clustering.data_cleaning;
        document.getElementById('dataCleaningSummary').innerHTML = `
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
        
        // Dimensionality reduction results
        const dimReduction = clustering.dimensionality_reduction;
        if (dimReduction.visualization) {
            const featureWeightsImg = document.getElementById('featureWeightsImg');
            featureWeightsImg.src = `data:image/png;base64,${dimReduction.visualization}`;
            featureWeightsImg.onerror = function() {
                this.alt = 'Feature weights visualization failed to load';
            };
        }
        
        // Ensemble clustering results - FIXED CLUSTER VISUALIZATION
        const ensemble = clustering.ensemble_clustering;
        if (ensemble.cluster_visualization) {
            const clusterImg = document.getElementById('clustersImg');
            clusterImg.src = `data:image/png;base64,${ensemble.cluster_visualization}`;
            clusterImg.onload = function() {
                console.log('Cluster visualization loaded successfully');
            };
            clusterImg.onerror = function() {
                console.error('Failed to load cluster visualization');
                this.alt = 'Cluster visualization failed to load. This may be due to data limitations.';
                // Create a fallback message
                const container = this.parentElement;
                container.innerHTML += `
                    <div class="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mt-2">
                        <p class="text-yellow-700 text-sm">
                            Note: Cluster visualization may not display properly with very small datasets or limited features.
                        </p>
                    </div>
                `;
            };
        }
        
        // Display additional clustering visualizations if available
        if (ensemble.clustering_visualizations) {
            displayAdditionalClusteringVisualizations(ensemble.clustering_visualizations);
        }
        
        // Display feature visualizations INSTEAD OF TABLES
        if (ensemble.feature_visualizations && Object.keys(ensemble.feature_visualizations).length > 0) {
            // Create interactive cluster analysis section
            const interactiveSection = document.createElement('div');
            interactiveSection.className = 'bg-white rounded-xl p-6 border border-gray-200';
            interactiveSection.innerHTML = `
                <h3 class="font-semibold text-gray-900 mb-6 flex items-center">
                    <svg class="w-5 h-5 text-purple-600 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path>
                    </svg>
                    Interactive Cluster Analysis
                </h3>
                
                <!-- Feature Filter -->
                <div class="mb-6">
                    <div class="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-4">
                        <div>
                            <h4 class="font-medium text-gray-900 mb-2">Filter Features</h4>
                            <div class="flex flex-wrap gap-2" id="featureFilterButtons"></div>
                        </div>
                        <div class="flex gap-2">
                            <input type="text" id="featureSearch" placeholder="Search features..." 
                                   class="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500">
                            <button id="selectAllFeatures" class="px-3 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors">
                                Select All
                            </button>
                            <button id="clearFeatures" class="px-3 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors">
                                Clear
                            </button>
                        </div>
                    </div>
                </div>
                
                <!-- Visualizations Container -->
                <div id="clusterVisualizations" class="space-y-6"></div>
            `;
            
            // Find or create the cluster summaries section
            let summariesSection = document.getElementById('clusterSummariesSection');
            if (!summariesSection) {
                summariesSection = document.createElement('div');
                summariesSection.id = 'clusterSummariesSection';
                summariesSection.className = 'bg-white rounded-xl p-6 border border-gray-200';
                
                // Find where to insert the section
                const clusteringResults = document.getElementById('clusteringResults');
                const spaceY8 = clusteringResults.querySelector('.space-y-8');
                if (spaceY8) {
                    spaceY8.appendChild(summariesSection);
                } else {
                    clusteringResults.appendChild(summariesSection);
                }
            }
            
            // Replace the content
            summariesSection.innerHTML = interactiveSection.innerHTML;
            
            // Setup the feature filter with visualizations
            setupClusteringFeatureFilter(ensemble.feature_visualizations);
        }
        
    } else {
        clusteringSection.classList.add('hidden');
    }
}

// Function to display demographics analysis
function displayDemographicsResults(data) {
    const demographicsSection = document.getElementById('demographicsSection');
    const demographicsResults = document.getElementById('demographicsResults');
    const demographicsComparisonImg = document.getElementById('demographicsComparisonImg');
    
    if (data.demographics_analysis && Object.keys(data.demographics_analysis).length > 0) {
        demographicsSection.classList.remove('hidden');
        
        // Display demographics results
        let demographicsHTML = '<div class="grid grid-cols-1 md:grid-cols-2 gap-6">';
        
        Object.entries(data.demographics_analysis).forEach(([feature, analysis]) => {
            if (analysis.type === 'numeric') {
                demographicsHTML += `
                    <div class="bg-white rounded-xl p-6 border border-gray-200">
                        <h4 class="font-semibold text-gray-900 mb-4">${feature}</h4>
                        <div class="space-y-2">
                            <div class="flex justify-between">
                                <span class="text-gray-600">Control Mean:</span>
                                <span class="font-semibold">${analysis.control_mean.toFixed(2)}</span>
                            </div>
                            <div class="flex justify-between">
                                <span class="text-gray-600">Treatment Mean:</span>
                                <span class="font-semibold">${analysis.treatment_mean.toFixed(2)}</span>
                            </div>
                            <div class="flex justify-between">
                                <span class="text-gray-600">P-value:</span>
                                <span class="font-semibold ${analysis.p_value > 0.05 ? 'text-green-600' : 'text-red-600'}">
                                    ${analysis.p_value.toFixed(4)}
                                </span>
                            </div>
                        </div>
                    </div>
                `;
            } else {
                // For categorical features
                demographicsHTML += `
                    <div class="bg-white rounded-xl p-6 border border-gray-200">
                        <h4 class="font-semibold text-gray-900 mb-4">${feature}</h4>
                        <div class="text-sm text-gray-600">
                            Categorical distribution available in visualization
                        </div>
                    </div>
                `;
            }
        });
        
        demographicsHTML += '</div>';
        demographicsResults.innerHTML = demographicsHTML;
        
        // Display visualization if available
        if (data.visualizations && data.visualizations.demographics_comparison) {
            demographicsComparisonImg.src = `data:image/png;base64,${data.visualizations.demographics_comparison}`;
        } else {
            demographicsComparisonImg.style.display = 'none';
        }
        
    } else {
        demographicsSection.classList.add('hidden');
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
            clustering_features: selectedClusteringFeatures.length > 0 ? selectedClusteringFeatures : null,
            demographics_features: selectedDemographicsFeatures.length > 0 ? selectedDemographicsFeatures : null
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
    
    // Display demographics results
    displayDemographicsResults(data);
    
    // Enhanced clustering results
    const clusteringSection = document.getElementById('clusteringResults');
    if (psmEval.passed && data.clustering_results) {
        displayClusteringResults(data);
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