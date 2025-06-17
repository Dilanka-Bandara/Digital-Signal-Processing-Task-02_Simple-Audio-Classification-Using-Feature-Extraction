%% Audio Classification - Part 2: Filter-Based Emergency Vehicle Classification
clear; clc; close all;

%% Set up paths
train_folder = 'filter/train';
test_folder = 'filter/test';

%% Analyze training data to design filters
fprintf('Analyzing training data...\n');
[ambulance_files, firetruck_files] = load_training_data(train_folder);

% Analyze frequency content
[ambulance_spectra, firetruck_spectra, fs] = analyze_frequency_content(ambulance_files, firetruck_files);

%% Design bandpass filters based on spectral analysis
fprintf('Designing bandpass filters...\n');
[filter1, filter2] = design_classification_filters(ambulance_spectra, firetruck_spectra, fs);

%% Calculate energy ratios for training data to set threshold
fprintf('Calculating training energy ratios...\n');
[ambulance_ratios, firetruck_ratios] = calculate_training_ratios(ambulance_files, firetruck_files, filter1, filter2, fs);

% Set classification threshold
threshold = set_classification_threshold(ambulance_ratios, firetruck_ratios);

%% Test on test data
fprintf('Testing on test data...\n');
test_results = classify_test_data(test_folder, filter1, filter2, threshold, fs);

%% Display results
display_filter_results(test_results, threshold);

%% Save results
save_filter_results(test_results, 'emergency_vehicle_classification_results.txt');
