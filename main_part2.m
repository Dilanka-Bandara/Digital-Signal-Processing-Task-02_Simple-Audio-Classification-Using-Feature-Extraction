%% Audio Classification - Part 2: Filter-Based Emergency Vehicle Classification
clear; clc; close all;

train_folder = 'filter/train';
test_folder = 'filter/test';

[ambulance_files, firetruck_files] = load_training_data(train_folder);
[ambulance_spectra, firetruck_spectra, fs] = analyze_frequency_content(ambulance_files, firetruck_files);
[filter1, filter2] = design_classification_filters(ambulance_spectra, firetruck_spectra, fs);
[ambulance_ratios, firetruck_ratios] = calculate_training_ratios(ambulance_files, firetruck_files, filter1, filter2);
threshold = set_classification_threshold(ambulance_ratios, firetruck_ratios);
test_results = classify_test_data(test_folder, filter1, filter2, threshold, fs);
display_filter_results(test_results, threshold);
save_filter_results(test_results, 'emergency_vehicle_classification_results.txt');
