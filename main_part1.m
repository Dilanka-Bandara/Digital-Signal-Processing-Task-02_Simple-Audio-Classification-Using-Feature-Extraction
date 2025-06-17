%% Audio Classification - Part 1
% Main script for classifying unknown audio files using MFCC and FFT features
% with multiple distance metrics

clear; clc; close all;

%% Set up paths
class1_folder = 'class_1';
class2_folder = 'class_2';
unknown_folder = 'unknown';

%% Extract features from training data
fprintf('Extracting features from Class 1...\n');
[class1_features_mfcc, class1_features_fft, class1_files] = extract_features_from_folder(class1_folder);
fprintf('Extracting features from Class 2...\n');
[class2_features_mfcc, class2_features_fft, class2_files] = extract_features_from_folder(class2_folder);

%% Extract features from unknown files
fprintf('Extracting features from Unknown files...\n');
[unknown_features_mfcc, unknown_features_fft, unknown_files] = extract_features_from_folder(unknown_folder);

%% Classification using MFCC features
fprintf('\nClassifying using MFCC features...\n');
results_mfcc = classify_audio_files(unknown_features_mfcc, class1_features_mfcc, class2_features_mfcc, unknown_files, 'MFCC');

%% Classification using FFT features
fprintf('Classifying using FFT features...\n');
results_fft = classify_audio_files(unknown_features_fft, class1_features_fft, class2_features_fft, unknown_files, 'FFT');

%% Display results
display_results(results_mfcc, 'MFCC');
display_results(results_fft, 'FFT');

%% Save results to file
save_results_to_file(results_mfcc, 'classification_results_mfcc.txt');
save_results_to_file(results_fft, 'classification_results_fft.txt');
