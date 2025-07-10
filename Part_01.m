%% Part 1: Audio Classification Using Feature Extraction
clear; clc; close all;

% ========== PATH CONFIGURATION (Relative Paths) ==========
class1_path = 'class_1';
class2_path = 'class_2';
unknown_path = 'unknown';

% Verify paths exist
if ~exist(class1_path, 'dir')
    error('Class 1 folder not found. Make sure "class_1" folder exists in the same directory as this script.');
end
if ~exist(class2_path, 'dir')
    error('Class 2 folder not found. Make sure "class_2" folder exists in the same directory as this script.');
end
if ~exist(unknown_path, 'dir')
    error('Unknown folder not found. Make sure "unknown" folder exists in the same directory as this script.');
end

fprintf('=== PART 1: AUDIO CLASSIFICATION USING FEATURE EXTRACTION ===\n\n');

%% Step 1: Load and Extract Features with Consistent Sizing

% Get file lists
class1_files = dir(fullfile(class1_path, '*.wav'));
class2_files = dir(fullfile(class2_path, '*.wav'));
unknown_files = dir(fullfile(unknown_path, '*.wav'));

fprintf('Found %d files in class_1\n', length(class1_files));
fprintf('Found %d files in class_2\n', length(class2_files));
fprintf('Found %d files in unknown\n', length(unknown_files));

% Fixed parameters for consistent feature extraction
NUM_MFCC_COEFFS = 13;
TARGET_SAMPLE_RATE = 16000;

% Initialize feature storage
class1_features = [];
class2_features = [];

fprintf('\nExtracting MFCC features from Class 1 files...\n');
for i = 1:length(class1_files)
    file_path = fullfile(class1_path, class1_files(i).name);
    
    try
        [audio, fs] = audioread(file_path);
        
        % Preprocessing for consistency
        if fs ~= TARGET_SAMPLE_RATE
            audio = resample(audio, TARGET_SAMPLE_RATE, fs);
            fs = TARGET_SAMPLE_RATE;
        end
        
        % Convert to mono if stereo
        if size(audio, 2) > 1
            audio = mean(audio, 2);
        end
        
        % Extract MFCC features
        mfcc_features = mfcc(audio, fs, 'NumCoeffs', NUM_MFCC_COEFFS);
        
        % Create fixed-size feature vector using statistical measures
        feature_vector = [
            mean(mfcc_features, 1), ...     % Mean of each coefficient (13 features)
            std(mfcc_features, 0, 1), ...  % Standard deviation (13 features)
            max(mfcc_features, [], 1), ... % Maximum values (13 features)
            min(mfcc_features, [], 1)      % Minimum values (13 features)
        ];
        
        class1_features = [class1_features; feature_vector];
        fprintf('  Processed: %s (Feature size: %d)\n', class1_files(i).name, length(feature_vector));
        
    catch ME
        fprintf('  Error processing %s: %s\n', class1_files(i).name, ME.message);
        continue;
    end
end

fprintf('\nExtracting MFCC features from Class 2 files...\n');
for i = 1:length(class2_files)
    file_path = fullfile(class2_path, class2_files(i).name);
    
    try
        [audio, fs] = audioread(file_path);
        
        % Preprocessing for consistency
        if fs ~= TARGET_SAMPLE_RATE
            audio = resample(audio, TARGET_SAMPLE_RATE, fs);
            fs = TARGET_SAMPLE_RATE;
        end
        
        % Convert to mono if stereo
        if size(audio, 2) > 1
            audio = mean(audio, 2);
        end
        
        % Extract MFCC features
        mfcc_features = mfcc(audio, fs, 'NumCoeffs', NUM_MFCC_COEFFS);
        
        % Create fixed-size feature vector using statistical measures
        feature_vector = [
            mean(mfcc_features, 1), ...     % Mean of each coefficient
            std(mfcc_features, 0, 1), ...  % Standard deviation
            max(mfcc_features, [], 1), ... % Maximum values
            min(mfcc_features, [], 1)      % Minimum values
        ];
        
        class2_features = [class2_features; feature_vector];
        fprintf('  Processed: %s (Feature size: %d)\n', class2_files(i).name, length(feature_vector));
        
    catch ME
        fprintf('  Error processing %s: %s\n', class2_files(i).name, ME.message);
        continue;
    end
end

% Verify feature consistency
fprintf('\nFeature Matrix Summary:\n');
fprintf('Class 1 features: %d samples x %d features\n', size(class1_features, 1), size(class1_features, 2));
fprintf('Class 2 features: %d samples x %d features\n', size(class2_features, 1), size(class2_features, 2));

% Check if feature sizes match
if size(class1_features, 2) ~= size(class2_features, 2)
    error('Feature sizes do not match! Class1: %d, Class2: %d', ...
          size(class1_features, 2), size(class2_features, 2));
end

%% Step 2: Classification of Unknown Files

fprintf('\nClassifying unknown files using Euclidean Distance...\n');
results = [];
expected_feature_size = size(class1_features, 2);

for i = 1:length(unknown_files)
    file_path = fullfile(unknown_path, unknown_files(i).name);
    
    try
        [audio, fs] = audioread(file_path);
        
        % Apply same preprocessing
        if fs ~= TARGET_SAMPLE_RATE
            audio = resample(audio, TARGET_SAMPLE_RATE, fs);
            fs = TARGET_SAMPLE_RATE;
        end
        
        if size(audio, 2) > 1
            audio = mean(audio, 2);
        end
        
        % Extract MFCC features with same structure
        mfcc_features = mfcc(audio, fs, 'NumCoeffs', NUM_MFCC_COEFFS);
        
        unknown_feature = [
            mean(mfcc_features, 1), ...
            std(mfcc_features, 0, 1), ...
            max(mfcc_features, [], 1), ...
            min(mfcc_features, [], 1)
        ];
        
        % Verify feature size consistency
        if length(unknown_feature) ~= expected_feature_size
            fprintf('  Warning: Feature size mismatch for %s. Expected: %d, Got: %d\n', ...
                    unknown_files(i).name, expected_feature_size, length(unknown_feature));
            continue;
        end
        
        % Calculate distances to all class 1 samples using Euclidean distance
        distances_class1 = [];
        for j = 1:size(class1_features, 1)
            dist = sqrt(sum((unknown_feature - class1_features(j, :)).^2));
            distances_class1 = [distances_class1; dist];
        end
        
        % Calculate distances to all class 2 samples
        distances_class2 = [];
        for j = 1:size(class2_features, 1)
            dist = sqrt(sum((unknown_feature - class2_features(j, :)).^2));
            distances_class2 = [distances_class2; dist];
        end
        
        % Find minimum distances
        min_dist_class1 = min(distances_class1);
        min_dist_class2 = min(distances_class2);
        
        % Classify based on minimum distance
        if min_dist_class1 < min_dist_class2
            predicted_class = 1;
            confidence = min_dist_class2 / (min_dist_class1 + min_dist_class2);
        else
            predicted_class = 2;
            confidence = min_dist_class1 / (min_dist_class1 + min_dist_class2);
        end
        
        % Store results
        results = [results; struct('filename', unknown_files(i).name, ...
                                  'predicted_class', predicted_class, ...
                                  'confidence', confidence, ...
                                  'dist_class1', min_dist_class1, ...
                                  'dist_class2', min_dist_class2)];
        
        fprintf('  %s -> Class %d (Confidence: %.3f)\n', ...
                unknown_files(i).name, predicted_class, confidence);
        
    catch ME
        fprintf('  Error processing %s: %s\n', unknown_files(i).name, ME.message);
        continue;
    end
end

%% Display Final Results
fprintf('\n=== PART 1 CLASSIFICATION RESULTS ===\n');
fprintf('%-25s %-15s %-12s %-12s %-12s\n', 'Filename', 'Predicted Class', 'Confidence', 'Dist Class1', 'Dist Class2');
fprintf('%s\n', repmat('-', 1, 80));
for i = 1:length(results)
    fprintf('%-25s %-15d %-12.3f %-12.3f %-12.3f\n', ...
            results(i).filename, results(i).predicted_class, ...
            results(i).confidence, results(i).dist_class1, results(i).dist_class2);
end

% Save results
save('part1_classification_results.mat', 'results', 'class1_features', 'class2_features');
fprintf('\nResults saved to part1_classification_results.mat\n');
fprintf('Part 1 completed successfully!\n\n');
