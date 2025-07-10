clear; clc; close all;

% --- Paths ---
train_amb = fullfile('filter','train','ambulance');
train_fire = fullfile('filter','train','firetruck');
test_amb = fullfile('filter','test','ambulance');
test_fire = fullfile('filter','test','firetruck');

% --- 1. Parameters for MFCC Feature Extraction ---
fs_target = 16000; % Standardize sample rate for consistency
n_mfcc = 13;       % Number of MFCCs to extract, a standard value

fprintf('Using MFCC features with a KNN classifier for high accuracy.\n');

% --- 2. Extract Robust Features from Training Data ---
fprintf('Extracting MFCC features from training data...\n');
amb_train_files = dir(fullfile(train_amb,'*.wav'));
fire_train_files = dir(fullfile(train_fire,'*.wav'));
X_train = [];
y_train = [];

for i = 1:numel(amb_train_files)
    [audio, fs] = audioread(fullfile(train_amb, amb_train_files(i).name));
    % Extract a rich feature vector for each file
    features = extract_mfcc_features(audio, fs, fs_target, n_mfcc);
    X_train = [X_train; features];
    y_train = [y_train; 1]; % 1 for ambulance
end
for i = 1:numel(fire_train_files)
    [audio, fs] = audioread(fullfile(train_fire, fire_train_files(i).name));
    features = extract_mfcc_features(audio, fs, fs_target, n_mfcc);
    X_train = [X_train; features];
    y_train = [y_train; 2]; % 2 for firetruck
end

% --- 3. Train a Simple but Powerful Classifier ---
% A K-Nearest Neighbors (KNN) classifier is perfect for finding the "closest"
% match in the feature space, aligning perfectly with the assignment's goals.
fprintf('Training K-Nearest Neighbors (KNN) classifier...\n');
knn_model = fitcknn(X_train, y_train, 'NumNeighbors', 5, 'Distance', 'euclidean');

% --- 4. Test the Classifier ---
fprintf('Classifying test files...\n');
amb_test_files = dir(fullfile(test_amb,'*.wav'));
fire_test_files = dir(fullfile(test_fire,'*.wav'));

% Use vertical concatenation [;] to avoid dimension errors
test_files = [arrayfun(@(f) fullfile(test_amb, f.name), amb_test_files, 'UniformOutput', false); ...
              arrayfun(@(f) fullfile(test_fire, f.name), fire_test_files, 'UniformOutput', false)];
test_labels = [ones(1,numel(amb_test_files)), 2*ones(1,numel(fire_test_files))];

class_names = {'ambulance', 'firetruck'}; 
correct = 0;
fprintf('\n%-25s %-12s %-12s %-10s\n','File','True Class','Prediction','Confidence');
fprintf('%s\n',repmat('-',[1 60]));

for i = 1:numel(test_files)
    [audio, fs] = audioread(test_files{i});
    
    % Extract the same features from the test file
    test_features = extract_mfcc_features(audio, fs, fs_target, n_mfcc);
    
    % Predict using the trained model
    [pred, score, ~] = predict(knn_model, test_features);
    
    % Confidence is based on the proportion of neighbors belonging to the winning class
    confidence = max(score); 
    
    correct = correct + (pred == test_labels(i));
    
    fprintf('%-25s %-12s %-12s %-10.2f\n', ...
        test_files{i}(max(1,end-20):end), ...
        class_names{test_labels(i)}, ...
        class_names{pred}, ...
        confidence);
end

acc = 100*correct/numel(test_files);
fprintf('\nFinal Accuracy: %.2f%% (%d/%d)\n', acc, correct, numel(test_files));

% --- Local Helper Function (must be at the end of the script) ---
function features = extract_mfcc_features(audio, fs, fs_target, n_mfcc)
    % 1. Preprocess the audio (essential for consistency)
    if size(audio,2)>1, audio = mean(audio,2); end % Convert to Mono
    if fs ~= fs_target, audio = resample(audio, fs_target, fs); end % Resample
    audio = audio / (max(abs(audio)) + eps); % Normalize volume
    
    % 2. Extract MFCCs
    mfccs = mfcc(audio, fs_target, 'NumCoeffs', n_mfcc);
    
    % 3. Create a descriptive feature vector by summarizing the MFCCs over time
    % We use both the mean (average spectral shape) and standard deviation 
    % (how much the shape varies), which is very powerful.
    features = [mean(mfccs, 1), std(mfccs, 0, 1)]; % Total: 26 features
end
