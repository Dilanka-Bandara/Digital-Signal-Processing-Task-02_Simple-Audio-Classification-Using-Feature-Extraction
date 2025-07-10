clear; clc; close all;

%% --- 1. Path Setup ---
train_amb = fullfile('filter','train','ambulance');
train_fire = fullfile('filter','train','firetruck');
test_amb = fullfile('filter','test','ambulance');
test_fire = fullfile('filter','test','firetruck');
fs_target = 16000;

%% --- 2. Analyze Frequency Content (Training Data) ---
fprintf('Analyzing frequency content of training data...\n');
amb_train_files = dir(fullfile(train_amb,'*.wav'));
fire_train_files = dir(fullfile(train_fire,'*.wav'));
NFFT = 2048;
amb_spectra = zeros(NFFT/2+1, numel(amb_train_files));
fire_spectra = zeros(NFFT/2+1, numel(fire_train_files));
for i = 1:numel(amb_train_files)
    [audio, fs] = audioread(fullfile(train_amb, amb_train_files(i).name));
    audio = preprocess_audio(audio, fs, fs_target);
    S = abs(fft(audio, NFFT));
    amb_spectra(:,i) = S(1:NFFT/2+1);
end
for i = 1:numel(fire_train_files)
    [audio, fs] = audioread(fullfile(train_fire, fire_train_files(i).name));
    audio = preprocess_audio(audio, fs, fs_target);
    S = abs(fft(audio, NFFT));
    fire_spectra(:,i) = S(1:NFFT/2+1);
end
mean_amb = mean(amb_spectra,2);
mean_fire = mean(fire_spectra,2);
freqs = linspace(0, fs_target/2, NFFT/2+1);

% Plot average spectra for both classes
figure;
plot(freqs, mean_amb, 'b', 'LineWidth',1.5); hold on;
plot(freqs, mean_fire, 'r', 'LineWidth',1.5);
xlabel('Frequency (Hz)'); ylabel('Magnitude');
legend('Ambulance','Firetruck'); title('Average Spectrum (Training)');
grid on;

%% --- 3. Design Multiple Bandpass Filters (Filter Bank) ---
% Choose 3 frequency bands based on spectrum plot
bands = [500 1100; 1300 2000; 2200 3200]; % [low high] for each band
numBands = size(bands,1);
fprintf('Designing %d bandpass filters...\n', numBands);
bpFilters = cell(numBands,1);
for b = 1:numBands
    bpFilters{b} = designfilt('bandpassiir','FilterOrder',8, ...
        'HalfPowerFrequency1',bands(b,1),'HalfPowerFrequency2',bands(b,2), ...
        'SampleRate',fs_target);
end

% Plot filter responses
figure;
for b = 1:numBands
    [h, f] = freqz(bpFilters{b}, 1024, fs_target);
    plot(f, 20*log10(abs(h)), 'DisplayName',sprintf('Band %d: %d-%d Hz',b,bands(b,1),bands(b,2))); hold on;
end
xlabel('Frequency (Hz)'); ylabel('Magnitude (dB)');
title('Bandpass Filter Responses'); legend; grid on;

%% --- 4. Compute Filtered Energies and Feature Vectors ---
fprintf('Extracting filter-bank energy features for training data...\n');
train_features = [];
train_labels = [];
for i = 1:numel(amb_train_files)
    [audio, fs] = audioread(fullfile(train_amb, amb_train_files(i).name));
    audio = preprocess_audio(audio, fs, fs_target);
    energies = zeros(1,numBands);
    for b = 1:numBands
        energies(b) = band_energy(audio, bpFilters{b});
    end
    % Use ratios between bands as features
    feat = [energies(1)/energies(2), energies(2)/energies(3), energies(1)/energies(3)];
    train_features = [train_features; feat];
    train_labels = [train_labels; 1];
end
for i = 1:numel(fire_train_files)
    [audio, fs] = audioread(fullfile(train_fire, fire_train_files(i).name));
    audio = preprocess_audio(audio, fs, fs_target);
    energies = zeros(1,numBands);
    for b = 1:numBands
        energies(b) = band_energy(audio, bpFilters{b});
    end
    feat = [energies(1)/energies(2), energies(2)/energies(3), energies(1)/energies(3)];
    train_features = [train_features; feat];
    train_labels = [train_labels; 2];
end

% Plot feature distributions
figure;
gscatter(train_features(:,1),train_features(:,2),train_labels, 'br','ox');
xlabel('Energy Ratio Band1/Band2'); ylabel('Energy Ratio Band2/Band3');
title('Training Feature Scatter Plot'); legend('Ambulance','Firetruck'); grid on;

%% --- 5. Train Linear Classifier (LDA) ---
% Still transparent, but more robust than a single threshold
lda = fitcdiscr(train_features, train_labels);

% --- LDA Decision Region Plot (after LDA training) ---
% Only uses first two features for 2D visualization

% 1. Create a grid over the feature space
x1 = linspace(min(train_features(:,1))-0.1, max(train_features(:,1))+0.1, 200);
x2 = linspace(min(train_features(:,2))-0.1, max(train_features(:,2))+0.1, 200);
[X1, X2] = meshgrid(x1, x2);
% Use mean of third feature for grid (since we plot in 2D)
grid_points = [X1(:), X2(:), repmat(mean(train_features(:,3)), numel(X1), 1)];

% 2. Predict class for each grid point
[grid_labels, ~] = predict(lda, grid_points);
Z = reshape(grid_labels, size(X1));

% 3. Plot decision regions
figure;
contourf(X1, X2, Z, [0.5 1.5 2.5], 'LineColor','none');
colormap([0.8 0.9 1; 1 0.8 0.8]); hold on;

% 4. Overlay training data
h = gscatter(train_features(:,1), train_features(:,2), train_labels, 'br', 'ox', 8, 'off');

xlabel('Energy Ratio Band1/Band2');
ylabel('Energy Ratio Band2/Band3');
title('LDA Decision Regions and Training Data');
legend(h, {'Ambulance','Firetruck'}, 'Location','best');
grid on;


%% --- 6. Test and Evaluate ---
fprintf('Classifying test files...\n');
amb_test_files = dir(fullfile(test_amb,'*.wav'));
fire_test_files = dir(fullfile(test_fire,'*.wav'));
test_files = [arrayfun(@(f) fullfile(test_amb, f.name), amb_test_files, 'UniformOutput', false); ...
              arrayfun(@(f) fullfile(test_fire, f.name), fire_test_files, 'UniformOutput', false)];
test_labels = [ones(1,numel(amb_test_files)), 2*ones(1,numel(fire_test_files))];
test_features = [];
for i = 1:numel(test_files)
    [audio, fs] = audioread(test_files{i});
    audio = preprocess_audio(audio, fs, fs_target);
    energies = zeros(1,numBands);
    for b = 1:numBands
        energies(b) = band_energy(audio, bpFilters{b});
    end
    feat = [energies(1)/energies(2), energies(2)/energies(3), energies(1)/energies(3)];
    test_features = [test_features; feat];
end
[pred, score] = predict(lda, test_features);

% Print results
class_names = {'ambulance','firetruck'};
correct = sum(pred(:)' == test_labels);
fprintf('\n%-25s %-12s %-12s %-10s\n','File','True Class','Prediction','Score');
fprintf('%s\n',repmat('-',[1 60]));
for i = 1:numel(test_files)
    fprintf('%-25s %-12s %-12s %-10.2f\n', ...
        test_files{i}(max(1,end-20):end), ...
        class_names{test_labels(i)}, ...
        class_names{pred(i)}, ...
        max(score(i,:)));
end
acc = 100*correct/numel(test_files);
fprintf('\nFinal Accuracy: %.2f%% (%d/%d)\n', acc, correct, numel(test_files));

% (Optional) Plot test features
figure;
gscatter(test_features(:,1),test_features(:,2),pred, 'br','ox');
xlabel('Energy Ratio Band1/Band2'); ylabel('Energy Ratio Band2/Band3');
title('Test Feature Scatter Plot (Predicted Classes)'); legend('Ambulance','Firetruck'); grid on;

%% --- Helper Functions ---
function audio = preprocess_audio(audio, fs, fs_target)
    if size(audio,2)>1, audio = mean(audio,2); end
    if fs ~= fs_target, audio = resample(audio, fs_target, fs); end
    audio = audio / (max(abs(audio)) + eps);
end
function e = band_energy(audio, filt)
    filtered = filtfilt(filt, audio);
    e = sum(filtered.^2);
end
