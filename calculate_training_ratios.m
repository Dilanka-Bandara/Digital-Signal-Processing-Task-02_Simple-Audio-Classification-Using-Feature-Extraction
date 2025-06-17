function [ambulance_ratios, firetruck_ratios] = calculate_training_ratios(ambulance_files, firetruck_files, filter1, filter2, fs)
% Calculate energy ratios for training data
    
    ambulance_ratios = zeros(length(ambulance_files), 1);
    firetruck_ratios = zeros(length(firetruck_files), 1);
    
    % Calculate ratios for ambulance files
    for i = 1:length(ambulance_files)
        [audioIn, ~] = audioread(ambulance_files(i).fullpath);
        if size(audioIn, 2) > 1
            audioIn = mean(audioIn, 2);
        end
        
        ambulance_ratios(i) = calculate_energy_ratio(audioIn, filter1, filter2);
    end
    
    % Calculate ratios for firetruck files
    for i = 1:length(firetruck_files)
        [audioIn, ~] = audioread(firetruck_files(i).fullpath);
        if size(audioIn, 2) > 1
            audioIn = mean(audioIn, 2);
        end
        
        firetruck_ratios(i) = calculate_energy_ratio(audioIn, filter1, filter2);
    end
end

function ratio = calculate_energy_ratio(audioIn, filter1, filter2)
% Calculate energy ratio between two filtered signals
    
    % Apply filters
    filtered1 = filtfilt(filter1.b, filter1.a, audioIn);
    filtered2 = filtfilt(filter2.b, filter2.a, audioIn);
    
    % Calculate energies
    energy1 = sum(filtered1.^2);
    energy2 = sum(filtered2.^2);
    
    % Calculate ratio (avoid division by zero)
    if energy2 == 0
        ratio = inf;
    else
        ratio = energy1 / energy2;
    end
end

function threshold = set_classification_threshold(ambulance_ratios, firetruck_ratios)
% Set classification threshold based on training data
    
    % Calculate means
    mean_ambulance = mean(ambulance_ratios);
    mean_firetruck = mean(firetruck_ratios);
    
    % Set threshold as midpoint
    threshold = (mean_ambulance + mean_firetruck) / 2;
    
    fprintf('Ambulance mean ratio: %.3f\n', mean_ambulance);
    fprintf('Firetruck mean ratio: %.3f\n', mean_firetruck);
    fprintf('Classification threshold: %.3f\n', threshold);
end

function test_results = classify_test_data(test_folder, filter1, filter2, threshold, fs)
% Classify test data using designed filters and threshold
    
    test_files = dir(fullfile(test_folder, '*.wav'));
    num_test_files = length(test_files);
    
    test_results = struct();
    test_results.filenames = cell(num_test_files, 1);
    test_results.energy_ratios = zeros(num_test_files, 1);
    test_results.predictions = cell(num_test_files, 1);
    test_results.confidences = zeros(num_test_files, 1);
    
    for i = 1:num_test_files
        filename = fullfile(test_folder, test_files(i).name);
        test_results.filenames{i} = test_files(i).name;
        
        % Read audio file
        [audioIn, ~] = audioread(filename);
        if size(audioIn, 2) > 1
            audioIn = mean(audioIn, 2);
        end
        
        % Calculate energy ratio
        ratio = calculate_energy_ratio(audioIn, filter1, filter2);
        test_results.energy_ratios(i) = ratio;
        
        % Classify based on threshold
        if ratio > threshold
            test_results.predictions{i} = 'ambulance';
            test_results.confidences(i) = min(0.95, 0.5 + abs(ratio - threshold) / (2 * threshold));
        else
            test_results.predictions{i} = 'firetruck';
            test_results.confidences(i) = min(0.95, 0.5 + abs(threshold - ratio) / (2 * threshold));
        end
    end
end
