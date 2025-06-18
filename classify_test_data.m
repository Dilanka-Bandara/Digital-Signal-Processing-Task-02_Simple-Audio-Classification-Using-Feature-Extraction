function test_results = classify_test_data(test_folder, filter1, filter2, threshold, fs)
    test_files = dir(fullfile(test_folder, '*.wav'));
    num_test_files = length(test_files);
    test_results.filenames = {};
    test_results.energy_ratios = [];
    test_results.predictions = {};
    test_results.confidences = [];
    if num_test_files == 0
        fprintf('No test files found in %s\n', test_folder);
        return;
    end
    for i = 1:num_test_files
        filename = fullfile(test_files(i).folder, test_files(i).name);
        try
            [audioIn, ~] = audioread(filename);
            if size(audioIn, 2) > 1
                audioIn = mean(audioIn, 2);
            end
            ratio = calculate_energy_ratio(audioIn, filter1, filter2);
            test_results.filenames{end+1,1} = test_files(i).name;
            test_results.energy_ratios(end+1,1) = ratio;
            if ratio > threshold
                test_results.predictions{end+1,1} = 'ambulance';
                test_results.confidences(end+1,1) = min(0.95, 0.5 + abs(ratio - threshold) / (2 * threshold));
            else
                test_results.predictions{end+1,1} = 'firetruck';
                test_results.confidences(end+1,1) = min(0.95, 0.5 + abs(threshold - ratio) / (2 * threshold));
            end
        catch ME
            fprintf('Skipping file %s: could not read or process (%s).\n', filename, ME.message);
        end
    end
end
