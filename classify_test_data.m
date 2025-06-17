function test_results = classify_test_data(test_folder, filter1, filter2, threshold, fs)
    test_files = dir(fullfile(test_folder, '*.wav'));
    num_test_files = length(test_files);
    test_results.filenames = cell(num_test_files, 1);
    test_results.energy_ratios = zeros(num_test_files, 1);
    test_results.predictions = cell(num_test_files, 1);
    test_results.confidences = zeros(num_test_files, 1);
    for i = 1:num_test_files
        filename = fullfile(test_folder, test_files(i).name);
        test_results.filenames{i} = test_files(i).name;
        [audioIn, ~] = audioread(filename);
        if size(audioIn, 2) > 1
            audioIn = mean(audioIn, 2);
        end
        ratio = calculate_energy_ratio(audioIn, filter1, filter2);
        test_results.energy_ratios(i) = ratio;
        if ratio > threshold
            test_results.predictions{i} = 'ambulance';
            test_results.confidences(i) = min(0.95, 0.5 + abs(ratio - threshold) / (2 * threshold));
        else
            test_results.predictions{i} = 'firetruck';
            test_results.confidences(i) = min(0.95, 0.5 + abs(threshold - ratio) / (2 * threshold));
        end
    end
end
