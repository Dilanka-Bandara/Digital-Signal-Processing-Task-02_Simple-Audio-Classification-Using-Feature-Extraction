function display_filter_results(test_results, threshold)
    fprintf('\n=== Emergency Vehicle Classification Results ===\n');
    fprintf('Threshold: %.3f\n', threshold);
    fprintf('File Name\t\tEnergy Ratio\tPrediction\tConfidence\n');
    fprintf('--------------------------------------------------------\n');
    n = length(test_results.filenames);
    if n == 0
        fprintf('No test files were processed.\n');
        return;
    end
    for i = 1:n
        fprintf('%-15s\t%.3f\t\t%-10s\t%.2f%%\n', ...
            test_results.filenames{i}, ...
            test_results.energy_ratios(i), ...
            test_results.predictions{i}, ...
            test_results.confidences(i) * 100);
    end
end
