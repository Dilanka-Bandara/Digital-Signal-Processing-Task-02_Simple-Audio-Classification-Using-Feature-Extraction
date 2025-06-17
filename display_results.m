function display_results(results, feature_type)
% Display classification results
    fprintf('\n=== %s Classification Results ===\n', feature_type);
    fprintf('File Name\t\tCosine\tEuclidean\tManhattan\n');
    fprintf('-----------------------------------------------\n');
    
    for i = 1:length(results.filenames)
        fprintf('%-15s\t%d\t%d\t\t%d\n', ...
            results.filenames{i}, ...
            results.cosine_predictions(i), ...
            results.euclidean_predictions(i), ...
            results.manhattan_predictions(i));
    end
end

function display_filter_results(test_results, threshold)
% Display filter-based classification results
    fprintf('\n=== Emergency Vehicle Classification Results ===\n');
    fprintf('Threshold: %.3f\n', threshold);
    fprintf('File Name\t\tEnergy Ratio\tPrediction\tConfidence\n');
    fprintf('--------------------------------------------------------\n');
    
    for i = 1:length(test_results.filenames)
        fprintf('%-15s\t%.3f\t\t%-10s\t%.2f%%\n', ...
            test_results.filenames{i}, ...
            test_results.energy_ratios(i), ...
            test_results.predictions{i}, ...
            test_results.confidences(i) * 100);
    end
end

function save_results_to_file(results, filename)
% Save classification results to text file
    fid = fopen(filename, 'w');
    fprintf(fid, '%s Classification Results\n', results.feature_type);
    fprintf(fid, 'File Name,Cosine,Euclidean,Manhattan,Cosine_Conf,Euclidean_Conf,Manhattan_Conf\n');
    
    for i = 1:length(results.filenames)
        fprintf(fid, '%s,%d,%d,%d,%.3f,%.3f,%.3f\n', ...
            results.filenames{i}, ...
            results.cosine_predictions(i), ...
            results.euclidean_predictions(i), ...
            results.manhattan_predictions(i), ...
            results.cosine_confidences(i), ...
            results.euclidean_confidences(i), ...
            results.manhattan_confidences(i));
    end
    
    fclose(fid);
    fprintf('Results saved to %s\n', filename);
end

function save_filter_results(test_results, filename)
% Save filter-based results to text file
    fid = fopen(filename, 'w');
    fprintf(fid, 'Emergency Vehicle Classification Results\n');
    fprintf(fid, 'File Name,Energy Ratio,Prediction,Confidence\n');
    
    for i = 1:length(test_results.filenames)
        fprintf(fid, '%s,%.3f,%s,%.3f\n', ...
            test_results.filenames{i}, ...
            test_results.energy_ratios(i), ...
            test_results.predictions{i}, ...
            test_results.confidences(i));
    end
    
    fclose(fid);
    fprintf('Filter results saved to %s\n', filename);
end
