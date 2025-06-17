function display_results(results, feature_type)
    fprintf('\n=== %s Classification Results ===\n', feature_type);
    fprintf('File Name\t\tCosine\tEuclidean\tManhattan\n');
    fprintf('-----------------------------------------------\n');
    n = length(results.filenames);
    if n == 0
        fprintf('No results to display.\n');
        return;
    end
    for i = 1:n
        fprintf('%-15s\t%d\t%d\t\t%d\n', ...
            results.filenames{i}, ...
            results.cosine_predictions(i), ...
            results.euclidean_predictions(i), ...
            results.manhattan_predictions(i));
    end
end
