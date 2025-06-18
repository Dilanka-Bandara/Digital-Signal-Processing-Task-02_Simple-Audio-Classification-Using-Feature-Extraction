function save_filter_results(test_results, filename)
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
