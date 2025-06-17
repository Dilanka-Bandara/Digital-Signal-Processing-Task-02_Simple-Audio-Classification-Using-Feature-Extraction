function threshold = set_classification_threshold(ambulance_ratios, firetruck_ratios)
    mean_ambulance = mean(ambulance_ratios(~isinf(ambulance_ratios)));
    mean_firetruck = mean(firetruck_ratios(~isinf(firetruck_ratios)));
    threshold = (mean_ambulance + mean_firetruck) / 2;
    fprintf('Ambulance mean ratio: %.3f\n', mean_ambulance);
    fprintf('Firetruck mean ratio: %.3f\n', mean_firetruck);
    fprintf('Classification threshold: %.3f\n', threshold);
end
