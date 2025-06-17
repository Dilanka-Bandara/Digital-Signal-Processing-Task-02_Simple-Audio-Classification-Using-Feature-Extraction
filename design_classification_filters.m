function [filter1, filter2] = design_classification_filters(~, ~, fs)
    if isempty(fs) || isnan(fs) || fs <= 0
        error('Sampling rate fs is invalid. Check your audio files and code.');
    end
    nyquist = fs / 2;
    % Typical siren ranges: ambulance (650-1000 Hz), firetruck (650-1550 Hz)[8]
    f1_low = 650;   % Hz
    f1_high = 1000; % Hz
    f2_low = 1200;  % Hz
    f2_high = 1550; % Hz
    % Ensure cutoffs are valid
    if any([f1_low, f1_high, f2_low, f2_high] <= 0) || ...
       any([f1_low, f1_high, f2_low, f2_high] >= nyquist)
        error('Cutoff frequencies must be positive and less than Nyquist frequency.');
    end
    [b1, a1] = butter(4, [f1_low f1_high] / nyquist, 'bandpass');
    filter1.b = b1;
    filter1.a = a1;
    filter1.freq_range = [f1_low f1_high];
    [b2, a2] = butter(4, [f2_low f2_high] / nyquist, 'bandpass');
    filter2.b = b2;
    filter2.a = a2;
    filter2.freq_range = [f2_low f2_high];
    fprintf('Filter 1: %.0f - %.0f Hz\n', f1_low, f1_high);
    fprintf('Filter 2: %.0f - %.0f Hz\n', f2_low, f2_high);
end
