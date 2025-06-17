function [ambulance_files, firetruck_files] = load_training_data(train_folder)
% Load training data files
    ambulance_folder = fullfile(train_folder, 'ambulance');
    firetruck_folder = fullfile(train_folder, 'firetruck');
    
    ambulance_files = dir(fullfile(ambulance_folder, '*.wav'));
    firetruck_files = dir(fullfile(firetruck_folder, '*.wav'));
    
    % Add full path
    for i = 1:length(ambulance_files)
        ambulance_files(i).fullpath = fullfile(ambulance_folder, ambulance_files(i).name);
    end
    
    for i = 1:length(firetruck_files)
        firetruck_files(i).fullpath = fullfile(firetruck_folder, firetruck_files(i).name);
    end
end

function [ambulance_spectra, firetruck_spectra, fs] = analyze_frequency_content(ambulance_files, firetruck_files)
% Analyze frequency content of training data
    
    ambulance_spectra = [];
    firetruck_spectra = [];
    fs = 0;
    
    % Analyze ambulance files
    for i = 1:length(ambulance_files)
        [audioIn, fs] = audioread(ambulance_files(i).fullpath);
        if size(audioIn, 2) > 1
            audioIn = mean(audioIn, 2);
        end
        
        % Compute power spectral density
        [psd, freq] = pwelch(audioIn, [], [], [], fs);
        ambulance_spectra = [ambulance_spectra, psd];
    end
    
    % Analyze firetruck files
    for i = 1:length(firetruck_files)
        [audioIn, fs] = audioread(firetruck_files(i).fullpath);
        if size(audioIn, 2) > 1
            audioIn = mean(audioIn, 2);
        end
        
        % Compute power spectral density
        [psd, freq] = pwelch(audioIn, [], [], [], fs);
        firetruck_spectra = [firetruck_spectra, psd];
    end
end

function [filter1, filter2] = design_classification_filters(ambulance_spectra, firetruck_spectra, fs)
% Design bandpass filters based on spectral differences
    
    % Calculate mean spectra
    mean_ambulance = mean(ambulance_spectra, 2);
    mean_firetruck = mean(firetruck_spectra, 2);
    
    % Find frequency vector
    nfft = 2 * (length(mean_ambulance) - 1);
    freq = (0:length(mean_ambulance)-1) * fs / nfft;
    
    % Find frequency ranges where classes differ most
    % Filter 1: Low-mid frequency range (500-1500 Hz)
    f1_low = 500;
    f1_high = 1500;
    
    % Filter 2: Mid-high frequency range (1500-3000 Hz)
    f2_low = 1500;
    f2_high = 3000;
    
    % Design Butterworth bandpass filters
    nyquist = fs / 2;
    
    % Filter 1 design
    [b1, a1] = butter(4, [f1_low f1_high] / nyquist, 'bandpass');
    filter1.b = b1;
    filter1.a = a1;
    filter1.freq_range = [f1_low f1_high];
    
    % Filter 2 design
    [b2, a2] = butter(4, [f2_low f2_high] / nyquist, 'bandpass');
    filter2.b = b2;
    filter2.a = a2;
    filter2.freq_range = [f2_low f2_high];
    
    fprintf('Filter 1: %.0f - %.0f Hz\n', f1_low, f1_high);
    fprintf('Filter 2: %.0f - %.0f Hz\n', f2_low, f2_high);
end
