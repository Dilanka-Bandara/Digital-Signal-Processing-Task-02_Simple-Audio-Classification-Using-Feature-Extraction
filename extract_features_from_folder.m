function [features_mfcc, features_fft, filenames] = extract_features_from_folder(folder_path)
% Extract MFCC and FFT features from all audio files in a folder
    
    % Get all .wav files in the folder
    audio_files = dir(fullfile(folder_path, '*.wav'));
    num_files = length(audio_files);
    
    if num_files == 0
        error('No .wav files found in folder: %s', folder_path);
    end
    
    % Initialize feature matrices
    features_mfcc = [];
    features_fft = [];
    filenames = cell(num_files, 1);
    
    for i = 1:num_files
        filename = fullfile(folder_path, audio_files(i).name);
        filenames{i} = audio_files(i).name;
        
        try
            % Read audio file
            [audioIn, fs] = audioread(filename);
            
            % Convert to mono if stereo
            if size(audioIn, 2) > 1
                audioIn = mean(audioIn, 2);
            end
            
            % Extract MFCC features
            mfcc_coeffs = mfcc(audioIn, fs, 'WindowLength', round(0.025*fs), ...
                              'OverlapLength', round(0.015*fs), 'NumCoeffs', 13);
            
            % Take mean of MFCC coefficients across time frames
            mfcc_features = mean(mfcc_coeffs, 2)';
            
            % Extract FFT-based features
            fft_features = extract_fft_features(audioIn, fs);
            
            % Store features
            features_mfcc = [features_mfcc; mfcc_features];
            features_fft = [features_fft; fft_features];
            
        catch ME
            fprintf('Error processing file %s: %s\n', filename, ME.message);
        end
    end
end

function fft_features = extract_fft_features(audioIn, fs)
% Extract frequency domain features using FFT
    
    % Window the signal
    window_length = round(0.025 * fs); % 25ms window
    hop_length = round(0.010 * fs);    % 10ms hop
    
    % Apply windowing and compute FFT
    num_frames = floor((length(audioIn) - window_length) / hop_length) + 1;
    fft_matrix = zeros(num_frames, window_length/2 + 1);
    
    for i = 1:num_frames
        start_idx = (i-1) * hop_length + 1;
        end_idx = start_idx + window_length - 1;
        
        if end_idx <= length(audioIn)
            frame = audioIn(start_idx:end_idx) .* hamming(window_length);
            fft_frame = fft(frame);
            fft_matrix(i, :) = abs(fft_frame(1:window_length/2 + 1));
        end
    end
    
    % Extract statistical features from FFT
    fft_features = [
        mean(fft_matrix, 1), ...           % Mean magnitude spectrum
        std(fft_matrix, 0, 1), ...         % Standard deviation
        max(fft_matrix, [], 1), ...        % Maximum values
        min(fft_matrix, [], 1)             % Minimum values
    ];
    
    % Reduce dimensionality by taking every 10th feature
    fft_features = fft_features(1:10:end);
end
