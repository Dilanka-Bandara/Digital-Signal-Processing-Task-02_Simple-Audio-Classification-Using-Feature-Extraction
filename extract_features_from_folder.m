function [features_mfcc, features_fft, filenames] = extract_features_from_folder(folder_path)
    audio_files = dir(fullfile(folder_path, '*.wav'));
    num_files = length(audio_files);
    features_mfcc = [];
    features_fft = [];
    filenames = {};
    for i = 1:num_files
        filename = fullfile(folder_path, audio_files(i).name);
        try
            [audioIn, fs] = audioread(filename);
            if isempty(audioIn) || isempty(fs) || fs <= 0
                fprintf('Skipping file %s: Invalid audio or sample rate\n', filename);
                continue;
            end
            if size(audioIn,2) > 1
                audioIn = mean(audioIn,2);
            end
            window_length = round(0.025 * fs);
            overlap_length = round(0.015 * fs);
            if window_length < 2 || overlap_length < 0 || window_length > length(audioIn)
                fprintf('Skipping file %s: Invalid window/overlap length\n', filename);
                continue;
            end
            mfcc_coeffs = mfcc(audioIn, fs, 'WindowLength', window_length, ...
                'OverlapLength', overlap_length, 'NumCoeffs', 13);
            mfcc_features = mean(mfcc_coeffs, 2)';
            fft_features = extract_fft_features(audioIn, fs, window_length, overlap_length);
            features_mfcc = [features_mfcc; mfcc_features];
            features_fft = [features_fft; fft_features];
            filenames{end+1,1} = audio_files(i).name;
        catch ME
            fprintf('Error processing file %s: %s\n', filename, ME.message);
        end
    end
end

function fft_features = extract_fft_features(audioIn, fs, window_length, hop_length)
    if nargin < 3
        window_length = round(0.025 * fs);
        hop_length = round(0.010 * fs);
    end
    if window_length < 2 || hop_length < 1
        fft_features = nan(1,10);
        return;
    end
    num_frames = floor((length(audioIn) - window_length) / hop_length) + 1;
    if num_frames < 1
        fft_features = nan(1,10);
        return;
    end
    fft_matrix = zeros(num_frames, floor(window_length/2)+1);
    for i = 1:num_frames
        start_idx = (i-1)*hop_length + 1;
        end_idx = start_idx + window_length - 1;
        if end_idx > length(audioIn)
            break;
        end
        frame = audioIn(start_idx:end_idx) .* hamming(window_length);
        fft_frame = fft(frame);
        fft_matrix(i,:) = abs(fft_frame(1:floor(window_length/2)+1));
    end
    stats = [mean(fft_matrix,1), std(fft_matrix,0,1), max(fft_matrix,[],1), min(fft_matrix,[],1)];
    fft_features = stats(1:10:end);
end
