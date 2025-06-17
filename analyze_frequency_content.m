function [ambulance_spectra, firetruck_spectra, fs] = analyze_frequency_content(ambulance_files, firetruck_files)
    nfft = 2048; % Fixed FFT length for all files
    ambulance_spectra = [];
    firetruck_spectra = [];
    fs = [];
    % Analyze ambulance files
    for i = 1:length(ambulance_files)
        [audioIn, fs_] = audioread(ambulance_files(i).fullpath);
        if size(audioIn, 2) > 1
            audioIn = mean(audioIn, 2);
        end
        [psd, ~] = pwelch(audioIn, hamming(nfft), round(0.5*nfft), nfft, fs_);
        if isempty(fs)
            fs = fs_;
        end
        if isempty(ambulance_spectra)
            ambulance_spectra = psd;
        else
            ambulance_spectra = [ambulance_spectra, psd];
        end
    end
    % Analyze firetruck files
    for i = 1:length(firetruck_files)
        [audioIn, fs_] = audioread(firetruck_files(i).fullpath);
        if size(audioIn, 2) > 1
            audioIn = mean(audioIn, 2);
        end
        [psd, ~] = pwelch(audioIn, hamming(nfft), round(0.5*nfft), nfft, fs_);
        if isempty(firetruck_spectra)
            firetruck_spectra = psd;
        else
            firetruck_spectra = [firetruck_spectra, psd];
        end
    end
end
