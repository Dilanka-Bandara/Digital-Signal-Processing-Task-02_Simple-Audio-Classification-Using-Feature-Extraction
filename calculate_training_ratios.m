function [ambulance_ratios, firetruck_ratios] = calculate_training_ratios(ambulance_files, firetruck_files, filter1, filter2)
    ambulance_ratios = zeros(length(ambulance_files), 1);
    firetruck_ratios = zeros(length(firetruck_files), 1);
    for i = 1:length(ambulance_files)
        [audioIn, ~] = audioread(ambulance_files(i).fullpath);
        if size(audioIn, 2) > 1
            audioIn = mean(audioIn, 2);
        end
        ambulance_ratios(i) = calculate_energy_ratio(audioIn, filter1, filter2);
    end
    for i = 1:length(firetruck_files)
        [audioIn, ~] = audioread(firetruck_files(i).fullpath);
        if size(audioIn, 2) > 1
            audioIn = mean(audioIn, 2);
        end
        firetruck_ratios(i) = calculate_energy_ratio(audioIn, filter1, filter2);
    end
end

function ratio = calculate_energy_ratio(audioIn, filter1, filter2)
    filtered1 = filtfilt(filter1.b, filter1.a, audioIn);
    filtered2 = filtfilt(filter2.b, filter2.a, audioIn);
    energy1 = sum(filtered1.^2);
    energy2 = sum(filtered2.^2);
    if energy2 == 0
        ratio = inf;
    else
        ratio = energy1 / energy2;
    end
end
