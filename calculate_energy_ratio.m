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
