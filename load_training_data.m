function [ambulance_files, firetruck_files] = load_training_data(train_folder)
    ambulance_folder = fullfile(train_folder, 'ambulance');
    firetruck_folder = fullfile(train_folder, 'firetruck');
    ambulance_struct = dir(fullfile(ambulance_folder, '*.wav'));
    firetruck_struct = dir(fullfile(firetruck_folder, '*.wav'));
    for i = 1:length(ambulance_struct)
        ambulance_files(i).fullpath = fullfile(ambulance_folder, ambulance_struct(i).name);
    end
    for i = 1:length(firetruck_struct)
        firetruck_files(i).fullpath = fullfile(firetruck_folder, firetruck_struct(i).name);
    end
end
