function results = classify_audio_files(unknown_features, class1_features, class2_features, unknown_files, feature_type)
    num_unknown = size(unknown_features, 1);
    results = struct();
    results.filenames = unknown_files;
    results.feature_type = feature_type;
    cosine_predictions = zeros(num_unknown, 1);
    euclidean_predictions = zeros(num_unknown, 1);
    manhattan_predictions = zeros(num_unknown, 1);
    cosine_confidences = zeros(num_unknown, 1);
    euclidean_confidences = zeros(num_unknown, 1);
    manhattan_confidences = zeros(num_unknown, 1);
    for i = 1:num_unknown
        unknown_feature = unknown_features(i, :);
        if any(isnan(unknown_feature))
            cosine_predictions(i) = 0;
            euclidean_predictions(i) = 0;
            manhattan_predictions(i) = 0;
            continue;
        end
        class1_similarities = arrayfun(@(j) cosine_similarity(unknown_feature, class1_features(j,:)), 1:size(class1_features,1));
        class2_similarities = arrayfun(@(j) cosine_similarity(unknown_feature, class2_features(j,:)), 1:size(class2_features,1));
        max_class1_sim = max(class1_similarities);
        max_class2_sim = max(class2_similarities);
        if max_class1_sim > max_class2_sim
            cosine_predictions(i) = 1;
            cosine_confidences(i) = max_class1_sim / (max_class1_sim + max_class2_sim);
        else
            cosine_predictions(i) = 2;
            cosine_confidences(i) = max_class2_sim / (max_class1_sim + max_class2_sim);
        end
        class1_distances = arrayfun(@(j) euclidean_distance(unknown_feature, class1_features(j,:)), 1:size(class1_features,1));
        class2_distances = arrayfun(@(j) euclidean_distance(unknown_feature, class2_features(j,:)), 1:size(class2_features,1));
        min_class1_dist = min(class1_distances);
        min_class2_dist = min(class2_distances);
        if min_class1_dist < min_class2_dist
            euclidean_predictions(i) = 1;
            euclidean_confidences(i) = min_class2_dist / (min_class1_dist + min_class2_dist);
        else
            euclidean_predictions(i) = 2;
            euclidean_confidences(i) = min_class1_dist / (min_class1_dist + min_class2_dist);
        end
        class1_manhattan = arrayfun(@(j) manhattan_distance(unknown_feature, class1_features(j,:)), 1:size(class1_features,1));
        class2_manhattan = arrayfun(@(j) manhattan_distance(unknown_feature, class2_features(j,:)), 1:size(class2_features,1));
        min_class1_manh = min(class1_manhattan);
        min_class2_manh = min(class2_manhattan);
        if min_class1_manh < min_class2_manh
            manhattan_predictions(i) = 1;
            manhattan_confidences(i) = min_class2_manh / (min_class1_manh + min_class2_manh);
        else
            manhattan_predictions(i) = 2;
            manhattan_confidences(i) = min_class1_manh / (min_class1_manh + min_class2_manh);
        end
    end
    results.cosine_predictions = cosine_predictions;
    results.euclidean_predictions = euclidean_predictions;
    results.manhattan_predictions = manhattan_predictions;
    results.cosine_confidences = cosine_confidences;
    results.euclidean_confidences = euclidean_confidences;
    results.manhattan_confidences = manhattan_confidences;
end

function similarity = cosine_similarity(vec1, vec2)
    dot_product = dot(vec1, vec2);
    norm1 = norm(vec1);
    norm2 = norm(vec2);
    if norm1 == 0 || norm2 == 0
        similarity = 0;
    else
        similarity = dot_product / (norm1 * norm2);
    end
end

function distance = euclidean_distance(vec1, vec2)
    distance = sqrt(sum((vec1 - vec2).^2));
end

function distance = manhattan_distance(vec1, vec2)
    distance = sum(abs(vec1 - vec2));
end
