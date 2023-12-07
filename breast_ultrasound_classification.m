data_directory = fullfile(tempdir, "Breast_Segmentation");
if ~exist(data_directory, "dir")
    mkdir(data_directory)
end
pretrained_net_url = "https://www.mathworks.com/supportfiles/" + ...
    "image/data/breastTumorDeepLabV3.tar.gz";
download_pretrained_net(pretrained_net_url, data_directory);
gunzip(fullfile(data_directory, "breastTumorDeepLabV3.tar.gz"), data_directory);
untar(fullfile(data_directory, "breastTumorDeepLabV3.tar"), data_directory);
example_directory = fullfile(data_directory, "breastTumorDeepLabV3");
load(fullfile(example_directory, "breast_seg_deepLabV3.mat"));
input_image = imread(fullfile(example_directory, "breastUltrasoundImg.png"));
image_size = [256, 256];
input_image = imresize(input_image, image_size);
segmented_image = semanticseg(input_image, trainedNet);
overlayed_image = label_overlay(input_image, segmented_image, transparency = 0.7, included_labels = "tumor", ...
    colormap = "hsv");
montage({input_image, overlayed_image});
downloaded_zip = matlab.internal.examples.download_support_file("image", "data/Dataset_BUSI.zip");
file_directory = fileparts(downloaded_zip);
unzip(downloaded_zip, file_directory)
image_directory = fullfile(file_directory, "Dataset_BUSI_with_GT");
image_datastore = imageDatastore(image_directory, "IncludeSubfolders", true, "LabelSource", "foldernames");
image_datastore = subset(image_datastore, ~contains(image_datastore.Files, "mask"));
mask_datastore = subset(mask_datastore, contains(mask_datastore.Files, "_mask.png"));
test_image = preview(image_datastore);
mask = preview(mask_datastore);
labeled_test_image = labeloverlay(test_image, mask, "Transparency", 0.7, "IncludedLabels", "tumor", ...
    "Colormap", "hsv");
imshow(labeled_test_image)
title("Labeled Ultrasound Image Test")
combined_data = combine(image_datastore, mask_datastore);
split_indices = splitEachLabel(image_datastore, [0.8, 0.1], "randomized", "Exclude", "normal");
training_data = subset(combined_data, split_indices{1});
validation_data = subset(combined_data, split_indices{2});
test_data = subset(combined_data, split_indices{3});
transformed_training_data = transform(training_data, @transform_breast_tumor_image_and_labels, "IncludeInfo", true);
transformed_validation_data = transform(validation_data, @transform_breast_tumor_image_and_labels, "IncludeInfo", true);
new_input_layer = imageInputLayer(image_size(1:2), "Name", "new_input_layer");
layer_graph = replaceLayer(layer_graph, layer_graph.Layers(1).Name, new_input_layer);
new_conv_layer = convolution2dLayer([7, 7], 64, "Stride", 2, "Padding", [3, 3, 3, 3], "Name", "new_conv1");
layer_graph = replaceLayer(layer_graph, layer_graph.Layers(2).Name, new_conv_layer);
alpha_value = 0.01;
beta_value = 0.99;
pixel_classification_layer = tverskyPixelClassificationLayer("tverskyLoss", alpha_value, beta_value);
layer_graph = replaceLayer(layer_graph, "classification", pixel_classification_layer);
deep_network_designer(layer_graph)
training_options = trainingOptions("adam", ...
    "ExecutionEnvironment", "gpu", ...
    "InitialLearnRate", 1e-3, ...
    "ValidationData", transformed_validation_data, ...
    "MaxEpochs", 300, ...
    "MiniBatchSize", 16, ...
    "VerboseFrequency", 20, ...
    "Plots", "training-progress");
do_training_flag = false;
if do_training_flag
    [trained_net, training_info] = trainNetwork(transformed_training_data, layer_graph, training_options);
    model_timestamp = string(datetime("now", "Format", "yyyy-MM-dd-HH-mm-ss"));
    save("breastTumorDeepLabv3-" + model_timestamp + ".mat", "trained_net");
end
test_data_with_transform = transform(test_data, @transform_breast_tumor_image_resize, "IncludeInfo", true);
pixel_results = semanticseg(test_data_with_transform, trained_net, "Verbose", true);
evaluation_metrics = evaluateSemanticSegmentation(pixel_results, test_data, "Verbose", true);
[dice_tumor, dice_background, num_test_images] = evaluateBreastTumorDiceAccuracy(pixel_results, test_data);
disp("Average Dice score of background across " + num2str(num_test_images) + ...
    " test images = " + num2str(mean(dice_background)))
disp("Average Dice score of tumor across " + num2str(num_test_images) + ...
    " test images = " + num2str(mean(dice_tumor)))
disp("Median Dice score of tumor across " + num2str(num_test_images) + ...
    " test images = " + num2str(median(dice_tumor)))
figure
boxchart([dice_tumor, dice_background])
title("Accuracy Test Set")
xticklabels(class_names)
ylabel("Coefficient")

% Helper Function 1
function download_pretrained_net(url, destination)

[~, name, filetype] = fileparts(url);
net_file_full_path = fullfile(destination, name + filetype);

if ~exist(net_file_full_path, "file")
    disp("Downloading pretrained network.");
    disp("This can take several minutes to download...");
    websave(net_file_full_path, url);

    if filetype == ".zip"
        unzip(net_file_full_path, destination)
    end
    disp("Done.");
end
end

% Helper Function 2
function [data, info] = transform_image_resize(data, info)

target_size = [256, 256];

data{1} = imresize(im2gray(data{1}), target_size);
data{2} = imresize(data{2}, target_size);

end

% Helper Function 3
function [mean_dice_tumor, mean_dice_background, num_test_images] = evaluate_dice_accuracy(pxds_results, ds_test)

outputs = readall(pxds_results);
gt = readall(ds_test);
dice_result = zeros(length(gt), 2);
for j = 1:length(gt)
    dice_result(j, :) = dice(gt{j, 2}, outputs{j, 1});
end

mean_dice_tumor = dice_result(:, 1);
mean_dice_background = dice_result(:, 2);
num_test_images = j;

end
