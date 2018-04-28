%attack_name = "DeepFoolAttack";
%attack_name = "GradientAttack";
attack_name = "LBFGSAttack";

data_folder = str2mat('training_data/' + attack_name +'/rel_images');
imds = imageDatastore(data_folder,'IncludeSubfolders',true,'LabelSource','foldernames');


numTrainFiles = 750;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');


layers = [
    imageInputLayer([28 28 3])
    
    convolution2dLayer(3,8,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
   
    convolution2dLayer(3,16,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding',1)
    batchNormalizationLayer
    reluLayer
  
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];


options = trainingOptions('sgdm', ...
    'MaxEpochs',100, ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');


net = trainNetwork(imdsTrain,layers,options);

YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation)