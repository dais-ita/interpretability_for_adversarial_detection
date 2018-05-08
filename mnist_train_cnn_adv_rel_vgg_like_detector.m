%attack_name = "DeepFoolAttack";
attack_name = "GradientAttack";
%attack_name = "LBFGSAttack";

data_folder = str2mat('training_data/' + attack_name +'/rel_images');
imds = imageDatastore(data_folder,'IncludeSubfolders',true,'LabelSource','foldernames');


numTrainFiles = round(size(imds.Labels,1) * 0.4); %num train files per class so 0.8/2 = 0.4
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');


layers = [
    imageInputLayer([28 28 3])
    
    convolution2dLayer(3,64,'Padding',1)
    reluLayer
    convolution2dLayer(3,64,'Padding',1)
    reluLayer
    convolution2dLayer(3,64,'Padding',1)
    reluLayer
    convolution2dLayer(3,64,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
   
    convolution2dLayer(3,128,'Padding',1)
    reluLayer
      convolution2dLayer(3,128,'Padding',1)
    reluLayer
      convolution2dLayer(3,128,'Padding',1)
    reluLayer
    convolution2dLayer(3,128,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,256,'Padding',1)
    batchNormalizationLayer
    reluLayer
  
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];


options = trainingOptions('sgdm', ...
    'MaxEpochs',1000, ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',30, ...
    'ValidationPatience',50, ...
    'Verbose',false, ...
    'Plots','training-progress');


net = trainNetwork(imdsTrain,layers,options);

YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation)