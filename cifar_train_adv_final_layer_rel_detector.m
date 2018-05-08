attack_name = "DeepFoolAttack";
%attack_name = "GradientAttack";
% attack_name = "LBFGSAttack";

rel_path = str2mat('cifar_training_data/'+attack_name+'_rels.csv');
rel_gt_path = str2mat('cifar_training_data/'+attack_name+'_rels_gt.csv');


rels = csvread(rel_path);
rels_gt = csvread(rel_gt_path);
rels_gt = categorical(rels_gt(:,1) == 0);

num_datas = size(rels,1);
data_size = size(rels,2);

final_class1 = num_datas/2;
class1_train_end = round(final_class1*0.8);

class2_start = final_class1+1;
class2_train_end = class2_start + class1_train_end-1;

num_train = class1_train_end * 2;

train_class_1 = rels(1:class1_train_end,:);
train_class_2 = rels(class2_start:class2_train_end,:);
x_train =  vertcat(train_class_1,train_class_2);
x_train = reshape(x_train,[data_size,1,1,num_train]);

gt_train_class_1 = rels_gt(1:class1_train_end,:);
gt_train_class_2 = rels_gt(class2_start:class2_train_end,:);
y_train =  vertcat(gt_train_class_1,gt_train_class_2);


val_class_1_start = class1_train_end+1;
val_class_1 = rels(val_class_1_start:final_class1,:);

val_class_2_start = class2_train_end+1;
val_class_2 = rels(val_class_2_start:end,:);

num_val = num_datas - num_train;

x_val = vertcat(val_class_1,val_class_2);
x_val = reshape(x_val,[data_size,1,1,num_val]);

gt_val_class_1 = rels_gt(val_class_1_start:final_class1,:);
gt_val_class_2 = rels_gt(val_class_2_start:end,:);
y_val = vertcat(gt_val_class_1,gt_val_class_2);

layers = [
    imageInputLayer([data_size 1 1])
    fullyConnectedLayer(100)
    fullyConnectedLayer(100)
    fullyConnectedLayer(100)
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm', ...
    'MaxEpochs',100, ...
    'ValidationData',{x_val,y_val}, ...
    'ValidationFrequency',30, ...
    'ValidationPatience',30, ...
    'Verbose',false, ...
    'Plots','training-progress');


net = trainNetwork(x_train,y_train,layers,options);

YPred = classify(net,x_val);
YValidation = y_val;

accuracy = sum(YPred == YValidation)/numel(YValidation)