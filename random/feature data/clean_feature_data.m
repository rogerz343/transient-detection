function [] = clean_feature_data(ptrain,ptest,numfolds)
% Takes proportions ptrain and ptest, and splits the data into ptrain%
% training data and ptest% testing data
% Also takes numfolds and uses the remaining data (i.e. if ptrain + ptest =
% 1) for CV data with numfolds folds

ID = load('ID.mat');
ID = ID.ID;
labels = load('labels.mat');
labels = labels.labels;
data = load('data.mat');
data = data.data;

% Index examples by label 0 or 1
zero_index = find(labels == 0);
one_index = find(labels == 1);
% Numbers of 0s and 1s
n0 = numel(zero_index);
n1 = numel(one_index);
% Percentages of train, test --> %CV = 1-%train-%test
proportions = [ptrain, ptest];
ntrain0 = floor(n0*proportions(1));
ntest0 = floor(n0*proportions(2));
nCV0 = n0-ntrain0-ntest0;
ntrain1 = floor(n1*proportions(1));
ntest1 = floor(n1*proportions(2));
nCV1 = n1-ntrain1-ntest1;
% Break 0s into training, testing and CV data
rand0 = zero_index(randperm(n0));
train0 = rand0(1:ntrain0);
train0 = [ID(train0) labels(train0) data(train0,:)];
test0 = rand0((ntrain0+1):(ntrain0+ntest0));
test0 = [ID(test0) labels(test0) data(test0,:)];
CV0 = rand0((ntrain0+ntest0+1):end);
CV0 = [ID(CV0) labels(CV0) data(CV0,:)];
% Break 1s into training, testing and CV data
rand1 = one_index(randperm(n1));
train1 = rand1(1:ntrain1);
train1 = [ID(train1) labels(train1) data(train1,:)];
test1 = rand1((ntrain1+1):(ntrain1+ntest1));
test1 = [ID(test1) labels(test1) data(test1,:)];
CV1 = rand1((ntrain1+ntest1+1):end);
CV1 = [ID(CV1) labels(CV1) data(CV1,:)];

% Compile training and testing data
train = [train0; train1];
dat = train;
save('train.mat','dat');
test = [test0; test1];
dat = test;
save('test.mat','dat');
disp('Training and Testing Data Done!');

% Split CV data into numfolds folds
foldsize0 = floor(nCV0/numfolds);
foldsize1 = floor(nCV1/numfolds);

% Starting and ending indices for each folder
CVindex0 = zeros(1,(numfolds+1));
CVindex0(2:(end-1)) = (1:(numfolds-1))*foldsize0;
CVindex0(end) = nCV0;
CVindex1 = zeros(1,(numfolds+1));
CVindex1(2:(end-1)) = (1:(numfolds-1))*foldsize1;
CVindex1(end) = nCV1;

% Compile CV data for each fold
for i = 1:numfolds
    fold0 = CV0(CVindex0(i)+1:CVindex0(i+1),:);
    fold1 = CV1(CVindex1(i)+1:CVindex1(i+1),:);
    dat = [fold0; fold1];
    foldname = ['CV',num2str(i),'.mat'];
    save(foldname, 'dat');
end

end