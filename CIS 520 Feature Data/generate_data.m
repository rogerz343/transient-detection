function [Xtrain, ytrain, Xtest, ytest] = generate_data(ntrain, ntest)

small_sample('train.mat', ntrain, 'newtrain.mat');
small_sample('test.mat', ntest, 'newtest.mat');

train = load('newtrain.mat');
train = train.dat;
Xtrain = train(:,3:end);
ytrain = train(:,2);
test = load('newtest.mat');
test = test.dat;
Xtest = test(:,3:end);
ytest = test(:,2);

Xtrain(isnan(Xtrain)) = 0;
Xtest(isnan(Xtest)) = 0;

end