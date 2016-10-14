function y = test_erp(X, model)
% Apply the ERP classifier to a data epoch
% y = test_erp(X, Model)
%
% In:
%   X : a single EEG epoch (#channels x #time points)
%
%   Model : the predictive model that was learned in train_erp;
%           this is a MATLAB struct
%
% Out:
%   y : the predicted class label (between -1 and +1)

% subtract trial mean
X = X - repmat(mean(X,2),1,size(X,2));

% extract per-trial features and the dim of the trialfeatures
% are 6 * 20!!
trialfeatures = zeros(length(model.ranges),size(X,1));
% for each time range...
for r=1:length(model.ranges)
    % calculate the mean for each channel and store it
    trialfeatures(r,:) = mean(X(:,model.ranges{r})');
end

% Change the trialfeatures in a form that we want to use 1 * 120 
trialfeatures = trialfeatures(:);

% apply LDA classifier (TODO: fill in)
%y = ... (model.w * trialfeatures) + model.b
y  = sign((model.w' * trialfeatures) + model.b);