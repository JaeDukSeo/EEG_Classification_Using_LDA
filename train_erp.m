function model = train_erp(EEG, Fs, ev_lats, ev_labels, epo_range, time_ranges, lambda)
% Train an ERP classifier on the given data
% Model = train_erp(Data,SamplingRate,EventLatencies,
%                   EventLabels,EpochRange,TimePoints,Lambda)
% There are 7 parameters given in the algorhythm


% In:
%   Data : raw multi-channel EEG signal, size is [#channels x #samples]
%
%   SamplingRate : sampling rate of the data, in Hz
%
%   EventLatencies : vector of sample offsets at which events occur
%
%   EventLabels : vector of true labels for each event (-1 = first class, +1 = second class)
%
%   EpochRange : time range relative to each event that shall be used for training
%                this is a 2-element vector with values in seconds [begin, end]
%
%   TimeRanges : time ranges in seconds relative to the epoch event
%                this is a [#ranges x 2] matrix with ranges in the rows
%                these ranges determine the time windows for which average features
%                should be extracted from the epochs
%
%   Lambda : regularization parameter for shrinkage LDA (between 0 and 1)
%
% Out:
%   Model : matlab struct that contains the model's parameters
%           (classifier weights, temporal filter)

% convert the epoch range into a vector sample offsets relative to the event
% (e.g., [-3,-2,-1,0,1,2,3,4,5,6])
% whos wnd = wnd   (Matrix) 1x101     double   
wnd = round(epo_range(1)*Fs) : round(epo_range(2)*Fs);

% convert time ranges into a cell array of sample offset vectors that can be 
% used to index the time points within an epoch
% time_ranges = 0.25 0.3 to 0.6 ( I guess this is the time range)
for r=1:length(time_ranges)
    model.ranges{r} = 1 + (round(time_ranges(r,1)*Fs) : ...
                           round(time_ranges(r,2)*Fs)) - wnd(1);
end

% extract training epochs (EPO is a 3d array of size (#channels x #samples x #trials)
EPO = EEG(:, repmat(ev_lats,length(wnd),1) + repmat(wnd',1,length(ev_lats)));
% Then reshape into a matrtix
% whos EPO =   20x101x238 ( One matrix is 20 * 101 and there are 238 layers of matrix)
EPO = reshape(EPO,size(EPO,1),[],length(ev_lats));

% determine number of channels, epoch time points, trials, and number of time ranges
[nbchan,pnts,trials] = size(EPO); % nbchan = 20, trials = 238
nbranges = size(time_ranges,1);

% extract features for each epoch - Now the features extraction
% features is a [#trials x #dims] matrix of feature vectors per trial

% Why is the feature only 2D matrix ? nbranges = 6 and nbchan = 20, total 120
% whos features = 238x120
features = zeros(trials, nbranges * nbchan);

% Now we are extracting the features
for e=1:length(ev_lats)
    % get epoch X and subtract per-trial mean
    X = EPO(:,:,e) - repmat(mean(EPO(:,:,e),2),1,size(EPO,2));
    
    % extract per-trial features
    trialfeatures = zeros(nbranges,nbchan);
    % for each time range...
    for r=1:length(model.ranges)
        % calculate the mean for each channel and store it
        trialfeatures(r,:) = mean(X(:,model.ranges{r})');
    end
    
    % turn per-trial features into a vector and store
    features(e,:) = trialfeatures(:);
end

% train shrinkage LDA classifier (TODO: fill in)
% The module have the w - weight and the b - bias value that can classify
% whos model = ranges: {1x6 cell}
%model.w = ...
%model.b = ...

one_array = [];
neg_one_array = [];

% Performing Linear discriminant analysis 
% 1. Get the mean value for each labels, for each label if they are either 
% labeled as 1 or -1
for x = 1:size(ev_labels,2)
    if ev_labels(x) == 1 
       one_array = [features(x,:);one_array];
    else  
       neg_one_array = [features(x,:);neg_one_array];
    end 
end

U_one_mean = mean(one_array);
U_neg_one_mean = mean(neg_one_array);

% 2. Calculate the Covariance for each labeled data
Covariance_one = cov(one_array);
Covariance_neg_one = cov(neg_one_array) ;

% 2.5 Add the regualrazation value --------- Both value S and I set to one 
Covariance_one = (1-lambda).*Covariance_one + (lambda) * (1) * eye(size(Covariance_one));
Covariance_neg_one = (1-lambda).*Covariance_neg_one + (lambda) * (1) * eye(size(Covariance_neg_one));

% 3. Calculate the theta value, which is w value
theta = (Covariance_one+Covariance_neg_one)\(U_one_mean - U_neg_one_mean)';
model.w = theta;

% 4. Calculate the bias value
%bias = -pinv(theta) * (U_one_mean + U_neg_one_mean)'/2;
model.b = -theta\(U_one_mean + U_neg_one_mean)'/2;

% There are 238 coloum of the features and 120 row values
%[rows, columns] = size(features);

% And I also need to use ev_labels 
%for row = 1:rows
    % This is 1 * 120 vector
%    current_feature = features(row,:);
%end

