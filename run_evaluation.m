%% === constants ===

% the time window relative to each event that may be used for classification
epoch_range = [-0.2 0.8];

% time ranges over which per-channel features should be extracted
time_ranges = [0.25 0.3; 0.3 0.35; 0.35 0.4; 0.4 0.45; 0.45 0.5; 0.5 0.6];

% regularization parameter for the shrinkage LDA
% This is to prevent from overfitting
% LDA - Linear Discriminant Analysis - 
lambda = 0.2125;

%% === train an ERP classifier ===

% load the calibration data set, this is the data set
load ERP_CALIB

% strcmp - this function compares string and then returns if they are the same

% identify the sample latencies at which relevant events occur and their 
% associated target class (=0 no error, 1=error)
% We do not give train_events to the classifier
train_events = strcmp({ERP_CALIB.event.type},'S 11') | ....
strcmp({ERP_CALIB.event.type},'S 12') | strcmp({ERP_CALIB.event.type},'S 13');

% These are related to the training set also, since this have 238 values.
train_latencies = round([ERP_CALIB.event(train_events).latency]);

% These are the label for the training set, and
% there are two class either 1 or -1 also with 238 there are 238 data sets
train_labels = (~strcmp({ERP_CALIB.event(train_events).type},'S 11'))*2-1;

%% The actucal training step - I have to write this function
model = train_erp(ERP_CALIB.data,... % EEG
                  ERP_CALIB.srate,... % Fs
                  train_latencies,... % ev_lats
                  train_labels,... % ev_labels
                  epoch_range,... % epo_range 
                  time_ranges,... % time_ranges
                  lambda);    % lambda

%% === apply the classifier to each event in the test data ===
load ERP_TEST

% determine the relevant event latencies and true labels
test_events = strcmp({ERP_TEST.event.type},'S 11') |....
    strcmp({ERP_TEST.event.type},'S 12') | strcmp({ERP_TEST.event.type},'S 13');

% 
test_latencies = round([ERP_TEST.event(test_events).latency]);

% These are the test label that will score my prediction
test_labels = (~strcmp({ERP_TEST.event(test_events).type},'S 11'))*2-1;

% also get the sample range that is used to extract epochs relative to the events
epoch_samples = round(epoch_range(1)*ERP_TEST.srate) : round(epoch_range(2)*ERP_TEST.srate);

%% Prediction phase
% for each test event make the prediction
predictions = [];
for e=1:length(test_latencies)
    % extract the epoch
    EPO = ERP_TEST.data(:,epoch_samples + test_latencies(e));
    % classify it and record the prediction
    predictions(e) = test_erp(EPO,model);
end

%% === evaluate the loss on the test set ===
% At the moment this will return the 100 - accuracy rate.
loss = eval_mcr(test_labels,predictions);
fprintf('The mis-classification rate on the test set is %.2f%% percent.\n',100*loss);
fprintf('The accuracy rate on the test set is %.2f%% percent.\n',100*(1-loss));

% Convert the prediction and test_label to view in a row - simple 
predict2 = predictions';
test_labels2 = test_labels';

