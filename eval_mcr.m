function loss = eval_mcr(targets,predictions)
% Evaluate the mis-classification rate loss
% Loss = eval_mcr(Predictions,Targets)
%
% In:
%   Predictions : vector of predictions made by the classifier
%
%   Targets : vector of true labels (-1 / +1)
%
% Out:
%   Loss : mis-classification rate

% calculate mis-classification rate (TODO: fill in)
% loss = ...

loss = 0;

% 1. Loop via the target values and see if there are anything wrong
for x = 1:size(targets,2)
    if targets(x) ~= predictions(x)
        loss = loss + 1;
    end 
end
loss = loss / size(targets,2);
