clear;
clc;

projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(projectRoot));

pRL = rlParameters();

actorNetwork = createActorNetwork(pRL);
criticNetwork = createCriticNetwork(pRL);

%% Case 1: No intervention

batchSize = 8;

batch.observation = dlarray( ...
    single(randn(pRL.observationDimension, batchSize)), ...
    'CB');

batch.action = dlarray( ...
    single(0.1 * randn(pRL.actionDimension, batchSize)), ...
    'CB');

batch.correctedAction = dlarray( ...
    single(zeros(pRL.actionDimension, batchSize)), ...
    'CB');

batch.oldLogProbability = dlarray( ...
    single(-ones(1, batchSize)), ...
    'CB');

batch.advantage = dlarray( ...
    single(randn(1, batchSize)), ...
    'CB');

batch.return = dlarray( ...
    single(randn(1, batchSize)), ...
    'CB');

batch.correctionAmplitude = dlarray( ...
    single(zeros(1, batchSize)), ...
    'CB');

batch.interventionFlag = ...
    false(1, batchSize);

[actorGradients, criticGradients, ...
    actorLoss, ppoLoss, bcLoss, criticLoss] = ...
    dlfeval( ...
    @calculateCBFGGradients, ...
    actorNetwork, ...
    criticNetwork, ...
    batch, ...
    pRL);

assert(isfinite(extractdata(actorLoss)));
assert(isfinite(extractdata(ppoLoss)));
assert(isfinite(extractdata(bcLoss)));
assert(isfinite(extractdata(criticLoss)));

assert(extractdata(bcLoss) == 0);

assert(istable(actorGradients));
assert(istable(criticGradients));

disp("Case 1 passed.");

%% Case 2: Mixed intervention

batch.interventionFlag = ...
    [false false true true false true false true];

batch.correctionAmplitude = dlarray( ...
    single([0 0 0.1 0.2 0 0.3 0 0.15]), ...
    'CB');

correctedAction = extractdata(batch.action);

correctedAction(:, batch.interventionFlag) = ...
    correctedAction(:, batch.interventionFlag) + 0.05;

batch.correctedAction = dlarray( ...
    single(correctedAction), ...
    'CB');

[actorGradients, criticGradients, ...
    actorLoss, ppoLoss, bcLoss, criticLoss] = ...
    dlfeval( ...
    @calculateCBFGGradients, ...
    actorNetwork, ...
    criticNetwork, ...
    batch, ...
    pRL);

assert(isfinite(extractdata(actorLoss)));
assert(isfinite(extractdata(ppoLoss)));
assert(isfinite(extractdata(bcLoss)));
assert(isfinite(extractdata(criticLoss)));

assert(extractdata(bcLoss) > 0);

assert(istable(actorGradients));
assert(istable(criticGradients));

disp("Case 2 passed.");

%% Gradient check

for i = 1:height(actorGradients)
    gradient = actorGradients.Value{i};

    if ~isempty(gradient)
        assert(all(isfinite(extractdata(gradient)), 'all'));
    end
end

for i = 1:height(criticGradients)
    gradient = criticGradients.Value{i};

    if ~isempty(gradient)
        assert(all(isfinite(extractdata(gradient)), 'all'));
    end
end

disp("Gradient check passed.");

disp("CBFG gradient test passed.");