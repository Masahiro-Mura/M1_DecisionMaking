clear;
clc;

projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(projectRoot));

pRL = rlParameters();

actorNetwork = createActorNetwork(pRL);
criticNetwork = createCriticNetwork(pRL);

buffer = createExperienceBuffer(pRL);

numSamples = 32;

for i = 1:numSamples

    observation = randn( ...
        pRL.observationDimension, 1);

    nextObservation = randn( ...
        pRL.observationDimension, 1);

    action = 0.1 * randn( ...
        pRL.actionDimension, 1);

    correctedAction = action;

    if mod(i, 4) == 0
        correctedAction = ...
            action + [0.05; -0.05];

        correctionAmplitude = ...
            norm(correctedAction - action);

        interventionFlag = true;
    else
        correctionAmplitude = 0;
        interventionFlag = false;
    end

    reward = randn();
    done = false;

    logProbability = -1.0;
    value = 0.5 * randn();

    buffer = storeExperience( ...
        buffer, ...
        observation, ...
        nextObservation, ...
        action, ...
        correctedAction, ...
        reward, ...
        done, ...
        correctionAmplitude, ...
        interventionFlag, ...
        logProbability, ...
        value);

end

lastValue = 0.0;

buffer = calculateAdvantage( ...
    buffer, ...
    lastValue, ...
    pRL);

actorLearnablesBefore = ...
    actorNetwork.Learnables.Value;

criticLearnablesBefore = ...
    criticNetwork.Learnables.Value;

optimizerState = [];

[actorNetwork, criticNetwork, ...
    optimizerState, lossInfo] = ...
    updateCBFGPPO( ...
    actorNetwork, ...
    criticNetwork, ...
    buffer, ...
    optimizerState, ...
    pRL);

assert(optimizerState.iteration == 1);

assert(isfinite(lossInfo.actorLoss));
assert(isfinite(lossInfo.ppoLoss));
assert(isfinite(lossInfo.bcLoss));
assert(isfinite(lossInfo.criticLoss));

actorChanged = false;

for i = 1:height(actorNetwork.Learnables)

    before = actorLearnablesBefore{i};
    after = actorNetwork.Learnables.Value{i};

    if any( ...
        abs(extractdata(after - before)) > 0, ...
        'all')

        actorChanged = true;
        break;
    end

end

criticChanged = false;

for i = 1:height(criticNetwork.Learnables)

    before = criticLearnablesBefore{i};
    after = criticNetwork.Learnables.Value{i};

    if any( ...
        abs(extractdata(after - before)) > 0, ...
        'all')

        criticChanged = true;
        break;
    end

end

assert(actorChanged);
assert(criticChanged);

disp("===== CBFG-PPO update information =====");

disp("Actor loss =");
disp(lossInfo.actorLoss);

disp("PPO loss =");
disp(lossInfo.ppoLoss);

disp("BC loss =");
disp(lossInfo.bcLoss);

disp("Critic loss =");
disp(lossInfo.criticLoss);

disp("Optimizer iteration =");
disp(optimizerState.iteration);

disp("Actor parameters updated.");
disp("Critic parameters updated.");

disp("CBFG-PPO update test passed.");