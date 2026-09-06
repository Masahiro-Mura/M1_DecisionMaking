clear;
clc;

projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(projectRoot));

pRL = rlParameters();

actorNetwork = createActorNetwork(pRL);
criticNetwork = createCriticNetwork(pRL);

buffer = createExperienceBuffer(pRL);

numSamples = 256;

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
    optimizerState, trainingInfo] = ...
    trainCBFGPPOEpochs( ...
    actorNetwork, ...
    criticNetwork, ...
    buffer, ...
    optimizerState, ...
    pRL);

expectedNumBatches = ...
    ceil(numSamples / pRL.miniBatchSize);

expectedTotalUpdates = ...
    pRL.numEpoch * expectedNumBatches;

assert( ...
    trainingInfo.numEpoch ...
    == pRL.numEpoch);

assert( ...
    trainingInfo.numBatches ...
    == expectedNumBatches);

assert( ...
    trainingInfo.totalUpdates ...
    == expectedTotalUpdates);

assert( ...
    optimizerState.iteration ...
    == expectedTotalUpdates);

assert(all(isfinite( ...
    trainingInfo.actorLoss)));

assert(all(isfinite( ...
    trainingInfo.ppoLoss)));

assert(all(isfinite( ...
    trainingInfo.bcLoss)));

assert(all(isfinite( ...
    trainingInfo.criticLoss)));

assert(all(isfinite( ...
    trainingInfo.meanActorLoss)));

assert(all(isfinite( ...
    trainingInfo.meanPPOLoss)));

assert(all(isfinite( ...
    trainingInfo.meanBCLoss)));

assert(all(isfinite( ...
    trainingInfo.meanCriticLoss)));

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

disp("===== CBFG-PPO epoch training information =====");

disp("Number of epochs =");
disp(trainingInfo.numEpoch);

disp("Number of batches per epoch =");
disp(trainingInfo.numBatches);

disp("Total updates =");
disp(trainingInfo.totalUpdates);

disp("Optimizer iteration =");
disp(optimizerState.iteration);

disp("Final mean actor loss =");
disp(trainingInfo.meanActorLoss(end));

disp("Final mean PPO loss =");
disp(trainingInfo.meanPPOLoss(end));

disp("Final mean BC loss =");
disp(trainingInfo.meanBCLoss(end));

disp("Final mean critic loss =");
disp(trainingInfo.meanCriticLoss(end));

disp("Actor parameters updated.");
disp("Critic parameters updated.");

disp("CBFG-PPO epoch training test passed.");