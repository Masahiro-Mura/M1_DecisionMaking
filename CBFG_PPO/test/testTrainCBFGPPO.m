clear;
clc;

projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(projectRoot));

pEnv = environmentParameters();
pMPC = mpcParameters();
pCBF = cbfParameters();
pIDM = idmParameters();
pRL = rlParameters();

pRL.totalTimesteps = 10;
pRL.numEpoch = 2;
pRL.miniBatchSize = 4;
pRL.bufferSize = 16;

pRL.interventionThreshold = inf;

[actorNetwork, criticNetwork, trainingResult] = ...
    trainCBFGPPO( ...
    pEnv, ...
    pMPC, ...
    pCBF, ...
    pIDM, ...
    pRL);

assert(isa(actorNetwork, 'dlnetwork'));
assert(isa(criticNetwork, 'dlnetwork'));

assert(trainingResult.numEpisodes >= 1);

assert( ...
    trainingResult.totalTimesteps(end) ...
    == pRL.totalTimesteps);

assert( ...
    length(trainingResult.episodeReward) ...
    == trainingResult.numEpisodes);

assert( ...
    length(trainingResult.episodeSteps) ...
    == trainingResult.numEpisodes);

assert(all(isfinite( ...
    trainingResult.episodeReward)));

assert(all(isfinite( ...
    trainingResult.meanActorLoss)));

assert(all(isfinite( ...
    trainingResult.meanPPOLoss)));

assert(all(isfinite( ...
    trainingResult.meanBCLoss)));

assert(all(isfinite( ...
    trainingResult.meanCriticLoss)));

assert(isfield( ...
    trainingResult, ...
    'optimizerState'));

assert( ...
    trainingResult.optimizerState.iteration > 0);

disp("===== CBFG-PPO training information =====");

disp("Number of episodes =");
disp(trainingResult.numEpisodes);

disp("Total timesteps =");
disp(trainingResult.totalTimesteps(end));

disp("Episode rewards =");
disp(trainingResult.episodeReward);

disp("Mean actor loss =");
disp(trainingResult.meanActorLoss);

disp("Mean PPO loss =");
disp(trainingResult.meanPPOLoss);

disp("Mean BC loss =");
disp(trainingResult.meanBCLoss);

disp("Mean critic loss =");
disp(trainingResult.meanCriticLoss);

disp("Optimizer iteration =");
disp(trainingResult.optimizerState.iteration);

disp("CBFG-PPO training loop test passed.");