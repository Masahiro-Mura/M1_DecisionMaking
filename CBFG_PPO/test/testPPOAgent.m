clear;
clc;

projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(projectRoot));

pRL = rlParameters();

observationInfo = rlNumericSpec( ...
    [pRL.observationDimension 1], ...
    'LowerLimit', -ones(pRL.observationDimension, 1), ...
    'UpperLimit',  ones(pRL.observationDimension, 1));

observationInfo.Name = 'observation';

actionInfo = rlNumericSpec( ...
    [pRL.actionDimension 1], ...
    'LowerLimit', pRL.actionMin, ...
    'UpperLimit', pRL.actionMax);

actionInfo.Name = 'action';

actorNetwork = createActorNetwork(pRL);
criticNetwork = createCriticNetwork(pRL);

assert(isa(actorNetwork, 'dlnetwork'));
assert(isa(criticNetwork, 'dlnetwork'));

agent = createPPOAgent( ...
    observationInfo, ...
    actionInfo, ...
    pRL);

assert(isa(agent, 'rl.agent.rlPPOAgent'));

disp("PPO agent test passed.");