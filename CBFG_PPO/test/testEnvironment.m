clear;
clc;

projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(projectRoot));

pEnv = environmentParameters();
pMPC = mpcParameters();
pCBF = cbfParameters();
pIDM = idmParameters();
pRL = rlParameters();

env = createEnvironment( ...
    pEnv, pMPC, pCBF, pIDM, pRL);

[observation, loggedSignals] = rlReset( ...
    pEnv, pRL);

assert(isequal(size(observation), [27 1]));
assert(all(isfinite(observation)));

action = zeros(2, 1);

[nextObservation, reward, isDone, loggedSignals] = ...
    rlStep( ...
    action, ...
    loggedSignals, ...
    pEnv, ...
    pMPC, ...
    pCBF, ...
    pIDM, ...
    pRL);

assert(isequal(size(nextObservation), [27 1]));
assert(all(isfinite(nextObservation)));
assert(isscalar(reward));
assert(islogical(isDone) || isnumeric(isDone));

disp("Environment test passed.");