clear;
clc;

projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(projectRoot));

pEnv = environmentParameters();
pMPC = mpcParameters();
pCBF = cbfParameters();
pIDM = idmParameters();
pRL = rlParameters();

[observation, loggedSignals] = rlReset( ...
    pEnv, pRL);

numTestSteps = 50;

for k = 1:numTestSteps

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
    assert(isfinite(reward));

    state = loggedSignals.state;

    assert(isequal(size(state.xEgo), [4 1]));
    assert(isequal(size(state.xSur), [4 1]));
    assert(all(isfinite(state.xEgo)));
    assert(all(isfinite(state.xSur)));

    assert(state.stepCount == k);
    assert(abs(state.time - k * pEnv.Ts) < 1e-10);

    observation = nextObservation;

    if isDone
        fprintf("Episode terminated at step %d.\n", k);
        break;
    end

end

disp("Multi-step test passed.");