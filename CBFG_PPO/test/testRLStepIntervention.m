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
assert(isfinite(reward));

assert(isfield(loggedSignals, 'state'));
assert(isfield(loggedSignals, 'rlAction'));
assert(isfield(loggedSignals, 'referenceAction'));
assert(isfield(loggedSignals, 'uMPC'));
assert(isfield(loggedSignals, 'uSafe'));
assert(isfield(loggedSignals, 'deltaU'));
assert(isfield(loggedSignals, 'correctionAmplitude'));
assert(isfield(loggedSignals, 'interventionFlag'));

assert(isequal(size(loggedSignals.rlAction), [2 1]));
assert(isequal(size(loggedSignals.referenceAction), [2 1]));
assert(isequal(size(loggedSignals.uMPC), [2 1]));
assert(isequal(size(loggedSignals.uSafe), [2 1]));
assert(isequal(size(loggedSignals.deltaU), [2 1]));

assert(all(isfinite(loggedSignals.rlAction)));
assert(all(isfinite(loggedSignals.referenceAction)));
assert(all(isfinite(loggedSignals.uMPC)));
assert(all(isfinite(loggedSignals.uSafe)));
assert(all(isfinite(loggedSignals.deltaU)));

assert(isscalar(loggedSignals.correctionAmplitude));
assert(isfinite(loggedSignals.correctionAmplitude));

expectedAmplitude = norm(loggedSignals.deltaU);

assert(abs( ...
    loggedSignals.correctionAmplitude ...
    - expectedAmplitude) < 1e-10);

expectedIntervention = ...
    loggedSignals.correctionAmplitude ...
    > pRL.interventionThreshold;

assert(loggedSignals.interventionFlag == expectedIntervention);

assert(norm( ...
    loggedSignals.uSafe ...
    - (loggedSignals.uMPC + loggedSignals.deltaU)) < 1e-10);

disp("===== RL step intervention information =====");

disp("RL action =");
disp(loggedSignals.rlAction);

disp("Reference action =");
disp(loggedSignals.referenceAction);

disp("uMPC =");
disp(loggedSignals.uMPC);

disp("deltaU =");
disp(loggedSignals.deltaU);

disp("uSafe =");
disp(loggedSignals.uSafe);

disp("Correction amplitude =");
disp(loggedSignals.correctionAmplitude);

disp("Intervention flag =");
disp(loggedSignals.interventionFlag);

disp("RL step intervention test passed.");