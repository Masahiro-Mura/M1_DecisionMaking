clear;
clc;

projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(projectRoot));

pEnv = environmentParameters();
pCBF = cbfParameters();

state = resetEnvironment(pEnv);

%% Normal case
xEgo = state.xEgo;
xSur = state.xSur;
road = state.road;
stepCount = 0;

[isDone, isSuccess, isFail] = checkTermination( ...
    xEgo, xSur, stepCount, road, pEnv, pCBF);

assert(~isDone);
assert(~isSuccess);
assert(~isFail);

%% Collision case
xEgoCollision = xEgo;
xSurCollision = xSur;
xSurCollision(1:2) = xEgoCollision(1:2);

[isDone, isSuccess, isFail] = checkTermination( ...
    xEgoCollision, xSurCollision, stepCount, road, pEnv, pCBF);

assert(isDone);
assert(~isSuccess);
assert(isFail);

%% Road departure case
xEgoRoadDeparture = xEgo;
xEgoRoadDeparture(2) = road.yMax + 1.0;

[isDone, isSuccess, isFail] = checkTermination( ...
    xEgoRoadDeparture, xSur, stepCount, road, pEnv, pCBF);

assert(isDone);
assert(~isSuccess);
assert(isFail);

%% Time limit case
stepCountTimeLimit = pEnv.maxSteps;

[isDone, isSuccess, isFail] = checkTermination( ...
    xEgo, xSur, stepCountTimeLimit, road, pEnv, pCBF);

assert(isDone);
assert(~isSuccess);
assert(isFail);

%% Success case
xEgoSuccess = xEgo;

xEgoSuccess(1) = road.targetX;
xEgoSuccess(2) = road.targetLaneCenterY;

xSurSuccess = xSur;
xSurSuccess(1) = xEgoSuccess(1) + 50.0;

[isDone, isSuccess, isFail] = checkTermination( ...
    xEgoSuccess, xSurSuccess, stepCount, road, pEnv, pCBF);

assert(isDone);
assert(isSuccess);
assert(~isFail);

disp("Termination test passed.");