clear;
clc;

projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(projectRoot));

pMPC = mpcParameters();
pCBF = cbfParameters();
pEnv = environmentParameters();

road.xPrev = 0.0;
road.yPrev = 1.8;
road.dCL = 0.0;
road.slope = 0.0;

road.laneWidth = pEnv.laneWidth;
road.yMin = pEnv.yMin;
road.yMax = pEnv.yMax;

road.targetX = pEnv.targetX;
road.targetLaneCenterY = pEnv.targetLaneCenterY;
road.mergeTolerance = pEnv.mergeTolerance;

road.lcf = 0;
road.rcf = 0;

uMPC = [
    0.0;
    0.0
];

%% Safe case

xEgoSafe = [
    10.0;
    1.8;
    5.0;
    0.0
];

xSurSafe = [
    50.0;
    1.8;
    5.0;
    0.0
];

xSurSafeNext = vehicleModelDiscrete( ...
    xSurSafe, ...
    [0.0; 0.0], ...
    pMPC);

[uSafeSafe, deltaUSafe, exitflagSafe] = CBFController( ...
    xEgoSafe, ...
    xSurSafe, ...
    xSurSafeNext, ...
    uMPC, ...
    road, ...
    pMPC, ...
    pCBF);

[cSafeZero, ~] = cbfConstraints( ...
    [0.0; 0.0], ...
    xEgoSafe, ...
    xSurSafe, ...
    xSurSafeNext, ...
    uMPC, ...
    road, ...
    pMPC, ...
    pCBF);

disp("===== Safe case =====");
disp("deltaUSafe =");
disp(deltaUSafe);

disp("exitflagSafe =");
disp(exitflagSafe);

disp("cSafeZero =");
disp(cSafeZero);

%% Intervention case

xEgoUnsafe = [
    10.0;
    1.8;
    5.0;
    0.0
];

xSurUnsafe = [
    15.1;
    1.8;
    4.9;
    0.0
];

xSurUnsafeNext = vehicleModelDiscrete( ...
    xSurUnsafe, ...
    [0.0; 0.0], ...
    pMPC);

[uSafeUnsafe, deltaUUnsafe, exitflagUnsafe] = CBFController( ...
    xEgoUnsafe, ...
    xSurUnsafe, ...
    xSurUnsafeNext, ...
    uMPC, ...
    road, ...
    pMPC, ...
    pCBF);

[cUnsafeZero, ~] = cbfConstraints( ...
    [0.0; 0.0], ...
    xEgoUnsafe, ...
    xSurUnsafe, ...
    xSurUnsafeNext, ...
    uMPC, ...
    road, ...
    pMPC, ...
    pCBF);

disp("===== Intervention case =====");
disp("deltaUUnsafe =");
disp(deltaUUnsafe);

disp("exitflagUnsafe =");
disp(exitflagUnsafe);

disp("cUnsafeZero =");
disp(cUnsafeZero);




%% Feasibility scan

accGrid = linspace( ...
    pMPC.uMin(1) - uMPC(1), ...
    pMPC.uMax(1) - uMPC(1), ...
    41);

steerGrid = linspace( ...
    pMPC.uMin(2) - uMPC(2), ...
    pMPC.uMax(2) - uMPC(2), ...
    41);

minConstraint = inf;
bestDeltaU = [0; 0];

for i = 1:length(accGrid)

    for j = 1:length(steerGrid)

        deltaUTest = [
            accGrid(i);
            steerGrid(j)
        ];

        [cTest, ~] = cbfConstraints( ...
            deltaUTest, ...
            xEgoUnsafe, ...
            xSurUnsafe, ...
            xSurUnsafeNext, ...
            uMPC, ...
            road, ...
            pMPC, ...
            pCBF);

        maxConstraint = max(cTest);

        if maxConstraint < minConstraint

            minConstraint = maxConstraint;
            bestDeltaU = deltaUTest;

        end

    end

end

disp("===== Feasibility scan =====");

disp("minimum constraint =");
disp(minConstraint);

disp("best deltaU =");
disp(bestDeltaU);