function state = resetEnvironment(pEnv, pIDM)

state.xEgo = [
    10.0;
    1.8;
    5.0;
    0.0
];

state.xSur = [
    80.0;
    1.8;
    10.0;
    0.0
];

state.aSur = 0.0;

state.road.xPrev = state.xEgo(1);
state.road.yPrev = state.xEgo(2);
state.road.dCL = 0.0;
state.road.slope = 0.0;

state.road.laneWidth = pEnv.laneWidth;
state.road.yMin = pEnv.yMin;
state.road.yMax = pEnv.yMax;

state.road.targetX = pEnv.targetX;
state.road.targetLaneCenterY = pEnv.targetLaneCenterY;
state.road.mergeTolerance = pEnv.mergeTolerance;

state.road.lcf = 0;
state.road.rcf = 0;

state.thetaRefPrev = 0.0;

state.stepCount = 0;

state.time = 0.0;

end