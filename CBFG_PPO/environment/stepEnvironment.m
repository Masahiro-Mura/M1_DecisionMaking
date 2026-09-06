function [stateNext, uMPC, uSafe, deltaU] = ...
    stepEnvironment(state, zRef, pMPC, pCBF, pIDM)

xEgo = state.xEgo;
xSur = state.xSur;
road = state.road;

xLeader = [];

uMPC = MPCController(xEgo, zRef, pMPC);

xSurNext = updateSurroundingVehicle( ...
    xSur, xLeader, pIDM, pMPC.Ts);

aSurNext = (xSurNext(3) - xSur(3)) / pMPC.Ts;

[uSafe, deltaU] = CBFController( ...
    xEgo, xSur, xSurNext, ...
    uMPC, road, pMPC, pCBF);

xEgoNext = vehicleModelDiscrete( ...
    xEgo, uSafe, pMPC);

stateNext = state;

stateNext.xEgo = xEgoNext;
stateNext.xSur = xSurNext;
stateNext.aSur = aSurNext;

stateNext.thetaRefPrev = zRef(2);

stateNext.stepCount = state.stepCount + 1;
stateNext.time = state.time + pMPC.Ts;

stateNext.road.xPrev = xEgoNext(1);
stateNext.road.yPrev = xEgoNext(2);

stateNext.road.dCL = ...
    xEgoNext(2) - stateNext.road.targetLaneCenterY;

end