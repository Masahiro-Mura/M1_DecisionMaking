function [c, ceq] = cbfConstraints( ...
    deltaU, xEgo, xSur, xSurNext, uMPC, road, pMPC, pCBF)

uSafe = uMPC + deltaU;

xEgoNext = vehicleModelDiscrete(xEgo, uSafe, pMPC);

h0 = calculateVehicleCBF(xEgo, xSur, pCBF);
h0Next = calculateVehicleCBF(xEgoNext, xSurNext, pCBF);

[h1, h2] = calculateRoadCBF(xEgo, road);
[h1Next, h2Next] = calculateRoadCBF(xEgoNext, road);

gamma = pCBF.gamma;

c = -(h0Next - (1 - gamma) * h0);

if road.lcf == 1
    c(end + 1, 1) = -(h1Next - (1 - gamma) * h1);
end

if road.rcf == 1
    c(end + 1, 1) = -(h2Next - (1 - gamma) * h2);
end

ceq = [];

end