function [isDone, isSuccess, isFail] = ...
    checkTermination(xEgo, xSur, stepCount, road, pEnv, pCBF)

isCollision = false;

for i = 1:size(xSur, 2)

    h0 = calculateVehicleCBF( ...
        xEgo, xSur(:, i), pCBF);

    if h0 <= 0
        isCollision = true;
        break;
    end

end

isRoadDeparture = ...
    xEgo(2) < road.yMin || ...
    xEgo(2) > road.yMax;

isTimeLimit = stepCount >= pEnv.maxSteps;

isMergeComplete = ...
    xEgo(1) >= road.targetX && ...
    abs(xEgo(2) - road.targetLaneCenterY) <= road.mergeTolerance;

isSuccess = isMergeComplete && ...
    ~isCollision && ...
    ~isRoadDeparture;

isFail = isCollision || ...
    isRoadDeparture || ...
    (isTimeLimit && ~isSuccess);

isDone = isSuccess || isFail;

end