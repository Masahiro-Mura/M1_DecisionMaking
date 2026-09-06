function reward = calculateReward( ...
    xEgo, dP, dF, thetaRef, thetaRefPrev, ...
    isSuccess, isFail, road, pRL)

dc = road.dCL;

rLd = 1 / (1 + pRL.laneDeviationWeight * dc^2);

rMerge = 1 - abs( ...
    exp(-(dP / pRL.mergeDistanceScale)^0.5) ...
    - exp(-(dF / pRL.mergeDistanceScale)^0.5));

rSteer = -abs(thetaRef - thetaRefPrev);

rSpeed = xEgo(3) / pRL.maxVelocity;

if isSuccess
    rTerminal = 1;
elseif isFail
    rTerminal = -1;
else
    rTerminal = 0;
end

reward = rLd + rMerge + rSteer + rSpeed + rTerminal;

end