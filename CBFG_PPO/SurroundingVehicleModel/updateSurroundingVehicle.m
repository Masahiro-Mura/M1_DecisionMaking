function xNext = updateSurroundingVehicle(x, xLeader, pIDM, Ts)

v = x(3);

if isempty(xLeader)

    a = pIDM.maxAcceleration * ...
        (1 - (v / pIDM.desiredVelocity)^4);

else

    vLeader = xLeader(3);

    gap = xLeader(1) - x(1) - pIDM.vehicleLength;
    gap = max(gap, pIDM.minimumGap);

    a = IDM(v, vLeader, gap, pIDM);

end

xNext = x;

xNext(1) = x(1) + v * Ts;
xNext(3) = max(0, v + a * Ts);

end