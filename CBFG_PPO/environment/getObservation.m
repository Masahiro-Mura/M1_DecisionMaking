function observation = getObservation(xEgo, xSur, aSur, road)

egoVelocity = [
    xEgo(3) * cos(xEgo(4));
    xEgo(3) * sin(xEgo(4))
];

egoPosition = xEgo(1:2);

relativePosition = zeros(8, 1);
surVelocity = zeros(8, 1);
surAcceleration = zeros(4, 1);

numSur = min(size(xSur, 2), 4);

for i = 1:numSur

    idx = 2 * i - 1;

    relativePosition(idx:idx+1) = ...
        xSur(1:2, i) - xEgo(1:2);

    surVelocity(idx:idx+1) = [
        xSur(3, i) * cos(xSur(4, i));
        xSur(3, i) * sin(xSur(4, i))
    ];

    surAcceleration(i) = aSur(i);

end

observation = [
    egoVelocity;
    egoPosition;
    road.dCL;
    road.lcf;
    road.rcf;
    surVelocity;
    relativePosition;
    surAcceleration
];

end