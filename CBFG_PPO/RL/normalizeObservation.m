function observationNormalized = normalizeObservation(observation, pRL)

observationNormalized = observation;

dynamicIndex = [
    1:5, ...
    8:27
];

observationNormalized(dynamicIndex) = 2 * ...
    (observation(dynamicIndex) - pRL.observationMin(dynamicIndex)) ./ ...
    (pRL.observationMax(dynamicIndex) - pRL.observationMin(dynamicIndex)) - 1;

observationNormalized(dynamicIndex) = max( ...
    min(observationNormalized(dynamicIndex), 1), -1);

end