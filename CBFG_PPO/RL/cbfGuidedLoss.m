function [actorLoss, ppoLoss, bcLoss] = cbfGuidedLoss( ...
    probabilityRatio, advantage, ...
    policyAction, correctedAction, ...
    correctionAmplitude, interventionFlag, clipFactor)

normalIndex = interventionFlag == 0;
interventionIndex = interventionFlag == 1;

ppoLoss = dlarray(0.0);
bcLoss = dlarray(0.0);

if any(normalIndex)

    ratioNormal = probabilityRatio(normalIndex);
    advantageNormal = advantage(normalIndex);

    clippedRatio = min( ...
        max(ratioNormal, 1 - clipFactor), ...
        1 + clipFactor);

    surrogate1 = ratioNormal .* advantageNormal;
    surrogate2 = clippedRatio .* advantageNormal;

    ppoObjective = mean( ...
        min(surrogate1, surrogate2));

    ppoLoss = -ppoObjective;

end

if any(interventionIndex)

    policyActionIntervention = ...
        policyAction(:, interventionIndex);

    correctedActionIntervention = ...
        correctedAction(:, interventionIndex);

    correctionAmplitudeIntervention = ...
        correctionAmplitude(interventionIndex);

    weight = 1 + exp( ...
        correctionAmplitudeIntervention);

    actionError = ...
        correctedActionIntervention ...
        - policyActionIntervention;

    squaredError = sum( ...
        actionError.^2, 1);

    bcLoss = mean( ...
        weight .* squaredError);

end

actorLoss = ppoLoss + bcLoss;

end