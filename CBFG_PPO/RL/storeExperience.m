function buffer = storeExperience( ...
    buffer, observation, nextObservation, ...
    action, correctedAction, ...
    reward, done, ...
    correctionAmplitude, interventionFlag, ...
    logProbability, value)

index = buffer.count + 1;

if index > size(buffer.observation, 2)
    error("Experience buffer is full.");
end

buffer.observation(:, index) = observation;
buffer.nextObservation(:, index) = nextObservation;

buffer.action(:, index) = action;
buffer.correctedAction(:, index) = correctedAction;

buffer.reward(index) = reward;
buffer.done(index) = done;

buffer.correctionAmplitude(index) = ...
    correctionAmplitude;

buffer.interventionFlag(index) = ...
    interventionFlag;

buffer.logProbability(index) = ...
    logProbability;

buffer.value(index) = value;

buffer.count = index;

end