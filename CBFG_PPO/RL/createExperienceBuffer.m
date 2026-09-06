function buffer = createExperienceBuffer(pRL)

N = pRL.bufferSize;

buffer.observation = ...
    zeros(pRL.observationDimension, N);

buffer.nextObservation = ...
    zeros(pRL.observationDimension, N);

buffer.action = ...
    zeros(pRL.actionDimension, N);

buffer.correctedAction = ...
    zeros(pRL.actionDimension, N);

buffer.reward = ...
    zeros(1, N);

buffer.done = ...
    false(1, N);

buffer.correctionAmplitude = ...
    zeros(1, N);

buffer.interventionFlag = ...
    false(1, N);

buffer.logProbability = ...
    zeros(1, N);

buffer.value = ...
    zeros(1, N);

buffer.advantage = ...
    zeros(1, N);

buffer.return = ...
    zeros(1, N);

buffer.count = 0;

end