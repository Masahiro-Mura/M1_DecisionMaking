function p = rlParameters()

% Observation
p.observationDimension = 27;

% Action
p.actionDimension = 2;
p.actionMin = [-1; -1];
p.actionMax = [ 1;  1];

% Reference action range
p.referenceVelocityMin = 0.0;
p.referenceVelocityMax = 20.0;

p.referenceHeadingMin = deg2rad(-30);
p.referenceHeadingMax = deg2rad(30);

% PPO
p.discountFactor = 0.99;
p.learningRate = 0.0006;
p.numEpoch = 40;
p.bufferSize = 128 * 24;
p.miniBatchSize = 128;
p.totalTimesteps = 150000;

p.clipFactor = 0.2;

p.actorLossCoefficient = 1.0;
p.criticLossCoefficient = 0.5;

% Network
p.actorHiddenUnits = [256 256];
p.criticHiddenUnits = [256 256];

% CBF intervention
p.interventionThreshold = 1e-3;

% Reward
p.laneDeviationWeight = 4.0;
p.mergeDistanceScale = 40.0;
p.maxVelocity = 20.0;

% Observation normalization
p.observationMin = [
    -20; -20;
      0; -10;
     -5;
      0;
      0;
    repmat(-20, 8, 1);
    repmat(-100, 8, 1);
    repmat(-5, 4, 1)
];

p.observationMax = [
     20; 20;
   1000; 10;
      5;
      1;
      1;
    repmat(20, 8, 1);
    repmat(100, 8, 1);
    repmat(3, 4, 1)
];

end