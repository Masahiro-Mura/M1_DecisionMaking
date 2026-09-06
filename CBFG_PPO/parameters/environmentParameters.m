function p = environmentParameters()

% Simulation
p.Ts = 0.2;   % [s]

% Episode
p.maxEpisodeTime5 = 40.0;    % [s]
p.maxEpisodeTime10 = 20.0;   % [s]

p.maxEpisodeTime = p.maxEpisodeTime5;
p.maxSteps = round(p.maxEpisodeTime / p.Ts);

% Road
p.roadLength = 1000.0;       % [m]
p.laneWidth = 3.6;           % [m] temporary value

p.yMin = 0.0;                % [m] temporary value
p.yMax = 7.2;                % [m] temporary value

% Target
p.targetX = 950.0;           % [m] temporary value
p.targetLaneCenterY = 1.8;   % [m] temporary value
p.mergeTolerance = 0.5;      % [m] temporary value

end