function p = mpcParameters()

% Simulation
p.Ts = 0.2;              % [s]

% Vehicle
p.wheelBase = 2.7;       % [m] temporary value

% MPC
p.predictionHorizon = 10; % temporary value

% Cost weights
p.Q = diag([1.0, 1.0]);
p.R = diag([0.1, 0.1]);

% Input constraints
p.accelerationMin = -5.0;            % [m/s^2]
p.accelerationMax =  3.0;            % [m/s^2]
p.steeringMin     = deg2rad(-30);    % [rad]
p.steeringMax     = deg2rad(30);     % [rad]

p.uMin = [
    p.accelerationMin;
    p.steeringMin
];

p.uMax = [
    p.accelerationMax;
    p.steeringMax
];

end