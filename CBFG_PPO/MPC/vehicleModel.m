function xDot = vehicleModel(x, u, p)
% Continuous-time kinematic bicycle model
% State: x = [X; Y; v; theta]
% Input: u = [aAcc; delta]

v     = x(3);
theta = x(4);

aAcc  = u(1);
delta = u(2);

% Paper Eq. (15)
beta = atan(0.5 * tan(delta));

% Paper Eq. (14)
xDot = zeros(4, 1);

xDot(1) = v * cos(theta + beta);
xDot(2) = v * sin(theta + beta);
xDot(3) = aAcc;
xDot(4) = (v / p.wheelBase) * sin(beta);

end