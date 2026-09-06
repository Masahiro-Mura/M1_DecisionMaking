function xNext = vehicleModelDiscrete(x, u, p)

v     = x(3);
theta = x(4);

aAcc  = u(1);
delta = u(2);

beta = atan(0.5 * tan(delta));

xNext = zeros(4, 1);

xNext(1) = x(1) + v * cos(theta + beta) * p.Ts;
xNext(2) = x(2) + v * sin(theta + beta) * p.Ts;
xNext(3) = v + aAcc * p.Ts;
xNext(4) = theta + (v / p.wheelBase) * sin(beta) * p.Ts;

end