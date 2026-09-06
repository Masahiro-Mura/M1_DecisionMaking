function l = ellipseDistance(position, heading, otherPosition, a, b)

dx = otherPosition(1) - position(1);
dy = otherPosition(2) - position(2);

lineAngle = atan2(dy, dx);

delta = lineAngle - heading;

l = (a * b) / sqrt( ...
    b^2 * cos(delta)^2 + ...
    a^2 * sin(delta)^2);

end