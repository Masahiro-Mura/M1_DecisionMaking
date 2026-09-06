function [h1, h2] = calculateRoadCBF(xEgo, road)

x = xEgo(1);
y = xEgo(2);

h1 = y - road.yPrev + road.dCL ...
    + (x - road.xPrev) * road.slope ...
    - road.laneWidth / 2;

h2 = -y + road.yPrev - road.dCL ...
    - (x - road.xPrev) * road.slope ...
    + road.laneWidth / 2;

end