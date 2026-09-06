function J = mpcCost(U, x0, zRef, p)

N = p.predictionHorizon;

U = reshape(U, 2, N);

xPred = x0;
J = 0;

for k = 1:N

    z = xPred(3:4);
    u = U(:, k);

    e = zRef - z;

    J = J + e' * p.Q * e + u' * p.R * u;

    xPred = vehicleModelDiscrete(xPred, u, p);

end

end