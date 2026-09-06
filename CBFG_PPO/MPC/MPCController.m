function uMPC = MPCController(x, zRef, p)

N = p.predictionHorizon;

U0 = zeros(2 * N, 1);

lb = repmat(p.uMin, N, 1);
ub = repmat(p.uMax, N, 1);

objective = @(U) mpcCost(U, x, zRef, p);

options = optimoptions('fmincon', ...
    'Algorithm', 'sqp', ...
    'Display', 'none');

Uopt = fmincon(objective, U0, [], [], [], [], lb, ub, [], options);

uMPC = Uopt(1:2);

end