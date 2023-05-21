clear;
rng(1);
% N = [50 64 82 100 128 150 200 256];
N = [128, 256, 512, 1024, 2048, 4096, 8142, 10000];
smooth = [1];

lsm = length(smooth);
algs = cell(lsm+1, 1); algs{1} = 'APG';
for i = 1 : lsm
    algs{i+1} = sprintf('MG-%d', smooth(i));
end
choice = [2];
infObj = 1e5;
tol = 1e-9;
verbose = 1;
% linespec = ['o', '+', '*', '.', 'x'];
% options = optimoptions('quadprog', 'Display', 'off', 'Algorithm', 'interior-point-convex', ...
%     'MaxIterations', 10, 'OptimalityTolerance', tol, 'StepTolerance', tol*0.01, 'LinearSolver', 'sparse');
for k = choice
    n = N(k);
    A = randn(n, n);
    A = A'*A / sqrt(n);
    L0 = norm(A'*A);
%     L0 = svds(Q0,1);
    mu0 = svds(A, 1, 'smallest');
    conv_fact = 1 - mu0 / L0;
    b = randn(n, 1);
    c = zeros(n, 1);
%     c = randn(n, 1) / sqrt(n);
    lambda = 1e2 / n;
%     x_ini = rand(n, 1);
    x_ini = rand(n, 1) / sqrt(n);
%     x_ini = randn(n, 1) / sqrt(n);
%     x_ini = zeros(n, 1);
%     x_ini = A'*A \ (A'*b - c);
    
    x = cell(1+lsm, 1);
    hist = cell(1+lsm, 1);
    disp('APG');
    t0 = tic;
    [x{1}, hist{1}] = apg_l0(A, b, c, lambda, L0, x_ini, tol, verbose);
    toc(t0);
    infObj = min(infObj, min(hist{1}.F));
%     return;
    levels = [1:5];
    llv = length(levels);
    for i = 1 : lsm
        s = smooth(i);
        for j = 1 : llv
            L = levels(j);
            fprintf('\nMGProx-%d with level %d\n', s, L);
            t0 = tic;
            [x{i*j+1}, hist{i*j+1}] = mgprox_l0(A, b, c, lambda, L0, x_ini, tol, L, s, verbose);
            toc(t0);
            infObj = min(infObj, min(hist{i*j+1}.F));
        end
    end
end