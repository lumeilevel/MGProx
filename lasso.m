clear;
% N = [50 64 82 100 128 150 200 256];
N = [128, 256, 512, 1024, 2048, 4096, 8142, 10000];
smooth = [1];

lsm = length(smooth);
algs = cell(lsm+1, 1); algs{1} = 'APG';
for i = 1 : lsm
    algs{i+1} = sprintf('MG-%d', smooth(i));
end
choice = [3];
infObj = 1e5;
tol = 1e-12;
verbose = 1;
linespec = ['o', '+', '*', '.', 'x'];
% options = optimoptions('quadprog', 'Display', 'off', 'Algorithm', 'interior-point-convex', ...
%     'MaxIterations', 10, 'OptimalityTolerance', tol, 'StepTolerance', tol*0.01, 'LinearSolver', 'sparse');
for k = choice
    n = N(k);
    A = randn(n, n);
%     A = A'*A;
    L0 = norm(A'*A);
%     L0 = svds(Q0,1);
    mu0 = svds(A, 1, 'smallest');
    conv_fact = 1 - mu0 / L0;
    b = randn(n, 1) / sqrt(n);
    c = zeros(n, 1);
%     c = randn(n, 1) / sqrt(n);
    lambda = 1000 / n;
    x_ini = rand(n, 1);
    
    x = cell(1+lsm, 1);
    hist = cell(1+lsm, 1);
    disp('APG');
    tic;
    [x{1}, hist{1}] = apg_lasso(A, b, c, lambda, L0, x_ini, tol, verbose);
    toc;
    infObj = min(infObj, min(hist{1}.F));
    
    levels = [1:8];
    llv = length(levels);
    for i = 1 : lsm
        s = smooth(i);
        for j = 1 : llv
            L = levels(j);
            fprintf('\nMGProx-%d with level %d\n', s, L);
            t0 = tic;
            [x{i*j+1}, hist{i*j+1}] = mgprox_lasso(A, b, c, lambda, L0, x_ini, tol, L, s, verbose);
            toc(t0);
            infObj = min(infObj, min(hist{i*j+1}.F));
        end
    end
end
tic;
cvx_begin quiet
    variable y(n)
    minimize (0.5*(A*y-b)'*(A*y-b) + c'*y + lambda*norm(y,1))
cvx_end
toc;

return;

figure(1);
for i = 1 : lsm + 1
    plot((hist{i}.F-infObj)/hist{i}.F(1), linespec(i));
    hold on;
end
hold off;
title('Function value');
xlabel('Iteration k');
ylabel('$(F_k-F_{\mbox{min}})/F_{\mbox{ini}}$', 'interpreter', 'latex');
legend(algs);
set(gca, 'YScale', 'log');

figure(2);
for i = 1 : lsm + 1
    plot(hist{i}.G/hist{i}.G(1), linespec(i));
    hold on;
end
hold off;
title('Norm of the proximal gradient mapping');
xlabel('Iteration k');
ylabel('$\Vert G_k \Vert/\Vert G_{\mbox{ini}} \Vert$', 'interpreter', 'latex');
legend(algs);
set(gca, 'YScale', 'log');

% function f = obj(q, p, x)
%     f = 0.5*x'*q*x + p'*x;
% end