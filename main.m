clear;
N = [50 64 82 100 128 150 200 256];
smooth = [1, 5, 10];
% smooth = [1];

lsm = length(smooth);
algs = cell(lsm+1, 1); algs{1} = 'APG';
for i = 1 : lsm
    algs{i+1} = sprintf('MG-%d', smooth(i));
end
choice = [4];
infObj = 0;
tol = 1e-15;
verbose = 1;
linespec = ['o', '+', '*', '.', 'x'];
options = optimoptions('quadprog', 'Display', 'off', 'Algorithm', 'interior-point-convex', ...
    'MaxIterations', 10, 'OptimalityTolerance', tol, 'StepTolerance', tol*0.01, 'LinearSolver', 'sparse');
for k = choice
    n = N(k);
    h = 1 / (n + 1);
    Q0 = gallery('poisson', n) / h^2;
    L0 = 8 / h^2;
%     L0 = svds(Q0,1);
    mu0 = svds(Q0, 1, 'smallest');
    conv_fact = 1 - mu0 / L0;
    phi = max(sin((1 : n)*3*pi/n), 0);
    phi = vec(phi' * phi);
    p0 = Q0 * phi;
    x_ini = rand(n^2, 1);
%     L = floor(2*log2(n)) - 1;
    
    x = cell(1+lsm, 1);
    hist = cell(1+lsm, 1);
    disp('APG');
    tic;
    [x{1}, hist{1}] = apg(Q0, p0, L0, x_ini, tol, verbose);
    toc;
    infObj = min(infObj, min(hist{1}.F));
    
    levels = [1];
    llv = length(levels);
    for i = 1 : lsm
        s = smooth(i);
        for j = 1 : llv
            L = levels(j);
            fprintf('\nMGProx-%d with level %d\n', s, L);
            t0 = tic;
            [x{i*j+1}, hist{i*j+1}] = mgproxL(Q0, p0, L0, x_ini, tol, L, s, verbose);
            toc(t0);
            infObj = min(infObj, min(hist{i*j+1}.F));
        end
    end
end
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

% tic;
% cvx_begin quiet
%     variable y(N^2)
%     minimize (0.5*quad_form(y,Q0) + p0'*y)
%     subject to
%         y >= 0;
% cvx_end
% toc

% function f = obj(q, p, x)
%     f = 0.5*x'*q*x + p'*x;
% end