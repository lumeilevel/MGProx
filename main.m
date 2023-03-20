clear;
N = [50, 100, 150, 200];
smooth = [1, 5, 10, 15];
lsm = length(smooth);
algs = cell(lsm+1, 1); algs{1} = 'APG';
for i = 1 : lsm
    algs{i+1} = sprintf('MG-%d', smooth(i));
end
choice = [1 2 3];
infObj = 0;
linespec = ['o', '+', '*', '.', 'x'];
for k = choice
    n = N(k);
    h = 1 / (n + 1);
    Q0 = gallery('poisson', n) / h^2;
    L0 = 8 / h^2;
    % L0 = svds(Q0,1);
    phi = max(sin((1 : n)*3*pi/n), 0);
    phi = vec(phi' * phi);
    p0 = Q0 * phi;
    x_ini = rand(n^2, 1);
    L = floor(2*log2(n)) - 1;
    tol = 1e-6;
    
    x = cell(1+lsm, 1);
    hist = cell(1+lsm, 1);
    disp('APG');
    tic;
    [x{1}, hist{1}] = apg(Q0, p0, L0, x_ini, tol);
    toc
    infObj = min(infObj, min(hist{1}.F));
    
    for i = 1 : lsm
        s = smooth(i);
        fprintf('MGProx-%d\n', s);
        tic;
        [x{i+1}, hist{i+1}] = mgprox(Q0, p0, L0, x_ini, tol, L, s);
        toc
        infObj = min(infObj, min(hist{i+1}.F));
    end
end

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