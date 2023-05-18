function [xk, hist] = mgprox_lasso(A0, b0, c0, lambda, L0, x_ini, tol, level, smooth, verbose)
    xk0 = x_ini; xk = xk0;
    [n, ~] = size(A0);
    max_iter = n*1e2;
    crp = 2;
%     t = 1;  t0 = 1;
    objold = 0.5*((A0*xk0-b0)'*(A0*xk0-b0)) + c0'*xk0 + lambda*norm(xk0,1);
    hist.time = 0;
    hist.F = zeros(max_iter, 1);
    hist.G = zeros(max_iter, 1);
    hist.dist = zeros(max_iter, 1);
    hist.relDist = zeros(max_iter, 1);
    hist.relObjdiff = zeros(max_iter, 1);
    % prepare the full version of non-adaptive \bar{R}_{l->l+1}
    Rbar = cell(level, 1);              % restriction matrix
    A = cell(level+1, 1); A{1} = A0;    % A_{l}
    L = [L0; zeros(level, 1)];          % L_{l}, Lipschitz constant
    c = cell(level+1, 1); c{1} = c0;
    for l = 1 : level
        np = floor(n/2);
        Rbar{l} = sparse([1:np,1:np,2:np], [2*(1:np)-1,2:2:n,2:2:2*np-2], [2*ones(np,1);1*ones(2*np-1,1)]);
        if mod(n, 2) == 1
            Rbar{l} = [Rbar{l}, sparse(np, 1)];
        end
        n = np;
        A{l+1} = crp * A{l} * Rbar{l}';
        c{l+1} = Rbar{l} * c{l};
        L(l+1) = norm(A{l+1}'*A{l+1});
    end
%     Q_inv = Q{L+1}^(-1);
    for iter = 1 : max_iter
        R = Rbar;                           % R_{l->l+1}
        tau = cell(level+1, 1); tau{1} = 0; % tau_{l->l+1}^{k+1}
        x = cell(level+1, 1); x{1} = xk;    % x_{l+1}^k
        y = cell(level, 1);                 % y_{l}^k        
        Axb = A0*xk-b0;
        grad = A0'*Axb + c0;
        hist.dist(iter) = norm(grad);
        obj = 0.5*(Axb'*Axb) + c0'*xk + lambda*norm(xk,1);
        hist.F(iter) = obj;
        hist.G(iter) = L0*norm(xk-prox_l1(xk-grad/L0,lambda/L0));
        hist.relDist(iter) = norm(xk-xk0) / norm(xk);
        hist.relObjdiff(iter) = abs(obj - objold) / max(obj, 1);

        % stopping criterion
        if hist.G(iter) / hist.G(1) <= tol
            hist.F = hist.F(1:iter);
            hist.G = hist.G(1:iter);
            hist.dist = hist.dist(1:iter);
            hist.relDist = hist.relDist(1:iter);
            hist.relObjdiff = hist.relObjdiff(1:iter);
            if verbose
                fprintf('\n MGProx early stopping--iteration: %d\n', iter);
                fprintf('[c] proximal first-order optimality condition satisfied\n')
            end
            break
        end
        if iter > 4
            if max(hist.relDist(iter), 0.1*hist.relObjdiff(iter)) < tol
                if verbose
                    fprintf("\n MGProx Early Stopping--iteration: %d\n", iter);
                    fprintf('[a] relDist < %3.2e\n', tol);
                    fprintf("norm(X-Xold,'fro')/norm(X,'fro') = %f\n", hist.relDist(iter));
                end
                hist.F = hist.F(1:iter);
                hist.G = hist.G(1:iter);
                hist.dist = hist.dist(1:iter);
                hist.relDist = hist.relDist(1:iter);
                hist.relObjdiff = hist.relObjdiff(1:iter);
                break
            end
            if max(0.5*hist.relDist(iter), 100*hist.relObjdiff(iter)) < tol
                if verbose
                    fprintf("\n MGProx Early Stopping--iteration: %d\n", iter);
                    fprintf('[b] relObjdiff < %3.2e\n', 0.01*tol);
                end
                hist.F = hist.F(1:iter);
                hist.G = hist.G(1:iter);
                hist.dist = hist.dist(1:iter);
                hist.relDist = hist.relDist(1:iter);
                hist.relObjdiff = hist.relObjdiff(1:iter);
                break;
            end
        end
        for l = 1 : level
            % pre-smoothing
            y{l} = x{l};
            for sm = 1 : smooth
                y{l} = prox_l1(y{l}-(A{l}'*(A{l}*y{l}-b0)+c{l}-tau{l})/L(l), lambda/L(l));
            end
            % generate the adaptive restriction operator
            R{l}(:,~y{l}) = 0;
            x{l+1} = R{l} * y{l};
            % create the tau vector
            tau{l+1} = A{l+1}'*(A{l+1}*x{l+1}-b0)+c{l+1} + sign(x{l+1})-R{l}*(A{l}'*(A{l}*y{l}-b0)+c{l}+sign(y{l}));
        end
        
        % Solve level-L coarse problem       
        tic;
        w = A{level+1}'*A{level+1} \ (A{level+1}'*b0+tau{level+1});
        hist.time = hist.time + toc;
%         w = zeros(n, 1);
%                 options = optimoptions('quadprog', 'Display', 'off', 'Algorithm', 'interior-point-convex', ...
%     'MaxIterations', 10, 'OptimalityTolerance', eps, 'StepTolerance', eps*0.01, 'LinearSolver', 'sparse');
%                 w = quadprog(Q{L+1}, b, [],[],[],[],zeros(n,1),[],[], options);
                [w, ~] = apg_lasso(A{level+1}, b0, c{level+1}-tau{level+1}, lambda, L(level+1), w, 1e-2, 0);
%                 [w, ~] = mgproxL(Q{L+1}, b, Ll, w, eps, L, 20, options, 0);
%                 [w, ~] = mgprox(Q{L+1}, b, Ll, w, eps*1e4, floor(2*log2(n)) - 1, smooth);
%             w(w < eps) = 0;

        

        for l = level : -1 : 1
            % coarse correction with line search
            alpha = 0.5;
            grad = A{l}'*(A{l}*y{l}-b0) + c{l};
            gy = lambda * norm(y{l},1);
            while 1
                cor = alpha*crp*R{l}'*(w-x{l+1});
                Ax = A{l}*cor;
                if 0.5*(Ax'*Ax) + grad'*cor + lambda*norm(y{l}+cor,1) <= gy
                    z = y{l} + cor;
                    break;
                elseif alpha > tol
                    alpha = alpha / 2;
                else
                    z = y{l};
                    break;
                end
            end
            % post-smoothing
            w = z;
            for sm = 1 : smooth
                w = prox_l1(w-(A{l}'*(A{l}*w-b0)+c{l}-tau{l})/L(l), lambda/L(l));
            end
        end
        % update the fine variable
        xk0 = xk;
        xk = w;
%         t0 = t; t = 0.5*(1+sqrt(1+4*t^2));
        objold = obj;
    end
end