function [xk, hist] = mgproxL(Q0, p0, L0, x_ini, eps, L, smooth, options)
    xk0 = x_ini; xk = xk0;
    [n, ~] = size(Q0);
    max_iter = n;
    c = 2;
%     t = 1;  t0 = 1;
    objold = 0.5*xk0'*Q0*xk0 + p0'*xk0;
    hist.time = 0;
    hist.F = zeros(max_iter, 1);
    hist.G = zeros(max_iter, 1);
    hist.dist = zeros(max_iter, 1);
    hist.relDist = zeros(max_iter, 1);
    hist.relObjdiff = zeros(max_iter, 1);
    % prepare the full version of non-adaptive \bar{R}_{l->l+1}
    Rbar = cell(L, 1);  % restriction matrix
    Q = cell(L+1, 1); Q{1} = Q0;     % Q_{l}
    p = cell(L+1, 1); p{1} = p0;     % p_{l}
    for l = 1 : L
        np = floor(n/2);
        Rbar{l} = sparse([1:np,1:np,2:np], [2*(1:np)-1,2:2:n,2:2:2*np-2], [2*ones(np,1);1*ones(2*np-1,1)]);
        if mod(n, 2) == 1
            Rbar{l} = [Rbar{l}, sparse(np, 1)];
        end
        n = np;
        Q{l+1} = c * Rbar{l} * Q{l} * Rbar{l}';
        p{l+1} = Rbar{l} * p{l};
    end
    Q_inv = Q{L+1}^(-1);
    for iter = 1 : max_iter
        R = Rbar;                       % R_{l->l+1}
        tau = cell(L+1, 1); tau{1} = 0; % tau_{l->l+1}^{k+1}
        x = cell(L+1, 1); x{1} = xk;    % x_{l+1}^k
        y = cell(L, 1);                 % y_{l}^k        
        df = Q0*xk + p0;
        obj = 0.5*xk'*(df+p0);
        hist.F(iter) = obj;
        hist.G(iter) = L0*norm(xk-max(0,xk-1/L0*df));
        hist.relDist(iter) = norm(xk-xk0) / norm(xk);
        hist.relObjdiff(iter) = abs(obj - objold) / max(obj, 1);
        hist.dist(iter) = norm(df);
        % stopping criterion
        if hist.G(iter) / hist.G(1) <= eps
            hist.F = hist.F(1:iter);
            hist.G = hist.G(1:iter);
            hist.dist = hist.dist(1:iter);
            hist.relDist = hist.relDist(1:iter);
            hist.relObjdiff = hist.relObjdiff(1:iter);
            fprintf('\n MGProx early stopping--iteration: %d\n', iter);
            fprintf('[c] proximal first-order optimality condition satisfied\n')
            break
        end
        if iter > 4
            if max(hist.relDist(iter), 0.1*hist.relObjdiff(iter)) < eps
                fprintf("\n APG Early Stopping--iteration: %d\n", iter);
                fprintf('[a] relDist < %3.2e', eps);
                fprintf("norm(X-Xold,'fro')/norm(X,'fro') = %f\n", hist.relDist(iter));
                hist.F = hist.F(1:iter);
                hist.G = hist.G(1:iter);
                hist.dist = hist.dist(1:iter);
                hist.relDist = hist.relDist(1:iter);
                hist.relObjdiff = hist.relObjdiff(1:iter);
                break
            end
            if max(0.5*hist.relDist(iter), 100*hist.relObjdiff(iter)) < eps
                fprintf("\n APG Early Stopping--iteration: %d\n", iter);
                fprintf('[b] relObjdiff < %3.2e', 0.01*eps);
                hist.F = hist.F(1:iter);
                hist.G = hist.G(1:iter);
                hist.dist = hist.dist(iter);
                hist.relDist = hist.relDist(iter);
                hist.relObjdiff = hist.relObjdiff(iter);
                break;
            end
        end
        for l = 1 : L
            % pre-smoothing
            y{l} = x{l};
            for sm = 1 : smooth
                y{l} = max(0, y{l}-1/L0*(Q{l}*y{l}+p{l}-tau{l}));
            end
            % generate the adaptive restriction operator
            R{l}(:,~y{l}) = 0;
            x{l+1} = R{l} * y{l};
            % create the tau vector
            tau{l+1} = Q{l+1}*x{l+1}+p{l+1}-R{l}*(Q{l}*y{l}+p{l});
        end
        
        % Solve level-L coarse problem
        tic;
        b = p{L+1}-tau{L+1};
        w = -Q_inv*b;
        w(w < eps) = 0;
        if any(w < -eps)
            if n == 2
                w = [0;0];
                if b(1) <= 0 && b(1)*Q{L+1}(2,1) <= b(2)*Q{L+1}(1,1)
                    w(1) = -b(1) / Q{L+1}(1,1);
                    w(2) = 0;
                else
                    w(2) = -b(2) / Q{L+1}(2,2);
                    w(1) = 0;
                end
            elseif n == 3
                w = [0;0;0];
                if b(1)<=0 && b(1)*Q{L+1}(2,1)<=b(2)*Q{L+1}(1,1) && b(1)*Q{L+1}(3,1)<=b(3)*Q{L+1}(1,1)
                    w(1) = -b(1) / Q{L+1}(1,1);
                elseif b(2)<=0 && b(2)*Q{L+1}(1,2)<=b(1)*Q{L+1}(2,2) && b(2)*Q{L+1}(3,2)<=b(3)*Q{L+1}(2,2)
                    w(2) = -b(2) / Q{L+1}(2,2);
                elseif b(3)<=0 && b(3)*Q{L+1}(1,3)<=b(1)*Q{L+1}(3,3) && b(3)*Q{L+1}(2,3)<=b(2)*Q{L+1}(3,3)
                    w(3) = -b(3) / Q{L+1}(3,3);
                else
                    w1 = Q{L+1}(1:2,1:2) \ b(1:2);
                    w2 = Q{L+1}([1 3],[1 3]) \ b([1 3]);
                    w3 = Q{L+1}(2:3,2:3) \ b(2:3);
                    if Q{L+1}(3,1:2)'*w1+b(3)>=0 && w1>=0
                        w=[w1;0];
                    elseif Q{L+1}(2,[1 3])'*w2+b(2)>=0 && w2>=0
                        w=[w2(1);0;w2(2)];
                    else
                        w=[0;w3];
                    end
                end
            else
                w = quadprog(Q{L+1}, b, [],[],[],[],zeros(n,1),[],[], options);
            end
            w(w < eps) = 0;
        end
        hist.time = hist.time + toc;

        for l = L : -1 : 1
            % coarse correction with line search
            alpha = 0.5;
            while 1
                cor = alpha*c*R{l}'*(w-x{l+1});
                zk = y{l} + cor;
                if 0.5*cor'*Q{l}*cor+cor'*(p{l}+Q{l}*y{l}) <= 0
                    z = zk;
                    break;
                elseif alpha > eps
                    alpha = alpha / 2;
                else
                    z = y{l};
                    break;
                end
            end
            % post-smoothing
            w = z;
            for sm = 1 : smooth
                w = max(0, w-1/L0*(Q{l}*w+p{l}-tau{l}));
            end
        end
        % update the fine variable
        xk0 = xk;
        xk = w;
%         t0 = t; t = 0.5*(1+sqrt(1+4*t^2));
        objold = obj;
    end
end