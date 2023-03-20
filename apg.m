function [x, hist] = apg(Q, p, L0, x_ini, eps)
    [n, ~] = size(Q);
    max_iter = n;
    t = 1;  t0 = 1;
    eta = 0.5;
    tauk = L0;
    x0 = x_ini; x = x0;
    objold = 0.5*x0'*Q*x0 + p'*x;
    hist.F = zeros(max_iter, 1);
    hist.G = zeros(max_iter, 1);
    hist.dist = zeros(max_iter, 1);
    hist.relDist = zeros(max_iter, 1);
    hist.relObjdiff = zeros(max_iter, 1);
    for iter = 1 : max_iter
        df = Q*x+p;
        obj = 0.5*x'*(df+p);
        hist.F(iter) = obj;
        hist.G(iter) = L0*norm(x-max(0,x-1/L0*df));
        hist.relDist(iter) = norm(x-x0) / norm(x);
        hist.relObjdiff(iter) = abs(obj - objold) / max(obj, 1);
        hist.dist(iter) = norm(df);
        % stopping criterion
        if hist.G(iter) / hist.G(1) <= eps
            hist.F = hist.F(1:iter);
            hist.G = hist.G(1:iter);
            hist.dist = hist.dist(iter);
            hist.relDist = hist.relDist(iter);
            hist.relObjdiff = hist.relObjdiff(iter);
            fprintf('\n APG early stopping--iteration: %d\n', iter);
            fprintf('[c] proximal first-order optimality condition satisfied\n')
            break
        end
        if iter > 4
            if max(hist.relDist(iter), 0.1*hist.relObjdiff(iter)) < eps
                fprintf("\n APG Early Stopping--iteration: %d\n", iter);
                fprintf('[a] relDist < %3.2e', tol);
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
                fprintf('[b] relObjdiff < %3.2e', 0.01*tol);
                hist.F = hist.F(1:iter);
                hist.G = hist.G(1:iter);
                hist.dist = hist.dist(1:iter);
                hist.relDist = hist.relDist(1:iter);
                hist.relObjdiff = hist.relObjdiff(1:iter);
                break;
            end
        end
        y = x + (t0-1)/t*(x-x0);
        tau = eta * tauk;
        for i = 1 : 1e2
            g = y - 1/tau*(Q*y+p);
            s = max(0, g);
            if (s-y)'*Q*(s-y) <= tau*(s-y)'*(s-y)
                tauk = tau;
                break;
            else
                tau = min(tau/eta, L0);
            end
        end
        x0 = x;
        x = s;
        t0 = t; t = 0.5*(1+sqrt(1+4*t^2));
        objold = obj;
    end
end