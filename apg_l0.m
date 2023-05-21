function [x, hist] = apg_l0(A, b, c, lambda, L0, x_ini, tol, verbose)
    [n, ~] = size(A);
    max_iter = n*1e2;
    t = 1;  t0 = 1;
    eta = 0.5;
    tauk = L0;
    x0 = x_ini; x = x0;
    objold = 0.5*((A*x0-b)'*(A*x0-b)) + c'*x0 + lambda*sum(abs(x0) > eps);
    hist.F = zeros(max_iter, 1);
    hist.G = zeros(max_iter, 1);
    hist.dist = zeros(max_iter, 1);
    hist.relDist = zeros(max_iter, 1);
    hist.relObjdiff = zeros(max_iter, 1);
    hist.time = 0;
    for iter = 1 : max_iter
        Axb = A*x-b;
        grad = A'*Axb + c;
        hist.dist(iter) = norm(grad);
        obj = 0.5*(Axb'*Axb) + c'*x + lambda*sum(abs(x) > eps);
        hist.F(iter) = obj;
        hist.G(iter) = L0*norm(x-prox_l0(x-grad/L0,lambda/L0));
        hist.relDist(iter) = norm(x-x0) / norm(x);
        hist.relObjdiff(iter) = abs(obj - objold) / max(obj, 1);       
        % stopping criterion
        if hist.G(iter) / hist.G(1) <= tol
            hist.F = hist.F(1:iter);
            hist.G = hist.G(1:iter);
            hist.dist = hist.dist(1:iter);
            hist.relDist = hist.relDist(1:iter);
            hist.relObjdiff = hist.relObjdiff(1:iter);
            if verbose
                fprintf('\n APG early stopping--iteration: %d\n', iter);
                fprintf('[c] proximal first-order optimality condition satisfied\n')
            end
            break
        end
        if iter > 4
            if ~verbose && hist.G(iter) > hist.G(iter-1) * 1e1
                iter = iter - 1;
                hist.F = hist.F(1:iter);
                hist.G = hist.G(1:iter);
                hist.dist = hist.dist(1:iter-1);
                hist.relDist = hist.relDist(1:iter);
                hist.relObjdiff = hist.relObjdiff(1:iter);
                x = x0;
                break
            end
            if max(hist.relDist(iter), 0.1*hist.relObjdiff(iter)) < tol
                if verbose
                    fprintf("\n APG Early Stopping--iteration: %d\n", iter);
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
                    fprintf("\n APG Early Stopping--iteration: %d\n", iter);
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
        y = x + (t0-1)/t*(x-x0);
        tic;
        tau = eta * tauk;   Ayb = A*y - b;  grad_c = A'*Ayb;
        for j = 1 : 1e3
            s = prox_l0(y - (grad_c+c) / tau, lambda / tau);
            Asb = A * s - b;
            sy = s - y;
            if (Asb'*Asb) <= (Ayb'*Ayb) + 2*sy'*grad_c + tau*(sy'*sy) + eps
                tauk = tau;
                break;
            else
                tau = min(tau/eta, L0);
            end
        end
        hist.time = hist.time + toc;
        x0 = x;
        x = s;
        t0 = t; t = 0.5*(1+sqrt(1+4*t^2));
        objold = obj;
    end
end