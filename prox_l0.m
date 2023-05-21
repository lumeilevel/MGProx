function x = prox_l0(x, t)
    x = (t < 0.5*x.^2) .* (x);
end