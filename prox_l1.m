function x = prox_l1(x, t)
    x = sign(x) .* max(abs(x)-t,0);
end