function [c, ceq] = const(config, data, i, x_hat)
c = zeros(1, config.K);
r = rate(config, data, i, x_hat);
c(1,:) = config.r_min - r + 1e-5;
ceq = 0;
end