function r = rate(config, data, i, x_hat)

g = squeeze(data.G(i, :, :));
h = squeeze(data.H(i, :, :));
[W, Theta_r, Theta_t] = get_theta(config, x_hat);


if config.star
    h_t = h(:, 1:2);
    h_r = h(:, 3:4);
    signal_t = h_t'*Theta_t*g*W;
    signal_r = h_r'*Theta_r*g*W;

else
    h_t = h(1:config.N/2, 1:2);
    h_r = h(config.N/2 + 1:end, 3:4);
    g_t = g(1:config.N/2, :);
    g_r = g(config.N/2 + 1:end, :);
    signal_t = h_t'*Theta_t*g_t*W;
    signal_r = h_r'*Theta_r*g_r*W;
end

all_signals = abs(cat(1, signal_t, signal_r)).^2;
recieved_mask = eye(config.K);
infer_mask = ones(config.K) - recieved_mask;
rec_signal = sum((recieved_mask.*all_signals), 2);
infer_signal = sum((infer_mask.*all_signals), 2);
sinr = rec_signal./(infer_signal + config.N0);
r = log(1 + sinr)./log(2);
end