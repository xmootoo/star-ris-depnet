function [W, Theta_r, Theta_t] = get_theta(config, x_hat)

if config.implicit
    if config.star
        phi_t = x_hat(1, 1:config.N);
        phi_r = x_hat(1, config.N +1 : 2*config.N);
        a = x_hat(1, 2*config.N +1 : 3*config.N);
        w = x_hat(1, 3*config.N +1 :end);
    else
        phi_t = x_hat(1, 1:config.N/2);
        phi_r = x_hat(1, config.N/2 +1 : config.N);
        w = x_hat(1, config.N +1 :end);
    end
    amp = w(1, 1);
    b = w(1, 2:end);
%     amp = 1/(1 + exp(-amp));
    b = reshape(b, 2, config.K, config.M)./norm(b);
    b = permute(b,[3 2 1]);
    W = amp.*(b(:,:,1) + 1j*b(:,:,2)).*sqrt(config.P_max);
    
    
    if config.star
%         coef = 1./(1  + exp(-a));
%         a_t = (1 - config.utol)*coef + (1-coef)*(config.ltol);
        a_t = a;
        a_r = 1 - a_t;
        
    
        Theta_t = diag(sqrt(a_t).*exp(1j*phi_t));
        Theta_r = diag(sqrt(a_r).*exp(1j*phi_r));
    else
        Theta_t = diag(exp(1j*phi_t));
        Theta_r = diag(exp(1j*phi_r));
    
    end



else
    if config.star
        phi_t = x_hat(1, 1:config.N);
        phi_r = x_hat(1, config.N +1 : 2*config.N);
        a_t = x_hat(1, 2*config.N +1 : 3*config.N);
        a_r = 1 - a_t;
        w = x_hat(1, 3*config.N +1 :end);
    else
        phi_t = x_hat(1, 1:config.N/2);
        phi_r = x_hat(1, config.N/2 +1 : config.N);
        w = x_hat(1, config.N +1 :end);
    end
    
    temp = reshape(w, 2, config.K, config.M);
    temp = permute(temp,[3 2 1]);
    W = temp(:,:,1) + 1j*temp(:,:,2);

    if config.star
    
        Theta_t = diag(sqrt(a_t).*exp(1j*phi_t));
        Theta_r = diag(sqrt(a_r).*exp(1j*phi_r));
    else
        Theta_t = diag(exp(1j*phi_t));
        Theta_r = diag(exp(1j*phi_r));
    
    end



end



end