clear all;
close all;
%%
% for id = [832, 8128, 8256]
%     if id == 832
%         N_id = 32
%     elseif id == 8128
%         N_id = 128
%     elseif id == 8256
%         N_id = 256
%     end
%     for flag = [true, false]
% id = 22;
% flag = true;
        clc;
    %% load the data and set the config
    
    % Dataset ID
    config.data_type = int2str(15);
    
    % Number of base station channels
    config.M = 8;
    
    % Number of users
    config.K = 4;
    
    % Number of RIS elements
    config.N = 64;
    config.utol = 1e-5;
    config.ltol = 1e-5;
    
    % Minimum rate constraint
    config.r_min = 2.5;
%     if mod(id, 5) ~= 0
%         config.P_max = mod(id,5);
%     else
%         config.P_max = 5;
%     end
    
    % Maximum power constraint
    config.P_max = 5;
    
    % STAR-RIS (true) or Normal RIS (false)
    config.star = false;
    
    N0_dbm = -170;
    config.N0 = (10^((N0_dbm - 30)/10))*180*1e3;
    config.implicit = true;
    config.Dataset_dir = ['.\Datasets\Star\' config.data_type  '\test\data.mat'];
    if config.star
        init_data_dir = ['.\Datasets\Star\' config.data_type '\init_star_wc.mat' ];
    else
        init_data_dir = ['.\Datasets\Star\' config.data_type '\init_normal_wc.mat' ];
    end
    
    if isfile(config.Dataset_dir)
        data = load(config.Dataset_dir);
    else
        fprintf('Error - file does not exist\n');
    end
    
    if isfile(init_data_dir)
        d_init = load(init_data_dir);
        x_init = d_init.init;
    else
        fprintf('Error - file does not exist\n');
    end

    %%
    % i = 10;
    % x_hat = ones(1, 2*config.N + 2*config.M*config.K + 1);
    % g = data.G(i, :, :);
    % h = data.H(i, :, :);
    % r = rate(config, data, i, x_hat);
    
    % lb = zeros(1, 3*config.M) + 1e-6;
    % ub = ones(1,3*config.M) - 1e-6;
    % ub(1,config.M + 1:end) = pi*ub(1,config.M + 1:end);
%     config.implicit = true;
%     config.star = true;
% for dtype = [true, false]


   
        if config.star
           dim = 3*config.N + 2*config.M*config.K + 1;
           lb = -Inf*ones(1, dim);
           lb(1, 1:3*config.N + 1) = 0;
           ub = +Inf*ones(1, dim);
           ub(1, 1:2*config.N) = 2*pi;
           ub(1, 2*config.N+1: 3*config.N + 1) = 1;
        else
    
           dim = config.N + 2*config.M*config.K + 1;

           lb = -Inf*ones(1, dim);
           lb(1, 1:config.N + 1) = 0;
           ub = +Inf*ones(1, dim);
           ub(1, 1:config.N) = 2*pi;
           ub(1, config.N + 1) = 1;
        end
   
    
    % i = 10;
    % x_hat = ones(1, dim);
    % g = data.G(i, :, :);
    % h = data.H(i, :, :);
    % r = rate(config, data, i, x_hat);
    
    % Number of data points
    num_data = 3;
    
    % Initialization
    x_ga = zeros(num_data,dim);
    compute_time = zeros(num_data, 1);
    constraint_vio = ones(num_data, 1);
    rate_ga = zeros(num_data, 1);
% if config.star
%         data_path = ['.\Datasets\Star\' config.data_type '\ga_star_wc.mat' ];
%     else
%         data_path = ['.\Datasets\Star\' config.data_type '\ga_normal_wc.mat' ];
%     end
% load(data_path);
    %%
    % you can comment the constraint if you don't have one and replace the
    % constraint in the ga with []. Also, please name the output file
    % accordingly. something like this: if you are using P_max=0.6 and r_min =
    % 2 with star IRS use 'ga_star_2_0.6.mat'
    % if you don't have the initial points from another algorithm, just comment
    % it.
    for i = 1:num_data
    i
    for j = 1:3
        j
    if constraint_vio(i,:) == 0
        break;
    elseif j > 1
        options.InitialPopulationMatrix = x_init(i,:);
    end
    
        
    func = @(x)-1*sum(rate(config, data, i, x));
    constraint = @(x)const(config, data, i, x);
    % x_init = data.x_init(i,:);
    % x_hat = zeros(1, 3*config.M);
    % x_hat(1, 1:2*config.M) = x_init(1, 1:2*config.M);
    % x_hat(1, 2*config.M + 1: end) = x_init(1, 3*config.M + 1:end);
    
    
    % r = Rate(config, x_fw, h, true);
    % t = A_hat*x_fw' - b_hat';
    s = tic;
    options = optimoptions('ga','FunctionTolerance', 1e-20,'ConstraintTolerance',1e-20,'MaxGenerations', 100,'PlotFcn', @gaplotbestf);
    
%     options.InitialPopulationRange = [-1;10];
    res = ga(func, dim, [], [], [], [], lb,ub, constraint, [], options);
    t = toc(s);
    
    r_n = func(res);
    constraint_vio(i,:) = sum(max(constraint(res), 0));
    compute_time(i,:) = t;
    
    x_ga(i, :) = res;
    rate_ga(i,:)= -r_n;
    end
    end
%     end
    %%
    
    % Compute mean values
    mean_ga = mean(x_ga, 1);
    mean_compute_time = mean(compute_time);
    mean_rate_ga = mean(rate_ga);
    mean_constraint_vio = mean(constraint_vio);
    
    % Save values
    if config.star
        data_path = ['.\Datasets\Star\' config.data_type '\ga_star_wc.mat' ];
    else
        data_path = ['.\Datasets\Star\' config.data_type '\ga_normal_wc.mat' ];
    end
    save(data_path, 'x_ga', 'compute_time', 'rate_ga', 'constraint_vio', 'mean_ga', 'mean_compute_time', 'mean_rate_ga', 'mean_constraint_vio');
%     end
% end
% end
%%
% i = 3;
% x_hat = x_ga(i,:);
% r = rate(config, data, i, x_hat);
% [c, ceq] = const(config, data, i, x_hat);
% i = 3;
% x_hat = x_init(i,:);
% rate(config, data, i, x_hat)
% const(config, data, i, x_hat)
%%
% for i,p,r = [[1;2;3], [4;3;2]]
%     i
%     p
%     r
% end

