%% Example - how to use the iPALM code for a CDL problem
clear; clc; 
figure(2); clf;
initpkg;

%% Generate some synthetic data, activation map values are {0,1}.
[A0, X0, Y] = genconvdata_sbd_constantX(struct('N', 5));

% Mess up the data a little
eta = 5e-2;                             % Add some noise
b0 = 1*rand(1,numel(Y));                   % Add a random constant bias

for i = 1:numel(Y)
    Y{i} = Y{i} + b0(i) + eta * randn(size(Y{i}));
end


%% Set up parameters for iPALM iterations to solve CDL problem. 
% Initial solve
p = size(A0{1});                        % Choose a recovery window size
lambda1 = 0.05;                          % Sparsity regularization parameter

xpos = true;                            % Recover a nonnegative activation map
getbias = true;                         % Recover a constant bias

% Reweighting
reweights = 0;                          % number of reweights
lambda2 = 1e-2;                         % lambda for reweighting
eps = 1e-2;                             % reweighting param

% Iterations and updates
maxit = 1e2 * ones(reweights+1,1);      % iterations per reweighting loop
maxit(1) = 1e4+200;                         % iterations in initial iPALM solve

centerfq = 5e2;                         % frequency to recenter the data
updates = [ 1 10:10:50 ...              % when to print updates
            100:100:500 ...
            600:200:max(maxit)];


%% Initialize solver + run some iterations of iPALM
solver = sbd_ipalm(Y, p, lambda1, xpos, getbias);     
reweight_loop;
%profile off; profile viewer;