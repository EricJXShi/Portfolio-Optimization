clc, clear, close all
options = optimset('Display', 'off'); xno = 0;
%% Find Expected Return, Standard Deviation, and Covariances
%% Given Information (All .csv files were taken from Yahoo Finance) - Dates from Jan 2005 to Nov 2008
path = 'csvfiles/'; % Specify the relative path to the location of all the .csv files
storage = dir(fullfile(path, '*.csv')); % Create a structure array to store filenames and file data
total_assets = numel(storage); % Number of csv files = number of assets
for i = 1:total_assets
    F = fullfile(path, storage(i).name); % i-th .csv file path
    opts = detectImportOptions(F); % Locate import options for the file
    opts.SelectedVariableNames = 6; % Only select the 6th column, which is Adjusted Closing Price
    storage(i).data = readmatrix(F, opts); % Store the data from the 6th column
end

%% Create an Array of Adjusted Closing Prices
% Stocks are ordered alphabetically: 
% [AAPL, C, CAT, DIS, ED, F, IBM, JNJ, JPM, KO, MCD, MRO, NEM, PEP, PFE, T, VZ, WFC, WMT, XOM]
total_points = length(storage(1).data)-2; % Number of data points per asset (Excluding Oct and Nov 2008)
total_months = total_points-1; % Number of elapsed months (Excluding Oct and Nov 2008)
adj_close = zeros(total_points+2, total_assets); % Initialize the array of Adjusted Closing Prices [row, col] = [month, stock]
for i = 1:total_assets
    adj_close(:, i) = storage(i).data; % Populate the array of Adjusted Closing Prices
end

% Find the monthly returns for each asset
r_m = zeros(total_months+2, total_assets); % Initialize the monthly returns matrix
for i = 1:total_months+2
    r_m(i, :) = adj_close(i+1, :)./adj_close(i, :) - 1; % Populate the monthly returns matrix
end

% Calculate the mean, variance, and covariance for each stock
mu = mean(r_m(1:end-2, :)); % Sample mean for each asset up to Sept 2008
Q = cov(r_m(1:end-2, :)); % Sample covariance for each asset up to Sept 2008 (Variances are in the diagonal)

%% Consider the Market Portfolio
% Market Capitalization Data (USD) was taken for Sept 30, 2008 from Bloomberg Terminal
AAPL_MC = 113919*10^6;
C_MC = 111770*10^6;
CAT_MC = 35953*10^6;
DIS_MC = 59700*10^6;
ED_MC = 11751*10^6;
F_MC = 12422*10^6;
IBM_MC = 157131*10^6;
JNJ_MC = 192955*10^6;
JPM_MC = 174048*10^6;
KO_MC = 121334*10^6;
MCD_MC = 68765*10^6;
MRO_MC = 28148*10^6;
NEM_MC = 17608*10^6;
PEP_MC = 107308*10^6;
PFE_MC = 125823*10^6;
T_MC = 164533*10^6;
VZ_MC = 91151*10^6;
WFC_MC = 124645*10^6;
WMT_MC = 230610*10^6;
XOM_MC = 395056*10^6;

MC_total = [
    AAPL_MC;
    C_MC;
    CAT_MC;
    DIS_MC;
    ED_MC;
    F_MC;
    IBM_MC;
    JNJ_MC;
    JPM_MC;
    KO_MC;
    MCD_MC;
    MRO_MC;
    NEM_MC;
    PEP_MC;
    PFE_MC;
    T_MC;
    VZ_MC;
    WFC_MC;
    WMT_MC;
    XOM_MC
    ];

x_mkt = 1/sum(MC_total)*MC_total; % Market Portfolio

%% Find the Risk Aversion Coefficient using the Idzorek Black-Litterman Paper
% Assume the market portfolio is the optimal Risk Aversion Coefficient
avg_rf_y = (.0366 + .0463 + .048 + .0429)/4; % Average yearly risk free rate. Taken as the average of the 10 Year US T-Bill (2005-2008). Backed by US Govt No Default Risk
rf = (1+avg_rf_y)^(1/12) - 1; % Monthly Effective Risk-free Rate.
r_mkt = mu*x_mkt; % Market Portfolio Expected Returns
var_mkt = x_mkt'*Q*x_mkt; % Market Portfolio Variance of Excess Returns
lambda_mkt = (r_mkt - rf)/var_mkt; % Risk Aversion Coefficient

%% (1) Mean-Variance Optimization (MVO)
H = 2*lambda_mkt*Q;
f = -mu';
A = [];
b = [];
Aeq = ones(1, total_assets);
beq = 1;
lb = [];
ub = [];
[x_MVO, fval_MVO] = quadprog(H, f, A, b, Aeq, beq, lb, ub, xno, options);

%% (2) Robust Mean-Variance Optimization (Box)
theta = 1/total_points*diag(diag(Q));
theta_half = sqrt(theta); % Diagonal matrix of standard errors of the asset expected returns
eps1_95 = 1.96; % 95% Confidence Level Based on Z-Score Tables
eps1_90 = 1.645; % 90% Confidence Level Based on Z-Score Tables
delta_95 = eps1_95*theta_half*ones(total_assets, 1); % Maximum distance between expected returns and true returns at 95% confidence
delta_90 = eps1_90*theta_half*ones(total_assets, 1); % Maximum distance between expected returns and true returns at 90% confidence

Q_RMVO = zeros(total_assets*2, total_assets*2); % Initialize covariance matrix including auxiliary variable y
Q_RMVO(1:20, 1:20) = Q_RMVO(1:20, 1:20) + Q; % Covariance matrix including auxiliary variable y

H = 2*lambda_mkt*Q_RMVO;
f_95 = [-mu'; delta_95];
f_90 = [-mu'; delta_90];
A = zeros(total_assets*2, total_assets*2);
for i = 1:total_assets
    A(i, i) = A(i, i) + 1;
    A(i+total_assets, i) = A(i+total_assets, i) - 1;
    A(i, i+total_assets) = A(i, i+total_assets) - 1;
    A(i+total_assets, i+total_assets) = A(i+total_assets, i+total_assets) - 1;
end
b = zeros(total_assets*2, 1);
Aeq = [ones(1, total_assets), zeros(1, total_assets)];
beq = 1;
lb = []; % Short selling is allowed
ub = [];

x_RMVO_box_95 = quadprog(H, f_95, A, b, Aeq, beq, lb, ub, xno, options);
x_RMVO_box_90 = quadprog(H, f_90, A, b, Aeq, beq, lb, ub, xno, options);

%% (2) Robust Mean-Variance Optimization (Ellipse)
eps2_95 = sqrt(chi2inv(0.95, total_assets)); % 95% Confidence Level Based on Chi-Squared Tables
eps2_90 = sqrt(chi2inv(0.9, total_assets)); % 90% Confidence Level Based on Chi-Squared Tables
x0 = repmat(1/total_assets, total_assets, 1); % Initial guess with equal weighted portfolio

func_95 = @(x) lambda_mkt*x'*Q*x - mu*x + eps2_95*sqrt(x'*theta*x);
func_90 = @(x) lambda_mkt*x'*Q*x - mu*x + eps2_90*sqrt(x'*theta*x);

A = []; b = []; % Inequality constraints
Aeq = ones(1, total_assets); beq = 1; % Equality constraints
lb = []; ub = []; % Short selling is allowed
nonlcon = [];

x_RMVO_ell_95 = fmincon(func_95, x0, A, b, Aeq, beq, lb, ub, nonlcon, options);
x_RMVO_ell_90 = fmincon(func_90, x0, A, b, Aeq, beq, lb, ub, nonlcon, options);

%% (3) Risk Parity Optimization with No Short Selling
syms x [total_assets 1]
syms k

func = 0; % Initialize Risk Parity Objective Function
for i = 1:total_assets
    func = func + (x(i)*Q(i, :)*x - k)^2; % Build Objective function as per Slide 17, k = theta in the slides
end
y = sym('y',[1 21]);
func = subs(func,[k x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15...
    x16 x17 x18 x19 x20],[y(1) y(2) y(3) y(4) y(5) y(6) y(7) y(8) y(9) ...
    y(10) y(11) y(12) y(13) y(14) y(15) y(16) y(17) y(18) y(19) y(20)...
    y(21)]); % Make the symbolic function compatible for conversion to function handle for fmincon
func = matlabFunction(func, 'Vars', {y}); % Convert the symbolic function into a matlab function handle for fmincon

x0 = [1, 1/total_assets*ones(1, total_assets)]; % Initial guess with the equal weighted portfolio and k = 1
A = [];
b = [];
Aeq = [0, ones(1, total_assets)];
beq = 1;
lb = [-inf, zeros(1, total_assets)]; % No short selling allowed
ub = [];
nonlcon = [];

x_RP = fmincon(func, x0, A, b, Aeq, beq, lb, ub, nonlcon, options);

%% Part a)
mu_oct = r_m(end-1, :); % Asset returns for Oct 2008

% Portfolio Returns for Oct 2008
r_mkt_oct = mu_oct*x_mkt;
r_MVO_oct = mu_oct*x_MVO;
r_RMVO_box_95_oct = mu_oct*x_RMVO_box_95(1:20);
r_RMVO_box_90_oct = mu_oct*x_RMVO_box_90(1:20);
r_RMVO_ell_95_oct = mu_oct*x_RMVO_ell_95;
r_RMVO_ell_90_oct = mu_oct*x_RMVO_ell_90;
r_RP_oct = mu_oct*x_RP(2:end)';

% Portfolio Variance and Standard Deviation for Oct 2008
var_mkt_oct = x_mkt'*Q*x_mkt;
var_MVO_oct = x_MVO'*Q*x_MVO;
var_RMVO_box_95_oct = x_RMVO_box_95(1:20)'*Q*x_RMVO_box_95(1:20);
var_RMVO_box_90_oct = x_RMVO_box_90(1:20)'*Q*x_RMVO_box_90(1:20);
var_RMVO_ell_95_oct = x_RMVO_ell_95(1:20)'*Q*x_RMVO_ell_95(1:20);
var_RMVO_ell_90_oct = x_RMVO_ell_90(1:20)'*Q*x_RMVO_ell_90(1:20);
var_RP_oct = x_RP(2:end)*Q*x_RP(2:end)';

std_mkt_oct = sqrt(var_mkt_oct);
std_MVO_oct = sqrt(var_MVO_oct);
std_RMVO_box_95_oct = sqrt(var_RMVO_box_95_oct);
std_RMVO_box_90_oct = sqrt(var_RMVO_box_90_oct);
std_RMVO_ell_95_oct = sqrt(var_RMVO_ell_95_oct);
std_RMVO_ell_90_oct = sqrt(var_RMVO_ell_90_oct);
std_RP_oct = sqrt(var_RP_oct);

% Sharpe Ratio
sharpe_mkt_oct = (r_mkt_oct-rf)/std_mkt_oct;
sharpe_MVO_oct = (r_MVO_oct-rf)/std_MVO_oct;
sharpe_RMVO_box_95_oct = (r_RMVO_box_95_oct-rf)/std_RMVO_box_95_oct;
sharpe_RMVO_box_90_oct = (r_RMVO_box_90_oct-rf)/std_RMVO_box_90_oct;
sharpe_RMVO_ell_95_oct = (r_RMVO_ell_95_oct-rf)/std_RMVO_ell_95_oct;
sharpe_RMVO_ell_90_oct = (r_RMVO_ell_90_oct-rf)/std_RMVO_ell_90_oct;
sharpe_RP_oct = (r_RP_oct-rf)/std_RP_oct;

%% Part b)
mu_nov = r_m(end, :); % Asset returns for Nov 2008

% Portfolio Returns for Nov 2008
r_mkt_nov = mu_nov*x_mkt;
r_MVO_nov = mu_nov*x_MVO;
r_RMVO_box_95_nov = mu_nov*x_RMVO_box_95(1:20);
r_RMVO_box_90_nov = mu_nov*x_RMVO_box_90(1:20);
r_RMVO_ell_95_nov = mu_nov*x_RMVO_ell_95;
r_RMVO_ell_90_nov = mu_nov*x_RMVO_ell_90;
r_RP_nov = mu_nov*x_RP(2:end)';

% Portfolio Variance and Standard Deviation for Nov 2008
var_mkt_nov = x_mkt'*Q*x_mkt;
var_MVO_nov = x_MVO'*Q*x_MVO;
var_RMVO_box_95_nov = x_RMVO_box_95(1:20)'*Q*x_RMVO_box_95(1:20);
var_RMVO_box_90_nov = x_RMVO_box_90(1:20)'*Q*x_RMVO_box_90(1:20);
var_RMVO_ell_95_nov = x_RMVO_ell_95(1:20)'*Q*x_RMVO_ell_95(1:20);
var_RMVO_ell_90_nov = x_RMVO_ell_90(1:20)'*Q*x_RMVO_ell_90(1:20);
var_RP_nov = x_RP(2:end)*Q*x_RP(2:end)';

std_mkt_nov = sqrt(var_mkt_nov);
std_MVO_nov = sqrt(var_MVO_nov);
std_RMVO_box_95_nov = sqrt(var_RMVO_box_95_nov);
std_RMVO_box_90_nov = sqrt(var_RMVO_box_90_nov);
std_RMVO_ell_95_nov = sqrt(var_RMVO_ell_95_nov);
std_RMVO_ell_90_nov = sqrt(var_RMVO_ell_90_nov);
std_RP_nov = sqrt(var_RP_nov);

% Sharpe Ratio
sharpe_mkt_nov = (r_mkt_nov-rf)/std_mkt_nov;
sharpe_MVO_nov = (r_MVO_nov-rf)/std_MVO_nov;
sharpe_RMVO_box_95_nov = (r_RMVO_box_95_nov-rf)/std_RMVO_box_95_nov;
sharpe_RMVO_box_90_nov = (r_RMVO_box_90_nov-rf)/std_RMVO_box_90_nov;
sharpe_RMVO_ell_95_nov = (r_RMVO_ell_95_nov-rf)/std_RMVO_ell_95_nov;
sharpe_RMVO_ell_90_nov = (r_RMVO_ell_90_nov-rf)/std_RMVO_ell_90_nov;
sharpe_RP_nov = (r_RP_nov-rf)/std_RP_nov;

%% Part c) Assume Short Selling is Allowed
% Define constants
n_points = 25; % Number of equally spaced points on the efficient frontier
lambda = linspace(50, 500, n_points); % Define the risk aversion coefficients
lb = []; ub = []; % Short selling is allowed
f_est = -mu'; f_true = -mu_oct;
A = []; b = []; % Inequality constraints
Aeq = ones(1, total_assets); beq = 1; % Equality constraints
x0 = repmat(1/total_assets, total_assets, 1); % Initial guess with equal weighted portfolio
nonlcon = []; % Non-linear constraints

% Generate the Estimated MVO Frontier
std_portfolio = zeros(n_points, 1); % Initialize a vector of standard deviations
optimal_weights = zeros(n_points, total_assets); % Initialize a matrix of weights
est_MVO_R = zeros(n_points, 1); % Initialize a vector of estimated returns
for i = 1:n_points
    H = 2*lambda(i)*Q;
    [x, fval] = quadprog(H, f_est, A, b, Aeq, beq, lb, ub, xno, options); % Find weights and lambda*portfolio variance (objective function)
    std_portfolio(i) = ((fval+mu*x)/lambda(i))^0.5; % Store the portfolio standard deviation for plotting
    optimal_weights(i, :) = x; % Store the optimal weights
    est_MVO_R(i) = mu*x; % Store the estimated MVO return
end

% Generate the True MVO Frontier
std_true = zeros(n_points, 1); % Initialize a vector of standard deviations
true_optimal_weights = zeros(n_points, total_assets); % Initialize a matrix of weights
true_MVO_R = zeros(n_points, 1); % Initialize a vector of true returns
for i = 1:n_points
    H = 2*lambda(i)*Q;
    [x, fval] = quadprog(H, f_true, A, b, Aeq, beq, lb, ub, xno, options); % Find weights and lambda*portfolio variance (objective function)
    std_true(i) = ((fval+mu_oct*x)/lambda(i))^0.5; % Store the portfolio standard deviation for plotting
    true_optimal_weights(i, :) = x; % Store the optimal weights
    true_MVO_R(i) = mu_oct*x; % Store the true MVO return
end

% Generate the Actual MVO Frontier
actual_MVO_R = zeros(n_points, 1);
std_actual_MVO = zeros(n_points, 1);
for i = 1:n_points
    actual_MVO_R(i) = mu_oct*optimal_weights(i, :)';
    std_actual_MVO(i) = (optimal_weights(i, :)*Q*optimal_weights(i, :)')^0.5;
end

% Generate the Estimated Robust Ellipsoid MVO Frontier 95% and 90%
std_est_ell_95 = zeros(n_points, 1); % Initialize a vector of standard deviations
std_est_ell_90 = zeros(n_points, 1); % Initialize a vector of standard deviations
optimal_ell_95_weights = zeros(n_points, total_assets);
optimal_ell_90_weights = zeros(n_points, total_assets);
est_95_ell_R = zeros(n_points, 1);
est_90_ell_R = zeros(n_points, 1);
for i = 1:n_points
    func_95 = @(x) lambda(i)*x'*Q*x - mu*x + eps2_95*sqrt(x'*theta*x);
    func_90 = @(x) lambda(i)*x'*Q*x - mu*x + eps2_90*sqrt(x'*theta*x);
    [x_RMVO_ell_95, fval_RMVO_ell_95] = fmincon(func_95, x0, A, b, Aeq, beq, lb, ub, nonlcon, options);
    [x_RMVO_ell_90, fval_RMVO_ell_90] = fmincon(func_90, x0, A, b, Aeq, beq, lb, ub, nonlcon, options);
    std_est_ell_95(i) = (x_RMVO_ell_95'*Q*x_RMVO_ell_95)^0.5;
    std_est_ell_90(i) = (x_RMVO_ell_90'*Q*x_RMVO_ell_90)^0.5;
    optimal_ell_95_weights(i, :) = x_RMVO_ell_95';
    optimal_ell_90_weights(i, :) = x_RMVO_ell_90';
    est_95_ell_R(i) = mu*x_RMVO_ell_95;
    est_90_ell_R(i) = mu*x_RMVO_ell_90;
end

% Generate the Actual Robust Ellipsoid MVO Frontier 95% and 90%
actual_95_ell_R = zeros(n_points, 1);
actual_90_ell_R = zeros(n_points, 1);
std_actual_95_ell = zeros(n_points, 1);
std_actual_90_ell = zeros(n_points, 1);
for i = 1:n_points
    actual_95_ell_R(i) = mu_oct*optimal_ell_95_weights(i, :)';
    actual_90_ell_R(i) = mu_oct*optimal_ell_90_weights(i, :)';
    std_actual_95_ell(i) = (optimal_ell_95_weights(i, :)*Q*optimal_ell_95_weights(i, :)')^0.5;
    std_actual_90_ell(i) = (optimal_ell_90_weights(i, :)*Q*optimal_ell_90_weights(i, :)')^0.5;
end

% Generate the Estimated Robust Box MVO Frontier 95% and 90%
A = zeros(total_assets*2, total_assets*2); % Initialize inequality constraints
for i = 1:total_assets % Inequality constraints for the auxiliary variable y
    A(i, i) = A(i, i) + 1;
    A(i+total_assets, i) = A(i+total_assets, i) - 1;
    A(i, i+total_assets) = A(i, i+total_assets) - 1;
    A(i+total_assets, i+total_assets) = A(i+total_assets, i+total_assets) - 1;
end
b = zeros(total_assets*2, 1);
Aeq = [ones(1, total_assets), zeros(1, total_assets)];
beq = 1;

std_box_95 = zeros(n_points, 1); % Initialize a vector of standard deviations
std_box_90 = zeros(n_points, 1); % Initialize a vector of standard deviations
optimal_box_95_weights = zeros(n_points, total_assets);
optimal_box_90_weights = zeros(n_points, total_assets);
est_RMVO_95_box_R = zeros(n_points, 1);
est_RMVO_90_box_R = zeros(n_points, 1);
for i = 1:n_points
    H = 2*lambda(i)*Q_RMVO;
    [x_RMVO_box_95, fval_RMVO_box_95] = quadprog(H, f_95, A, b, Aeq, beq, lb, ub, xno, options);
    [x_RMVO_box_90, fval_RMVO_box_90] = quadprog(H, f_90, A, b, Aeq, beq, lb, ub, xno, options);
    std_box_95(i) = (x_RMVO_box_95'*Q_RMVO*x_RMVO_box_95)^0.5;
    std_box_90(i) = (x_RMVO_box_90'*Q_RMVO*x_RMVO_box_90)^0.5;
    optimal_box_95_weights(i, :) = x_RMVO_box_95(1:20)';
    optimal_box_90_weights(i, :) = x_RMVO_box_90(1:20)';
    est_RMVO_95_box_R(i) = mu*x_RMVO_box_95(1:20);
    est_RMVO_90_box_R(i) = mu*x_RMVO_box_90(1:20);
end

% Generate the Actual Robust Box MVO Frontier 95% and 90%
actual_95_box_R = zeros(n_points, 1);
actual_90_box_R = zeros(n_points, 1);
std_actual_95_box = zeros(n_points, 1);
std_actual_90_box = zeros(n_points, 1);
for i = 1:n_points
    actual_95_box_R(i) = mu_oct*optimal_box_95_weights(i, :)';
    actual_90_box_R(i) = mu_oct*optimal_box_90_weights(i, :)';
    std_actual_95_box(i) = (optimal_box_95_weights(i, :)*Q*optimal_box_95_weights(i, :)')^0.5;
    std_actual_90_box(i) = (optimal_box_90_weights(i, :)*Q*optimal_box_90_weights(i, :)')^0.5;
end

%% Plotting
figure(1)
hold on
plot(std_portfolio, est_MVO_R, '-ko', 'LineWidth', 2) % Plot the estimated MVO efficient frontier
plot(std_actual_MVO, actual_MVO_R, '-kx', 'LineWidth', 2) % Plot the actual MVO efficient frontier
plot(std_true, true_MVO_R, '-kd', 'LineWidth', 2) % Plot the true MVO efficient frontier
plot(std_box_95, est_RMVO_95_box_R, '-bo', 'LineWidth', 2) % Plot the estimated box RMVO 95% efficient frontier
plot(std_box_90, est_RMVO_90_box_R, '-mo', 'LineWidth', 2) % Plot the estimated box RMVO 90% efficient frontier
plot(std_actual_95_box, actual_95_box_R, '-bx', 'LineWidth', 2) % Plot the actual box RMVO 95% efficient frontier
plot(std_actual_90_box, actual_90_box_R, '-mx', 'LineWidth', 2) % Plot the actual box RMVO 90% efficient frontier
plot(std_est_ell_95, est_95_ell_R, '-co', 'LineWidth', 2) % Plot the estimated ellipse RMVO 95% efficient frontier
plot(std_est_ell_90, est_90_ell_R, '-ro', 'LineWidth', 2) % Plot the estimated ellipse RMVO 90% efficient frontier
plot(std_actual_95_ell, actual_95_ell_R, '-cx', 'LineWidth', 2) % Plot the actual ellipse RMVO 95% efficient frontier
plot(std_actual_90_ell, actual_90_ell_R, '-rx', 'LineWidth', 2) % Plot the actual ellipse RMVO 90% efficient frontier
hold off
title('Part C - All Efficient Frontiers in One Plot')
xlabel('Volatility (\sigma)')
ylabel('Rate of Return (r)')
legend('Estimated MVO', 'Actual MVO', 'True MVO', 'Estimated Box 95',...
    'Estimated Box 90', 'Actual Box 95', 'Actual Box 90',...
    'Estimated Ellipse 95', 'Estimated Ellipse 90', 'Actual Ellipse 95',...
    'Actual Ellipse 90', 'Location', 'northwest')
grid on

%% Results used in Report
% Print the Sample Mean
Table1 = array2table(round(100*mu', 3),...
    'VariableNames',{'Mean Monthly Returns (%)'},...
     'RowNames',{'AAPL' 'C' 'CAT' 'DIS' 'ED' 'F' 'IBM' 'JNJ' 'JPM'...
    'KO' 'MCD' 'MRO' 'NEM' 'PEP' 'PFE' 'T' 'VZ' 'WFC' 'WMT' 'XOM'}); 
disp(Table1)

% Print the Covariance Matrix
Table2 = array2table(Q,...
    'VariableNames',{'AAPL' 'C' 'CAT' 'DIS' 'ED' 'F' 'IBM' 'JNJ' 'JPM'...
    'KO' 'MCD' 'MRO' 'NEM' 'PEP' 'PFE' 'T' 'VZ' 'WFC' 'WMT' 'XOM'},...
    'RowNames',{'AAPL' 'C' 'CAT' 'DIS' 'ED' 'F' 'IBM' 'JNJ' 'JPM'...
    'KO' 'MCD' 'MRO' 'NEM' 'PEP' 'PFE' 'T' 'VZ' 'WFC' 'WMT' 'XOM'});
disp(Table2)

% Print the Risk Aversion and Market Portfolio
fprintf('Expected Market Return: %.6f\n', r_mkt)
fprintf('Market Portfolio Variance: %.6f\n', var_mkt)
fprintf('Risk Aversion Parameter: %.6f\n\n', lambda_mkt)

% Plot the Market Portfolio
figure(2)
stocks = {'AAPL','C','CAT','DIS','ED','F','IBM','JNJ','JPM',...
   'KO','MCD','MRO','NEM','PEP','PFE','T','VZ','WFC','WMT','XOM'};
bar(x_mkt)
set(gca, 'XTickLabel',stocks,'xtick',1:numel(stocks))
xlabel('Stock Ticker')
ylabel('Portfolio Weighting')
title('The Market Portfolio')

% Plot the MVO Portfolio
figure(3)
bar(x_MVO)
set(gca, 'XTickLabel',stocks,'xtick',1:numel(stocks))
xlabel('Stock Ticker')
ylabel('Portfolio Weighting')
title('Mean-Variance Optimization Portfolio')

% Plot the RMVO Box Portfolio
figure(4)
bar([x_RMVO_box_95(1:20), x_RMVO_box_90(1:20)])
set(gca, 'XTickLabel',stocks,'xtick',1:2*numel(stocks))
xlabel('Stock Ticker')
ylabel('Portfolio Weighting')
title('Robust MVO Box Portfolios')
legend('95% Confidence', '90% Confidence', 'Location', 'southeast')

% Plot the RMVO Ellipsoid Portfolio
figure(5)
bar([x_RMVO_ell_95, x_RMVO_ell_90])
set(gca, 'XTickLabel',stocks,'xtick',1:2*numel(stocks))
xlabel('Stock Ticker')
ylabel('Portfolio Weighting')
title('Robust MVO Ellipsoid Portfolios')
legend('95% Confidence', '90% Confidence', 'Location', 'southeast')

% Plot the Risk-Parity Portfolio
figure(6)
bar(x_RP(2:21))
set(gca, 'XTickLabel',stocks,'xtick',1:numel(stocks))
xlabel('Stock Ticker')
ylabel('Portfolio Weighting')
title('Risk-Parity Portfolio')

% Summarize the Optimal Portfolios
sums = [sum(x_mkt), sum(x_MVO), sum(x_RMVO_box_95(1:20)),...
    sum(x_RMVO_box_90(1:20)), sum(x_RMVO_ell_95), sum(x_RMVO_ell_90),...
    sum(x_RP(2:21))];
x_total = [x_mkt, x_MVO, x_RMVO_box_95(1:20), x_RMVO_box_90(1:20),...
    x_RMVO_ell_95, x_RMVO_ell_90, x_RP(2:21)'; sums];
Table3 = array2table(x_total,...
    'VariableNames',{'Market Portfolio Weight' 'MVO Weight'...
    'RMVO Box 95 Weight' 'RMVO Box 90 Weight' 'RMVO Ellipsoid 95 Weight'...
    'RMVO Ellipsoid 90 Weight' 'Risk-Parity Weight'},...
    'RowNames',{'AAPL' 'C' 'CAT' 'DIS' 'ED' 'F' 'IBM' 'JNJ' 'JPM'...
    'KO' 'MCD' 'MRO' 'NEM' 'PEP' 'PFE' 'T' 'VZ' 'WFC' 'WMT' 'XOM' 'Total'});
disp(Table3)

% Display Part a) Answers
r_oct_total = [r_mkt_oct; r_MVO_oct; r_RMVO_box_95_oct; r_RMVO_box_90_oct;...
    r_RMVO_ell_95_oct; r_RMVO_ell_90_oct; r_RP_oct];
var_oct_total = [var_mkt_oct; var_MVO_oct; var_RMVO_box_95_oct;...
    var_RMVO_box_90_oct; var_RMVO_ell_95_oct; var_RMVO_ell_90_oct;...
    var_RP_oct];
std_oct_total = [std_mkt_oct; std_MVO_oct; std_RMVO_box_95_oct;...
    std_RMVO_box_90_oct; std_RMVO_ell_95_oct; std_RMVO_ell_90_oct;...
    std_RP_oct];
sharpe_oct_total = [sharpe_mkt_oct; sharpe_MVO_oct; sharpe_RMVO_box_95_oct;...
    sharpe_RMVO_box_90_oct; sharpe_RMVO_ell_95_oct; sharpe_RMVO_ell_90_oct;...
    sharpe_RP_oct];
oct_total = [r_oct_total, var_oct_total, std_oct_total, sharpe_oct_total];
Table4 = array2table(oct_total,...
    'VariableNames',{'Portfolio Return' 'Portfolio Variance'...
    'Portfolio Standard Deviation' 'Sharpe Ratio'},...
    'RowNames',{'Market Portfolio' 'MVO' 'RMVO Box 95'...
    'RMVO Box 90' 'RMVO Ellipsoid 95' 'RMVO Ellipsoid 90' 'Risk-Parity'});
disp(Table4)

% Display return of each portfolio broken down by asset for Oct 2008
oct_return_breakdown = [x_total(1:20, :).*mu_oct'; r_oct_total'];
Table5 = array2table(oct_return_breakdown,...
    'VariableNames',{'Market Portfolio Return' 'MVO Return'...
    'RMVO Box 95 Return' 'RMVO Box 90 Return' 'RMVO Ellipsoid 95 Return'...
    'RMVO Ellipsoid 90 Return' 'Risk-Parity Return'},...
    'RowNames',{'AAPL' 'C' 'CAT' 'DIS' 'ED' 'F' 'IBM' 'JNJ' 'JPM'...
    'KO' 'MCD' 'MRO' 'NEM' 'PEP' 'PFE' 'T' 'VZ' 'WFC' 'WMT' 'XOM' 'Total'});
disp(Table5)

% Display Part b) Answers
r_nov_total = [r_mkt_nov; r_MVO_nov; r_RMVO_box_95_nov; r_RMVO_box_90_nov;...
    r_RMVO_ell_95_nov; r_RMVO_ell_90_nov; r_RP_nov];
var_nov_total = [var_mkt_nov; var_MVO_nov; var_RMVO_box_95_nov;...
    var_RMVO_box_90_nov; var_RMVO_ell_95_nov; var_RMVO_ell_90_nov;...
    var_RP_nov];
std_nov_total = [std_mkt_nov; std_MVO_nov; std_RMVO_box_95_nov;...
    std_RMVO_box_90_nov; std_RMVO_ell_95_nov; std_RMVO_ell_90_nov;...
    std_RP_nov];
sharpe_nov_total = [sharpe_mkt_nov; sharpe_MVO_nov; sharpe_RMVO_box_95_nov;...
    sharpe_RMVO_box_90_nov; sharpe_RMVO_ell_95_nov; sharpe_RMVO_ell_90_nov;...
    sharpe_RP_nov];
nov_total = [r_nov_total, var_nov_total, std_nov_total, sharpe_nov_total];
Table6 = array2table(nov_total,...
    'VariableNames',{'Portfolio Return' 'Portfolio Variance'...
    'Portfolio Standard Deviation' 'Sharpe Ratio'},...
    'RowNames',{'Market Portfolio' 'MVO' 'RMVO Box 95'...
    'RMVO Box 90' 'RMVO Ellipsoid 95' 'RMVO Ellipsoid 90' 'Risk-Parity'});
disp(Table6)

% Display return of each portfolio broken down by asset for Nov 2008
nov_return_breakdown = [x_total(1:20, :).*mu_nov'; r_nov_total'];
Table7 = array2table(nov_return_breakdown,...
    'VariableNames',{'Market Portfolio Return' 'MVO Return'...
    'RMVO Box 95 Return' 'RMVO Box 90 Return' 'RMVO Ellipsoid 95 Return'...
    'RMVO Ellipsoid 90 Return' 'Risk-Parity Return'},...
    'RowNames',{'AAPL' 'C' 'CAT' 'DIS' 'ED' 'F' 'IBM' 'JNJ' 'JPM'...
    'KO' 'MCD' 'MRO' 'NEM' 'PEP' 'PFE' 'T' 'VZ' 'WFC' 'WMT' 'XOM' 'Total'});
disp(Table7)

% Display the mean expected return and the true return of each asset for
% Oct 2008
Table8 = array2table(100*[mu; mu_oct]',...
    'VariableNames',{'Expected Asset Returns (%)' 'True Asset Returns (%)'},...
    'RowNames',{'AAPL' 'C' 'CAT' 'DIS' 'ED' 'F' 'IBM' 'JNJ' 'JPM'...
    'KO' 'MCD' 'MRO' 'NEM' 'PEP' 'PFE' 'T' 'VZ' 'WFC' 'WMT' 'XOM'});
disp(Table8)