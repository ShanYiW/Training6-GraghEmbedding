function [E] = kPCA_sigmod(X, beta, theta)
% X: D*N
% beta, theta: paras
N = size(X,2); % 样本数
%% Sigmod 核函数
K = tanh(X'*X.*beta + theta); % N*N
%% 核函数中心化
K2 = sum(K,2); % N*1
Ko = K - (repmat(K2, [1,N])' + repmat(K2, [1,N]))./N + sum(sum(K))/(N*N);
%% kernel PCA
[E, ~] = eig(Ko); % N*N  v=diag(v); v=v(end:-1:1); v=v./sum(v);
E = E(:,end:-1:1); 

return;