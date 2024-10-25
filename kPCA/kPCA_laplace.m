function [E] = kPCA_laplace(X, sig)
% kernel PCA (polynomial kernel)
% X: D*N
% sig: Laplace 核函数的参数
N = size(X,2); % 样本数
%% Laplace 核函数
X2 = sum(X.*X, 1); % 1*N
dist = repmat(X2, [N,1]) * repmat(X2, [N,1])' - 2.*X'*X; % N*N dist(i,j)=||xi-xj||2^2
K = exp(-dist./sig); 
%% 核函数中心化
K2 = sum(K,2); % N*1
Ko = K - (repmat(K2, [1,N])' + repmat(K2, [1,N]))./N + sum(sum(K))/(N*N);
%% kernel PCA
[E, v] = eig(Ko); % N*N  v=diag(v); v=v(end:-1:1); v=v./sum(v);
% E = E(:,end:-1:1); 

return;

