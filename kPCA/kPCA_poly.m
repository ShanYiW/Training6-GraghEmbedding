function [Alpha] = kPCA_poly(X, p)
% kernel PCA (polynomial kernel)
% X: D*N
% p: 多项式的阶数
%% 原数据集 中心化
N = size(X,2); % 样本数
% mX = mean(X,2); % N*1
% X = X - repmat(mX, [1,N]);
%% 构造核函数
K = (X'*X).^p; % N*N p阶多项式核
%% 核函数中心化
K2 = sum(K,2); % N*1
Ko = K - (repmat(K2, [1,N])' + repmat(K2, [1,N]))./N + sum(sum(K))/(N*N);
%%
[Alpha, v] = eig(Ko); % N*N  
v=diag(v); v=v(end:-1:1); %v=v./sum(v);
Alpha = Alpha(:,end:-1:1); 
Alpha = Alpha*diag(1./sqrt(v)); % 让||alpha_i||_2^2 = 1 -> 1/vi (符合理论)
return;

