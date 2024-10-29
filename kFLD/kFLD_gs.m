function [Alpha] = kFLD_gs(X, gnd, sig2)
% Input:
% X:D*N  D维数  N样本数
% gnd:N*1  样本的类别标签
% sig2:1*1  高斯核的带宽
% Output:
% Alpha:N*d  输出的投影

%% 准备 G, P
cls = unique(gnd); % 类别标签
C = length(cls); % 类别数
Nc = zeros([C,1]); % 各类的样本数
N = size(X,2); % 样本数
G = zeros([N,C]);
P = zeros([N,N]); % 
for c=1:C
    idx_c = gnd==cls(c); % N*1
    Nc(c) = sum(idx_c);
    G(:,c) = ones([N,1]).*idx_c./Nc(c) - ones([N,1])./N;
    P(:,idx_c) = repmat(-1.*ones([N,1]).*idx_c./Nc(c), [1,Nc(c)]); % N*Nc(c)
end
P = P + diag(ones([N,1])); % 不是[N,N]!!!
%% 准备 核函数矩阵
X2 = sum(X.*X,1); % 1*N
dist = repmat(X2, [N,1]) + repmat(X2, [N,1])' - 2.*X'*X;
K = exp(-dist./sig2);
[~,Lmd] = eig(K); 
Lmd = diag(Lmd); 
Lmd = Lmd(end:-1:1);
Lmd(Lmd<0) = eps;
%% trick
[U,S,~] = svd(K*P); % U,S: N*N
invS = diag(1./diag(S)); % N*N
Half = invS*U'*K*G; % N*C
[Evec, v] = svd(Half); % v=diag(v);
Alpha = U*invS'*Evec; % N*N
%% 放缩 ||Alpha_i||_2^2 -> 1/Lmd_i, 1<= i <=N
Alp2 = sum(Alpha.*Alpha,1)'; % N*1
Alpha = Alpha*diag(1./sqrt(Alp2)); % ||Alpha_i||_2^2 = 1
Alpha = Alpha*diag(1./sqrt(Lmd)); % ||Alpha_i||_2^2 = 1/Lmd_i

return;