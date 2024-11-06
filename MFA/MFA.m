function [E] = MFA(X, gnd, ki, ko)
% X: D*N 数据. D:维数  N:样本数
% k: kNN  t: 热核函数的参数
% E: D*d 投影
[D,N] = size(X);
cls_label = unique(gnd); % 类别标签
C = length(cls_label); % 类别数
%% PCA降维 预处理
mX = sum(X,2)./N; % D*1
Xo = X - repmat(mX, [1,N]);
[Wpca,~,~] = svd(Xo); % D*D
Wpca = Wpca(:,1:min(D,N-C)); % D*d  (d=N-C)
X = Wpca'*X; % D*N -> d*N
%% 
X2 = sum(X.*X, 1); % 1*N
dist = repmat(X2, [N,1]) + repmat(X2', [1,N]) - 2.*X'*X; % N*N
%% 对于每一类, 构造其与同类样本的邻接矩阵, 其与异类样本的邻接矩阵
Adj_i = false([N,N]); % 同类样本的邻接矩阵
Adj_o = false([N,N]); % 异类样本的邻接矩阵
for i=1:C
    Xi_idx = gnd==cls_label(i); % N*1, T/F 第i类的样本的编号
    Xo_idx = ~Xi_idx; % 不是第i类的样本编号
    Ni = sum(Xi_idx); % 第i类样本数
    No = N - Ni; % 不是第i类的样本数
    
    glb_idx_i = zeros([Ni,1]); glb_idx_o = zeros([No,1]);
    j=1; % 构造 第i类 类内编号->全局编号的映射 glb_idx_i
    for k=1:Ni
        while ~Xi_idx(j) % Xc_idx(k) == 0
            j = j + 1; % 若j号样本不是i类，则检查下一个
        end
        glb_idx_i(k) = j; % 第k个i类的样本 在All样本中 下标为j号
        j = j + 1;
    end
    j=1; % 构造 非第i类编号->全局编号的映射 glb_idx_o
    for k=1:No
        while ~Xo_idx(j) % Xc_idx(k) == 0
            j = j + 1;
        end
        glb_idx_o(k) = j;
        j = j + 1;
    end
    dist_i = dist(Xi_idx, Xi_idx); % Ni*Ni
    [~, nei_idx_i] = sort(dist_i); % Ni*Ni
    nei_idx_i = nei_idx_i(2:min(ki+1, Ni),:); % ki*Ni 不包括自己 若类内样本不足ki+1, 则全做邻居
    dist_o = dist(Xo_idx, Xi_idx); % No*Ni
    [~, nei_idx_o] = sort(dist_o); % No*Ni
    nei_idx_o = nei_idx_o(1:min(ko, No),:); % ko*Ni
    for k=1:Ni % 对于第i类的每个样本
        Adj_i(glb_idx_i(nei_idx_i(:,k)), glb_idx_i(k)) = true;
        Adj_i(glb_idx_i(k), glb_idx_i(nei_idx_i(:,k))) = true;
        Adj_o(glb_idx_o(nei_idx_o(:,k)), glb_idx_i(k)) = true;
        Adj_o(glb_idx_i(k), glb_idx_o(nei_idx_o(:,k))) = true;
    end
end
%% 构造 邻接矩阵Wi, Wo, 构造拉普拉斯Li, Lo
Wi = zeros([N,N]); Wo = zeros([N,N]);
Wi(Adj_i) = 1; % N*N
Wo(Adj_o) = 1; % N*N
Di = sum(Wi, 2); % N*1
Do = sum(Wo, 2); 
Li = diag(Di) - Wi; % N*N 类内 rank(Li)=N-c, 因为有c个连通分量
Lo = diag(Do) - Wo; % 类间  Lo正定  rank(Lo)=N-1
XLiXt = X*Li*X'; % d*d
XLiXt = (XLiXt + XLiXt')./2;%max(XLiXt, XLiXt');
XLoXt = X*Lo*X'; % d*d  正定  rank(XLoXt) = N-1
XLoXt = (XLoXt + XLoXt')./2;%max(XLoXt, XLoXt');
%% 直接eig(广义特征值分解)会出虚特征值, 必须用trick
% [E, v] = eig(XLiXt, XLoXt); % Evec: D*D  v=diag(v);
% E = Wpca*E; % (D*d) * (d*d) = D*d
%% max Tr(PX Lo X'P'), s.t. PX Li X'P'=I
% [Q,S] = eig(XLiXt); % Q,S: D*D
% S = diag(S); % 默认升序
% Positive = S>0; lenPos = sum(Positive); S(~Positive) = 0.1*S(lenPos);
% S = sqrt(S);
% invS = diag(1./S); 
% M = invS*Q'*XLoXt*Q*invS';
% M = (M+M')./2;%max(M,M'); % D*D
% [Evec, ~] = eig(M);
% Evec = Evec(:, end:-1:1); 
% E = Q*invS'*Evec; % D*D
% E = Wpca*E;
%% min Tr(PX Li X'P'), s.t. PX Lo X'P'=I
[Q,S] = eig(XLoXt); % Q,S: d*d
S = diag(S); % d*1  默认升序
Positive = S>0; lenPos = sum(Positive); S(~Positive) = 0.1*S(lenPos);
invS = diag(1./sqrt(S)); % d*d
M = invS*Q'*XLiXt*Q*invS'; % d*d
M = max(M,M'); % d*d
[Evec, ~] = eig(M); % d*d
E = Q*invS'*Evec; % d*d
E = Wpca*E; % (D*d) * (d*d) = D*d

return