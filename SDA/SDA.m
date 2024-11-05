function [A] = SDA(X, gnd, k, alpha, beta)
% X: D*N
% gnd: L*1 L个样本标签  N-L个未标注样本
% k: 近邻数
% A: D*N
[D,N] = size(X); % 样本数
L = length(gnd); % 标注的样本数
%% 准备 G, P  M = XG  Xo = X(I-P)
cls = unique(gnd); % 类别标签
C = length(cls); % 类别数
Nc = zeros([C,1]); % 各类的样本数
G = zeros([L,C]);
% P = zeros([L,L]); % 
for c=1:C
    idx_c = gnd==cls(c); % L*1
    Nc(c) = sum(idx_c);
    G(:,c) = sqrt(Nc(c)).*(ones([L,1]).*idx_c./Nc(c) - ones([L,1])./L);
%     P(:,idx_c) = repmat(-1.*ones([L,1]).*idx_c./Nc(c), [1,Nc(c)]); % L*Nc(c)
end
% P = P + diag(ones([L,1])); % L*L
P = eye(L) - ones([L,L])./L;
%% 构造Laplacian矩阵
X2 = sum(X.*X, 1); % 1*N, [||X_1||2^2 ... ||X_N||2^2 ]
dist = repmat(X2, [N,1]) + repmat(X2', [1,N]) - 2.*X'*X; 
[~, nei_idx] = sort(dist);
nei_idx = nei_idx(2:k+1, :); % k*N
Adj = false([N,N]);
for j = 1:N
    Adj(nei_idx(:,j), j) = true;
    Adj(j, nei_idx(:,j)) = true;
end
W = zeros([N,N]); 
W(Adj) = 1; % Naive
Dvec = sum(W,2); % N*1
%% L = Evv'E'
I = [eye(L); zeros([N-L,L])]; % N*L
IP = I*P; % N*L
T = IP*IP'+ alpha.*(diag(Dvec) - W); % N*N (对称)
T = (T + T')./2;
[E,v] = eig(T);
v = diag(v);
v(v<0) = eps;
v = sqrt(v);
%% Trick
% [U,S,~] = svd(X*E*diag(v)); % XEv = USS'U'  U:D*D  S:D*N
% [row, col] = size(S);
% S = diag(S);
% invS = zeros([col, row]);
% if row > col
%     invS(:,1:col) = diag(1./S); % col*col
% else % row <= col
%     invS(1:row, :) = diag(1./S); % row*row
% end
% Core_half = invS*U'*X*I*G; % N*L
% [Evec, ~] = svd(Core_half);
% A = U*invS'*Evec;
%% No trick
Sb_half = X*I*G; % D*C
St_half = X*E*diag(v); % D*N
Sb = Sb_half*Sb_half';
St = St_half*St_half' + beta.*eye(D);
[A,ev] = eig(Sb, St); % ev = diag(ev);ev=ev(end:-1:1);
A = A(:,end:-1:1);
return;