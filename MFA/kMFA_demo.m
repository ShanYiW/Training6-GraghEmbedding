clear;
% load('ORL_12_area.mat');
% DATA = ORL; clear ORL;
% C = 40; Nc = 10; % 每类的样本数
load('Yale_64.mat');
DATA = fea'; clear fea;
C = 15; Nc = 11;
[D,N] = size(DATA);
%%
L = 3;
Ntrain = L*C; % 训练样本数
Ntest = N-Ntrain;
trainset = zeros([D, Ntrain]); % 训练集
testset  = zeros([D, Ntest]); % 测试集
gndtrain = zeros([Ntrain,1]);
gndtest  = zeros([Ntest, 1]); % gnd_test(i)=第i个测试样本的类别

lb_d=2; ub_d=Ntrain; len_d = ub_d-lb_d+1;
lb_ki=1; ub_ki=L-1; s_ki=1; len_ki = (ub_ki-lb_ki)/s_ki + 1; % L=2
lb_ko=2; ub_ko=32; s_ko=2; len_ko = floor((ub_ko-lb_ko)/s_ko) + 1;

N_rnd = 20; % 
rand('seed', 6);
acc_rnd_d = zeros([len_ki, len_ko, N_rnd, len_d]); % 
for rnd = 1:N_rnd
%% 生成训练集, 测试集
    train_idx = sort(randperm(Nc,L)); % 升序排序
    test_idx = setdiff(1:Nc,train_idx); % 集合作差: 全集-train_idx
    i_tr = 0; i_te = 0;
    for j=1:C % 40个人
        trainset(:,i_tr+1:i_tr+L) = DATA(:, [(j-1)*Nc + train_idx]); % 不能[(j-1)*N_per_class +train_idx]
        gndtrain(i_tr+1:i_tr+L) = gnd([(j-1)*Nc + train_idx]);
        i_tr = i_tr+L;
        testset(:,i_te+1:i_te+Nc-L) = DATA(:,[(j-1)*Nc + test_idx]);
        gndtest(i_te+1:i_te+Nc-L) = gnd([(j-1)*Nc + test_idx]);
        i_te = i_te+Nc-L;
    end
    Tr2 = sum(trainset.*trainset, 1); % 1*Ntr
    dis_tr = repmat(Tr2, [Ntrain,1]) + repmat(Tr2', [1,Ntrain]) - 2.*trainset'*trainset; % Ntr*Ntr
    Te2 = sum(testset.*testset, 1); % 1*Nte
    dis_te = repmat(Te2, [Ntrain,1]) + repmat(Tr2', [1,Ntest]) - 2.*trainset'*testset; % Ntr*Nte
    for sig2 = [2e9]
    K = exp(-dis_tr./sig2); % 高斯核
    Ke = exp(-dis_te./sig2); % 
    for ki=lb_ki:s_ki:ub_ki
    for ko=lb_ko:s_ko:ub_ko
        %% 运行 MFA, 输出E
        [Alpha] = kMFA(trainset, gndtrain, ki,ko, sig2); % E: D*N
        %% 计算分类准确度
        for d=lb_d:ub_d
            Ytrain = Alpha(:,1:d)'*K;
            Ytest  = Alpha(:,1:d)'*Ke;
            acc = 0;
            for j=1:Ntest
                y = Ytest(:,j); % d*1
                dist = sum((repmat(y,[1,Ntrain]) - Ytrain).^2, 1);
                Mx_i = 1; Mx = dist(1);
                for t=2:Ntrain
                    if dist(t) < Mx
                        Mx = dist(t);
                        Mx_i = t;
                    end
                end
                pred = gndtrain(Mx_i); % 最近邻的类别 = j号测试样本的类别
                acc = acc + (pred==gndtest(j));
            end
            acc = acc/Ntest; % 
            acc_rnd_d((ki-lb_ki)/s_ki+1, (ko-lb_ko)/s_ko+1, rnd, d-lb_d+1) = acc; % 每次随机测试的准确度 <- 所有测试样本
        end
    end
    end
    end
end
Acc_kkd = mean(acc_rnd_d, 3); % len_ki * len_ko * len_d
Acc_kkd = squeeze(Acc_kkd);
[max,max_i] = max(Acc_kkd(:));
max_d = floor((max_i-1)/(len_ki*len_ko)) + lb_d;
max_ki= lb_ki + mod(mod(max_i-1,len_ki*len_ko),len_ki);
max_ko= lb_ko + floor(mod(max_i-1,len_ki*len_ko)/len_ki)*s_ko;
fprintf('Max d=%d  ki=%d  ko=%d  sig2=%g  acc=%f\n', max_d, max_ki, max_ko, sig2, max );

