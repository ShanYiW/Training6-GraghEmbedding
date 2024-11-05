clear;
load("PIE_C27_32x32.mat"); % PIE, gnd
N = length(gnd);
%% 每类随机抽30张图象 (第1张给标签)
C = 68; % 类别数
Nc = 30; % 每类样本数
Ntrain = C*Nc;
Ntest = N - Ntrain;
rand('seed', 7);

Nrnd = 20;
Acc_tr_max = zeros([Nrnd,1]); Acc_te_max = zeros([Nrnd,1]);
for rnd = 1:Nrnd
    chosen_train = zeros([C,Nc]);
    lb = 0; step = 49;
    chosen = randperm(step, Nc); % 从1~step中抽Nc个整数
    for p = 1:C
        chosen_train(p,:) = chosen + lb;
        lb = lb + step;
        if p == 38
            lb = lb - step;
            chosen = randperm(46, Nc); % 从1~step中抽Nc个整数
            chosen_train(p,:) = chosen + lb;
            lb = lb + 46;
        end
    end
    chosen_train = chosen_train(:); % 列优先 拉伸成列向量
    %% 构造训练集, 测试集
    Xtrain = PIE(:,chosen_train); % 带标签的排前1~C
    gndtrain = gnd(chosen_train);
    is_test = true([N, 1]);
    is_test(chosen_train) = false;
    gndtest = gnd(is_test);
    Xtest = PIE(:,is_test); % D*Ntest
    %% PCA
    mX = mean(Xtrain);
    [U,~,~] = svd(Xtrain - mX); % U:D*D

    d_lb = 1; d_ub = 200; d_len = d_ub-d_lb+1;
    Acc_tr_rnd = zeros([d_len,1]);
    Acc_te_rnd = zeros([d_len,1]);
    for d=d_lb:d_ub
        %%
        Ytr = U(:,1:d)'*Xtrain; % D*Ntr
        Xlab2  = sum(Ytr(:,1:C).*Ytr(:,1:C), 1); % 1*C
        Xulab2 = sum(Ytr(:,C+1:Ntrain).*Ytr(:,C+1:Ntrain), 1); % 1*(Ntr-C)
        dis_tr = repmat(Xlab2', [1, Ntrain-C]) + repmat(Xulab2, [C,1]) - 2.*Ytr(:,1:C)'*Ytr(:,C+1:Ntrain); 
        [~,idx] = sort(dis_tr);
        pred = gndtrain(idx(1,:)');
        Acc_tr_rnd(d-d_lb+1) = sum(pred==gndtrain(C+1:Ntrain))/(Ntrain - C);
        %% 测试样本
        Yte = U(:,1:d)'*Xtest; % D*Nte
        Xte2 = sum(Yte.*Yte, 1); % 1*Ntest
        dis_te = repmat(Xte2, [C,1]) + repmat(Xlab2', [1,Ntest]) - 2.*Ytr(:,1:C)'*Yte;
        [~, idx_te] = sort(dis_te);
        pred = gndtrain(idx_te(1,:)'); % 1*Ntest
        Acc_te_rnd(d-d_lb+1) = sum(pred==gndtest)/Ntest;
    end
    Acc_tr_max(rnd) = max(Acc_tr_rnd);
    Acc_te_max(rnd) = max(Acc_te_rnd);
end
fprintf('%f±%f\n%f±%f\n', mean(Acc_tr_max), std(Acc_tr_max), mean(Acc_te_max), std(Acc_te_max));
