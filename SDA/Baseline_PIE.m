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
Acc_tr = zeros([Nrnd,1]); Acc_te = zeros([Nrnd,1]);
for rnd = 1:Nrnd
    chosen_train = zeros([C,Nc]);
    lb = 0; step = 49;
    for p = 1:C
        chosen = randperm(step, Nc); % 从1~step中抽Nc个整数
        chosen_train(p,:) = chosen + lb;
        lb = lb + step;
        if p == 38
            lb = lb - step;
            chosen = randperm(46, Nc);
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
    %% Nearest Neighbor
    Xlab2  = sum(Xtrain(:,1:C).*Xtrain(:,1:C), 1); % 1*C
    Xulab2 = sum(Xtrain(:,C+1:Ntrain).*Xtrain(:,C+1:Ntrain), 1); % 1*(Ntr-C)
    dis_tr = repmat(Xlab2', [1, Ntrain-C]) + repmat(Xulab2, [C,1]) - 2.*Xtrain(:,1:C)'*Xtrain(:,C+1:Ntrain); 
    [~,idx] = sort(dis_tr);
    pred = gndtrain(idx(1,:)');
    Acc_tr(rnd) = sum(pred==gndtrain(C+1:Ntrain))/(Ntrain - C);
    % Xtr2 = sum(Xtrain.*Xtrain, 1); % 1*Ntrain
    % dis_tr = repmat(Xtr2, [Ntrain, 1]) + repmat(Xtr2, [Ntrain, 1])' - 2.*Xtrain'*Xtrain;
    % [~,idx_tr] = sort(dis_tr);
    % pred = gndtrain(idx_tr(2,C+1:Ntrain));
    % Acc_tr = sum(pred==gndtrain(C+1:Ntrain))/(Ntrain - C);
    %% 测试样本
    Xte2 = sum(Xtest.*Xtest, 1); % 1*Ntest
    dis_te = repmat(Xte2, [C,1]) + repmat(Xlab2', [1,Ntest]) - 2.*Xtrain(:,1:C)'*Xtest;
    % dis_te = repmat(Xte2, [Ntrain, 1]) + repmat(Xtr2', [1,Ntest]) - 2.*Xtrain'*Xtest;
    [~, idx_te] = sort(dis_te);
    pred = gndtrain(idx_te(1,:)'); % 1*Ntest
    Acc_te(rnd) = sum(pred==gndtest)/Ntest;
end
fprintf('%f±%f\n%f±%f\n', mean(Acc_tr), std(Acc_tr), mean(Acc_te), std(Acc_te));
