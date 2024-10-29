clear;
% mode = 'area';
% load(append("ORL_14_", mode)); % vector-LDE 用1/4下采样的ORL
% DATA = double(ORL); clear ORL;
load('Yale_64.mat');
DATA = fea'; clear fea; % X:D*N  gnd:N*1
[D,N] = size(DATA);
fold = N; % N-fold cross validation
Ntest = floor(N/fold);
Ntrain = N - Ntest;
rand('seed', 6); % 
cv = randperm(N)';
p = 2;
% NN1 = ones([Ntrain,Ntrain])./Ntrain; NNe1 = ones([Ntrain,Ntest])./Ntrain;
% sig2 = 3e8; % Yale 
% X2 = sum(DATA.*DATA, 1); % 1*N
% dist = repmat(X2, [N,1]) + repmat(X2, [N,1])' - 2.*DATA'*DATA;
% for sig2=31e6:1e6:40e6
%%
lb_d=2; ub_d=60; len_d = ub_d-lb_d+1;
Acc_cv = zeros([fold, len_d]);
% time0 = clock;
for f=1:fold
    is_test = false([N,1]);
    is_test((f-1)*Ntest+1:f*Ntest) = true;
    is_train = ~is_test;
    
    testset  = DATA(:, cv(is_test)); % D*Ntest
    gndtest  = gnd(cv(is_test)); % Ntest*1
    trainset = DATA(:, cv(is_train)); % D*Ntrain
    gndtrain = gnd(cv(is_train)); % Ntrain*1
    K = (trainset'*trainset).^p; % Ntr * Ntr
    Ke = (trainset'*testset).^p; % Ntr * Nte
%     K = exp(-dist(cv(is_train), cv(is_train))./sig2); % Ntr * Ntr
%     Ke = exp(-dist(cv(is_train), cv(is_test))./sig2); % Ntr * Nte
    %% kPCA
    [Alpha] = kPCA_poly(trainset, p);
%     [Alpha] = kPCA_gs(trainset, sig2);
    for d=lb_d:ub_d
%         Ytrain = Alpha(:,1:d)'*(K - K*NN1 - NN1*K + NN1*K*NN1); % 结果 与不中心完全一样
%         Ytest  = Alpha(:,1:d)'*(Ke - K*NNe1 - NN1*Ke + NN1*K*NNe1); % 
        Ytrain = Alpha(:,1:d)'*K;
        Ytest = Alpha(:,1:d)'*Ke;
        acc = 0;
        for j=1:Ntest
            y = Ytest(:,j); % d*1
            dis = sum((repmat(y,[1,N-Ntest]) - Ytrain).^2, 1);
            [~,idx] = sort(dis); % 距离 升序排
            pred = gndtrain(idx(1)); % 最近邻的类别 = j号测试样本的类别
            acc = acc + (pred==gndtest(j));
        end
        acc = acc/Ntest; % 
        Acc_cv(f, d-lb_d+1) = acc;
    end
end
% Time_used = etime(clock, time0);
% fprintf('Time: %f s\n', Time_used);
Acc_avg = mean(Acc_cv)'; % len_d*1
maxAcc = Acc_avg(1); maxidx = 1;
for j=2:len_d
    if Acc_avg(j)>maxAcc
        maxAcc = Acc_avg(j);
        maxidx = j;
    end
end
fprintf('Max: %f %d\n', maxAcc, maxidx);
% fprintf('sig2=%d  Max: %f %d\n',sig2, maxAcc, maxidx);
% figure;
% plot(lb_d:ub_d, Acc_avg, '^-', 'Color',[162,20,47]./255,...
%     'MarkerFaceColor','w','Linewidth',1.5);
% xlabel('Dims', 'Fontsize', 16);
% ylabel('Recognition rate (%)', 'Fontsize', 16);
% end
