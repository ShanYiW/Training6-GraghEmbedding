clear;
load("USPS.mat"); % fea:N*D  gnd:N*1
d = [32 64 128 256]; len_d = length(d);
seed = [0:6]; len_sd = length(seed);
Ntest = 2007;
Ntrain = 7291;
Nc = length(unique(gnd)); % 类别数
fea = fea'; % N*D -> D*N
gndtest = gnd(Ntrain+1:end);
%%
Acc_arr = zeros([len_d,len_sd]);
for sd = 1:len_sd
    rand('seed', seed(sd));
    chosen_train = randperm(Ntrain, 3000); % 从1到Ntrain的3000个不重复数
    chosen_train = sort(chosen_train); % 升序
    gndtrain = gnd(chosen_train);
    
    Xtr_ori = fea(:,chosen_train); % D*3000
    mX = mean(Xtr_ori, 2); % D*1
    Xtr_ori = Xtr_ori - mX; % 中心化
    [U,~,~] = svd(Xtr_ori); % U: D*D
    for i=1:len_d
        fprintf('seed=%d  d=%d: ', seed(sd), d(i));
        Xtrain = U(:,1:d(i))'*fea(:,chosen_train);
        Xtest  = U(:,1:d(i))'*fea(:,Ntrain+1:end);
%         varXtr = sum(Xtrain.*Xtrain, 2)./3000;
%         varXte = sum(Xtest.*Xtest, 2)./Ntest;
%         Xtrain = diag(1./sqrt(varXtr))*Xtrain;
%         Xtest = diag(1./sqrt(varXte))*Xtest;
        %% OVR SVM
        Est_each_model = zeros([Nc, Ntest]); % j列: 各个模型对j号测试样本的分类结果
        t_start = clock;
        for c=1:Nc
            idx_c = gndtrain==c;
            y = -1.*ones([3000, 1]);
            y(idx_c) = 1;
            SVM = fitcsvm(Xtrain', y);
            w = SVM.Beta; % 超平面法向量
            w2 = sqrt(sum(w.*w)); % ||w||_2
            b = SVM.Bias; % 超平面截距
            estimate = (w'*Xtest + b)./w2; % 1*Ntest
            Est_each_model(c,:) = estimate;
        end

        Est = zeros([Ntest,1]);
        for j=1:Ntest
            estj = 1; max = Est_each_model(estj, j);
            for c=2:Nc
                test_c = Est_each_model(c, j);
                if test_c > max
                    max = test_c;
                    estj = c;
                end
            end
            Est(j) = estj;
        end
        t_end = clock;
        Acc_arr(i,sd) = sum(Est==gndtest)/Ntest;
        fprintf('Acc = %f\n', Acc_arr(i,sd));
        fprintf('Time used = %f\n', etime(t_end, t_start));
    end
end
Acc = mean(Acc_arr,2);
