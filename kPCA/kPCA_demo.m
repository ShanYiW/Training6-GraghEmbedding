clear;
load("USPS.mat"); % fea:N*D  gnd:N*1
d = [32 64 128 256 512 1024 2048]; len_d = length(d);
p = [2:7]; len_p = length(p);
seed = [0 1 2]; len_sd = length(seed);
Ntest = 2007;
Ntrain = 7291;
Nc = length(unique(gnd)); % 类别数
fea = fea'; % N*D -> D*N
gndtest = gnd(Ntrain+1:end);
%% PCA
% m = mean(fea, 2); % D*1
% fea_o = fea - m;
% [E,~] = eig(fea_o*fea_o'); % D*D
% E = E(:,end:-1:1);
% Xtest  = E(:,1:d)'*fea(:,Ntrain+1:end); % d*2007
% Xtrain = E(:,1:d)'*fea(:,1:Ntrain); % d*7291
%%
Acc_arr = zeros([len_d,len_p,len_sd]);
for sd = 1:len_sd
    rand('seed', seed(sd));
    chosen_train = randperm(Ntrain, 3000); % 从1到Ntrain的3000个不重复数
    chosen_train = sort(chosen_train); % 升序
    gndtrain = gnd(chosen_train);
    for k = 1:len_p
        [Alpha] = kPCA_poly(fea(:,chosen_train), p(k)); % kPCA_poly
        for i=1:len_d
            fprintf('seed=%d  d=%d  p=%d: ', seed(sd), d(i), p(k));
            Xtrain = Alpha(:,1:d(i))'*(fea(:,chosen_train)'*fea(:,chosen_train)).^p(k);%diag(1./sqrt(v(1:d(i))))*
            Xtest = Alpha(:,1:d(i))'*(fea(:,chosen_train)'*fea(:,Ntrain+1:end)).^p(k);%diag(1./sqrt(v(1:d(i))))*
            varXtr = sum(Xtrain.*Xtrain, 2)./3000;
            varXte = sum(Xtest.*Xtest, 2)./Ntest;
            Xtrain = diag(1./sqrt(varXtr))*Xtrain;
            Xtest = diag(1./sqrt(varXte))*Xtest;
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
            Acc_arr(i,k,sd) = sum(Est==gndtest)/Ntest;
            fprintf('Acc = %f\n', Acc_arr(i,k,sd));
            fprintf('Time used = %f\n', etime(t_end, t_start));
        end
    end
end
Acc = mean(Acc_arr,3);
% fprintf('p=%d  d=%d  Avg. acc=%f\n',p,d(i), mean(Acc_arr));