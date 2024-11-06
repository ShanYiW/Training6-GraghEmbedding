clear;
load('ORL_12_area.mat');
DATA = ORL; clear ORL;
C = 40; Nc = 10; % 每类的样本数
% load('Yale_64.mat');
% DATA = fea'; clear fea;
% C = 15; Nc = 11;
[D,N] = size(DATA);
%%
L = 5;
Ntrain = L*C; % 训练样本数
Ntest = N-Ntrain;
trainset = zeros([D, Ntrain]); % 训练集
testset = zeros([D,N-Ntrain]); % 测试集
gndtrain = zeros([Ntrain,1]);
gndtest = zeros([Ntest,1]); % gnd_test(i)=第i个测试样本的类别

lb_d=2; ub_d=Ntrain-C; len_d = ub_d-lb_d+1;
lb_ki=1; ub_ki=4; s_ki=1; len_ki = (ub_ki-lb_ki)/s_ki + 1; % L=2
lb_ko=2; ub_ko=30; s_ko=2; len_ko = (ub_ko-lb_ko)/s_ko + 1;

N_rnd = 20; % 重复次数
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
    for ki=lb_ki:s_ki:ub_ki
    for ko=lb_ko:s_ko:ub_ko
        %% 运行 MFA, 输出E
        [E] = MFA(trainset, gndtrain, ki,ko); % E: D*N
        %% 计算分类准确度
        for d=lb_d:ub_d
            Ytrain = E(:,1:d)'*trainset;
            Ytest  = E(:,1:d)'*testset;
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
Acc_kkd = mean(acc_rnd_d, 3);
Acc_kkd = squeeze(Acc_kkd); % len_ki * len_ko * len_d
[max,max_i] = max(Acc_kkd(:));
max_d = floor(max_i/(len_ki*len_ko)) + lb_d;
max_ki= lb_ki + mod(mod(max_i,len_ki*len_ko)-1,len_ki);
max_ko= lb_ko + floor((mod(max_i,len_ki*len_ko)-1)/len_ki);
fprintf('Max acc = %f\n', max );
%% 画图
% Color = [237,177,32;
%     217,83,25;
%     255,153,200;
%     77,190,238;
%     162,20,47;
%     125,46,143;
%     119,172,48]./255;
% node_shape =['+-';'x-';'<-';'>-';'v-';'^-';'o-';'s-';'p-';'*-'; 'o:';'*:';'s:';'.-';];
% figure;
% for k=1:len_ko
%     plot(lb_ko:s_ko:ub_ko, Acc_kkd(k,:), node_shape(k,:), 'MarkerFaceColor','w','Linewidth', 1.5, 'Color', Color(k,:)); hold on;
% end
% xlabel('Dims', 'Fontsize', 16);
% ylabel('Recognition rate (%)', 'Fontsize', 16);
% legend('$k_i=2$','$k_i=3$','$k_i=4$','Interpreter','latex','Fontsize', 16);


