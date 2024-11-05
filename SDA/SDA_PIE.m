clear;
load("PIE_C27_32x32.mat"); % PIE, gnd
N = length(gnd);
%% 每类随机抽30张图象 (第1张给标签)
C = 68; % 类别数
Nc = 30; % 每类样本数
Ntrain = C*Nc;
Ntest = N - Ntrain;

k = 1;
% alp_arr = 0.01:0.01:0.1; beta_arr = 0.2:0.2:1;
alp_arr = 0.01; beta_arr = 1;
len_alp = length(alp_arr); len_beta = length(beta_arr);

step = 49;
d_lb = 2; d_ub = 100; d_len = d_ub - d_lb + 1;

Nround = 20;
Acc_tr = zeros([Nround, len_alp, len_beta, d_len]);
Acc_te = zeros([Nround, len_alp, len_beta, d_len]);
rand('seed', 7);
for round = 1:Nround
    lb = 0; % 因为
    chosen_train = zeros([C,Nc]); % 因为后面要拉成向量，所以每次都要初始化
    for p = 1:C
        chosen = randperm(step, Nc); % 从1~step中抽Nc个整数
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
    Xtrain = PIE(:, chosen_train); % 带标签的排前1~C
    gndtrain = gnd(chosen_train);
    is_test = true([N, 1]);
    is_test(chosen_train) = false;
    gndtest = gnd(is_test);
    %% 运行SDA, 得到 Ytrain, Ytest
    for alp_i = 1:len_alp
    for beta_i = 1:len_beta
        [A] = SDA(Xtrain, gndtrain(1:C), k, alp_arr(alp_i), beta_arr(beta_i));
        for d = d_lb:d_ub
            Ytrain = A(:,1:d)'*Xtrain;
            Ytest = A(:,1:d)'*PIE(:,is_test);
            %% 未标注的训练样本
            Ytr2 = sum(Ytrain.*Ytrain, 1); % 1*Ntrain
            dis_tr = repmat(Ytr2, [C, 1]) + repmat(Ytr2(1:C)', [1, Ntrain]) - 2.*Ytrain(:,1:C)'*Ytrain;
            [~, idx] = sort(dis_tr);
            pred = gndtrain(idx(1,C+1:Ntrain)');
            Acc_tr(round, alp_i, beta_i, d - d_lb + 1) = sum(pred==gndtrain(C+1:Ntrain))/(Ntrain - C);
            %% 测试样本
            acc = 0;
            for j=1:Ntest
                y = Ytest(:,j);
                dis_te = sum((repmat(y,[1,C]) - Ytrain(:,1:C)).^2); % 1*C
                nearest_i = 1; nearest_dis = dis_te(nearest_i);
                for i=2:C
                    if dis_te(i)< nearest_dis
                        nearest_dis = dis_te(i);
                        nearest_i = i;
                    end
                end
                pred = gndtrain(nearest_i);
                acc = acc + (pred==gndtest(j));
            end
            Acc_te(round, alp_i, beta_i, d - d_lb + 1) = acc/Ntest;
        end
    end
    end
end
Acc_tr = squeeze(Acc_tr); Acc_te = squeeze(Acc_te);
% figure('Name',['k',num2str(k),'_alp',num2str(alpha),'_beta',num2str(beta)]); 
% plot(lb_d:ub_d, mean(Acc_tr,1),'o-'); hold on;
% plot(lb_d:ub_d, mean(Acc_te,1),'s-'); hold off;
mAccTr = mean(Acc_tr); mx_itr=1;mx_tr=mAccTr(1);
for i=2:d_len; if mAccTr(i)>mx_tr; mx_tr=mAccTr(i); mx_itr=i;end;end
mAccTe = mean(Acc_te); mx_ite=1;mx_te=mAccTe(1);
for i=2:d_len; if mAccTe(i)>mx_te; mx_te=mAccTe(i); mx_ite=i;end;end
fprintf('%f %f\n%f %f\n', mx_tr, std(Acc_tr(:,mx_itr)), mx_te, std(Acc_te(:,mx_ite)) );

figure;
bar_3 = bar3(Acc_te(:,:,end));
for k=1:length(bar_3)
    bar_3(k).CData = bar_3(k).ZData; % 颜色与Z取值成正比
    bar_3(k).FaceColor = 'interp';
end
zlim([0.395,0.425]);
% set(gca, 'xTick', [1 4:5:19]); % d=2:20
set(gca, 'xTicklabel', split(num2str( beta_arr )) ); % 不能split(, ' '), 因为'2   3'-> {'2'},{' '},{' '},{'3'}
% set(gca, 'yTick', [1:2:13]); % 
set(gca, 'yTicklabel', split(num2str(alp_arr)) );
set(gca,'XGrid', 'off', 'YGrid', 'off','ZGrid', 'on')
xlabel('$\beta$', 'Interpreter', 'latex', 'Fontsize', 14);
ylabel('$\alpha$', 'Interpreter', 'latex', 'Fontsize', 14);
zlabel('$Accuracy$', 'Interpreter', 'latex', 'Fontsize', 14);
view([220, 10]); % 0度:延第二维度(↓)的反方向(↑) 正视第一维度(→)
