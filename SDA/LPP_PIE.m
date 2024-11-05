clear;
load("PIE_C27_32x32.mat"); % PIE, gnd
N = length(gnd);
%% 每类随机抽30张图象 (第1张给标签)
C = 68; % 类别数
Nc = 30; % 每类样本数
Ntrain = C*Nc;
Ntest = N - Ntrain;
step = 49;

k_arr = 1:1; t_arr = [16e4:2e4:22e4];
% k_arr = 0.01; t_arr = 1;
k_len = length(k_arr); t_len = length(t_arr);
d_lb = 2; d_ub = 100; d_len = d_ub - d_lb + 1;

Nround = 6;
Acc_tr = zeros([Nround, k_len+1, t_len, d_len]);
Acc_te = zeros([Nround, k_len+1, t_len, d_len]);
rand('seed', 7);
for round = 1:Nround
    lb = 0;
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
    for ki = 1:k_len
    for ti = 1:t_len
        [A] = LPP_my(Xtrain, k_arr(ki), t_arr(ti));
        d_ub = min(d_ub, size(A,2));
        for d = d_lb:d_ub
            Ytrain = A(:,1:d)'*Xtrain;
            Ytest = A(:,1:d)'*PIE(:,is_test);
            %% 未标注的训练样本
            Ytr2 = sum(Ytrain.*Ytrain, 1); % 1*Ntrain
            dis_tr = repmat(Ytr2,[C,1]) + repmat(Ytr2(1:C)',[1,Ntrain]) - 2.*Ytrain(:,1:C)'*Ytrain;
            [~, idx] = sort(dis_tr);
            pred = gndtrain(idx(1,C+1:Ntrain)');
            Acc_tr(round, ki, ti, d - d_lb + 1) = sum(pred==gndtrain(C+1:Ntrain))/(Ntrain - C);
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
            Acc_te(round, ki, ti, d - d_lb + 1) = acc/Ntest;
        end
    end
    end
end
mAccTr = mean(Acc_tr,1); mAccTe = mean(Acc_te);
mAccTr = squeeze(mAccTr); mAccTe = squeeze(mAccTe); % k_len * t_len * d_len
[mx_tr,mx_itr] = max(mAccTr(:)); 
d_mxtr = floor(mx_itr/((k_len+1)*t_len)) + 1;
[mx_te,mx_ite] = max(mAccTe(:)); 
d_mxte = floor(mx_ite/((k_len+1)*t_len)) + 1;
fprintf('%f %f\n%f %f\n', mx_tr, std(Acc_tr(:,1,1,d_mxtr)), mx_te, std(Acc_te(:,1,3,d_mxte)) )
% figure;
% bar_3 = bar3(mean(mAccTe(:,:,1:d_ub), 3));
% for ki=1:length(bar_3)
%     bar_3(ki).CData = bar_3(ki).ZData; % 颜色与Z取值成正比
%     bar_3(ki).FaceColor = 'interp';
% end
% zlim([0.43,0.432]);
% % set(gca, 'xTick', [1 4:5:19]); % d=2:20
% set(gca, 'xTicklabel', split(num2str( t_arr )) ); 
% % set(gca, 'yTick', [1:2:13]); % 
% set(gca, 'yTicklabel', split(num2str(k_arr)) );
% set(gca,'XGrid', 'off', 'YGrid', 'off','ZGrid', 'on')
% xlabel('$\beta$', 'Interpreter', 'latex', 'Fontsize', 14);
% ylabel('$\alpha$', 'Interpreter', 'latex', 'Fontsize', 14);
% zlabel('$Accuracy$', 'Interpreter', 'latex', 'Fontsize', 14);
% view([220, 10]); % 0度:延第二维度(↓)的反方向(↑) 正视第一维度(→)
