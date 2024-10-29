clear;
%% ORL
% load('ORL_Eig.mat');
% Err_Eigen = 1 - Acc_avg;
% load('ORL_Fish.mat');
% Err_Fisher = 1 - Acc_avg;
% load('ORL_kEig_p2.mat');
% Err_keig_p2 = 1 - Acc_avg;
% load('ORL_kEig_p3.mat');
% Err_keig_p3 = 1 - Acc_avg;
% load('ORL_kFLD_p2.mat');
% Err_kFLD_p2 = 1 - Acc_avg;
% load('ORL_kFLD_gs.mat');
% Err_kFLD_gs = 1 - Acc_avg;
% figure;
% plot(10:ub_d,Err_Eigen(9:end),'o-','Color',[237,177,32]./255,'MarkerFaceColor','w','Linewidth',1.5); hold on;
% plot(10:ub_d,Err_Fisher(9:end),'+-','Color',[217,83,25]./255,'MarkerFaceColor','w','Linewidth',1.5); hold on;
% plot(10:ub_d,Err_keig_p2(9:end),'^-','Color',[255,153,200]./255,'MarkerFaceColor','w','Linewidth',1.5); hold on;
% plot(10:ub_d,Err_keig_p3(9:end),'s-','Color',[77,190,238]./255,'MarkerFaceColor','w','Linewidth',1.5); hold on;
% plot(10:ub_d,Err_kFLD_p2(9:end),'p-','Color',[162,20,47]./255,'MarkerFaceColor','w','Linewidth',1.5); hold on;
% plot(10:ub_d,Err_kFLD_gs(9:end),'*-','Color',[125,46,143]./255,'MarkerFaceColor','w','Linewidth',1.5); hold on;
% legend('Eigenfaces', 'Fisherfaces','Kernel eigenfaces (p=2)', ...
%     'Kernel eigenfaces (p=3)','kFLD (p=2)','kFLD (Gaussian)','Location','Northeast');
% xlabel('Dims', 'Fontsize', 16);
% ylabel('Error rate (%)', 'Fontsize', 16);
%% Yale
load('Yale_Eig.mat');
Err_Eigen = 1 - Acc_avg;
load('Yale_Fish.mat');
Err_Fisher = 1 - Acc_avg;
load('Yale_kEig_p2.mat');
Err_keig_p2 = 1 - Acc_avg;
load('Yale_kEig_p3.mat');
Err_keig_p3 = 1 - Acc_avg;
load('Yale_kFLD_p2.mat');
Err_kFLD_p2 = 1 - Acc_avg;
load('Yale_kFLD_gs.mat');
Err_kFLD_gs = 1 - Acc_avg;
figure;
plot(lb_d:ub_d,Err_Eigen,'o-','Color',[237,177,32]./255,'MarkerFaceColor','w','Linewidth',1.5); hold on;
plot(lb_d:ub_d,Err_Fisher,'+-','Color',[217,83,25]./255,'MarkerFaceColor','w','Linewidth',1.5); hold on;
plot(lb_d:ub_d,Err_keig_p2,'^-','Color',[255,153,200]./255,'MarkerFaceColor','w','Linewidth',1.5); hold on;
plot(lb_d:ub_d,Err_keig_p3,'s-','Color',[77,190,238]./255,'MarkerFaceColor','w','Linewidth',1.5); hold on;
plot(lb_d:ub_d,Err_kFLD_p2,'p-','Color',[162,20,47]./255,'MarkerFaceColor','w','Linewidth',1.5); hold on;
plot(lb_d:ub_d,Err_kFLD_gs,'*-','Color',[125,46,143]./255,'MarkerFaceColor','w','Linewidth',1.5); hold on;
legend('Eigenfaces', 'Fisherfaces','Kernel eigenfaces (p=2)', ...
    'Kernel eigenfaces (p=3)','kFLD (p=2)','kFLD (Gaussian)','Location','Northeast');
xlabel('Dims', 'Fontsize', 16);
ylabel('Error rate (%)', 'Fontsize', 16);