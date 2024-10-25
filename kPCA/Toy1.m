clear;
N = 100; % 样本数
d = 1; % 多项式核的阶数
%% 构造数据集 X: D*N
rand('seed',0); randn('seed',0);
x = 2.*rand([1,N]) - 1; % [-1,1]
y = x.*x + 0.2.*randn([1,N]);
X = [x;y]; % 2*N
%% kernel PCA
[E] = kPCA_poly(X, d); % E:N*N
%% 画等高图
x_grid = -1:0.05:1;
y_grid = -0.5:0.05:1.5;
[Xgrid,Ygrid] = meshgrid(x_grid, y_grid);
lenx = length(x_grid);
leny = length(y_grid);
PC = zeros([lenx, leny]); % principal component
for p=1:d+1
for i=1:lenx
    for j=1:leny
        new_x = [Xgrid(i,j); Ygrid(i,j)];
        kernel = (X'*new_x).^d; % d阶多项式核
        PC(j,i) = E(:,p)'*kernel;
    end
end
figure;
plot(x,y,'o', 'MarkerFaceColor',"#0072bd", 'MarkerSize',3);hold on;
contour(Xgrid,Ygrid,PC, 'ShowText','on', 'LabelSpacing',288); hold off;
xlim([-1,1]);ylim([-0.5,1.5]); xlabel('x'); ylabel('y'); axis('equal');
end