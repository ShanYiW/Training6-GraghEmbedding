clear;
N = 30;
sig = 0.1;
X = zeros([2, 3*N]);
randn('seed',0);
%%
x = sig.*randn([2,N]);
X(:,1:N) = x + [0.55; 0];
x = sig.*randn([2,N]);
X(:,N+1:2*N) = x + [-0.55; -0.2];
x = sig.*randn([2,N]);
X(:,2*N+1:3*N) = x + [0; 0.6];
X2 = sum(X.*X, 1); % 1*3N
%% kernel PCA
[E] = kPCA_laplace(X, sig); % E:N*N
%% 画等高图
x_grid = -1:0.05:1;
y_grid = -0.5:0.05:1;
[Xgrid,Ygrid] = meshgrid(x_grid, y_grid);
lenx = length(x_grid);
leny = length(y_grid);
PC = zeros([leny, lenx]); % principal component
for p=1:8
for i=1:lenx
    for j=1:leny
        new_x = [Xgrid(j,i); Ygrid(j,i)];
        dist = X2' + sum(new_x.*new_x) - 2.*X'*new_x; % N*1
        kernel = exp(-dist./sig); % laplace核
        PC(j,i) = E(:,p)'*kernel;
    end
end
figure;
plot(X(1,:),X(2,:),'o', 'MarkerFaceColor',"#0072bd", 'MarkerSize',3);hold on;
contour(Xgrid,Ygrid,PC, 'ShowText','on', 'LabelSpacing',288); hold off;
xlim([-1,1]);ylim([-0.5,1]); xlabel('x'); ylabel('y'); axis('equal');
end