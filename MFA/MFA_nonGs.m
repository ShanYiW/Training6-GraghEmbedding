clear;
x0 = [2.5; -1.25];
theta_len = 10; theta_s = 2*pi/theta_len; theta = -2*pi/(2*theta_len);
r1 = 0.5; r2 = 1;
X_half = zeros([2, 30]);
for i=1:theta_len
    theta = theta + theta_s;
%     fprintf('%.2fΠ\n', theta/pi);
    X_half(:,i) = x0 + r1.*[cos(theta); sin(theta)];
    X_half(:,i+10) = x0 + r2.*[cos(theta); sin(theta)];
end
x1 = [2.1; -0.25];
for i=1:10
    X_half(:,i+20) = x1 + (i-1).*[-0.4; 0];
end

theta = -pi;
Rotate = [cos(theta), -sin(theta); sin(theta), cos(theta)];
X_half2 = Rotate*X_half;

X = [X_half, X_half2];
gnd = [zeros([30,1]); ones([30,1])];
ki=14; ko=1;
[E] = MFA(X, gnd, ki,ko);
E2 = sqrt(sum(E.*E));
E = E*diag(1./E2);
figure;
plot(X_half(1,:), X_half(2,:), 'o'); hold on;
plot(X_half2(1,:), X_half2(2,:), 's'); hold on;
plot([0, E(1,1)], [0, E(2,1)], '-','Linewidth',2); hold on;
% plot([0, E(1,2)], [0, E(2,2)], '-','Linewidth',2);
axis('equal'); hold off;
