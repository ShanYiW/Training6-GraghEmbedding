%生成PIE, gnd 
clear;
D = 1024; 
N = 3329;
gnd = zeros([N,1]);
lb_p = 0; step = 49;
for p = 1:68
    gnd(lb_p+1:lb_p+step) = p;
    lb_p = lb_p + step;
    if p == 38 % 38号有164张，其余170张
        lb_p = lb_p - 3; 
    end
end

PIE = zeros([D,N]);
for j = 1:N
    img = imread(['PIE_C27_32_area/', num2str(j),'.jpg']);
    PIE(:,j) = img(:);
end
save("PIE_C27_32x32.mat", "PIE", "gnd");


