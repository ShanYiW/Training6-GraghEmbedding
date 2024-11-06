clear;
ORL = uint8(zeros([2576, 400])); % 28*23, 400张
gnd = uint8(zeros([400,1])); % 类别标签
mode = 'area';
for i=1:400
    idx = num2str(i);
    face = imread(['ORL_12_', mode,'/orl', idx, '.pgm']);
    face_heq = histeq(face);
    ORL(:,i) = face_heq(:);
end
for i=1:40; gnd((i-1)*10+1:i*10) = i;end
save(append("ORL_12_", [mode,'_heq']), "ORL", "gnd"); % 生成ORL: 644*400(uint8)
%%
% ORL = uint8(zeros([10304, 400])); % 28*23, 400张
% gnd = uint8(zeros([400,1])); % 类别标签
% for i=1:400
%     face = imread(['ORL/orl', num2str(i), '.pgm']);
%     ORL(:,i) = face(:);
% end
% for i=1:40; gnd((i-1)*10+1:i*10) = i;end
% save("ORL", "ORL", "gnd"); % 生成ORL: 644*400(uint8)