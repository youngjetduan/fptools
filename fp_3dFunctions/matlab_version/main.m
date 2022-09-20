data_dir = 'D:\GuanXJ\3D_FP\3D_PtPoseDataGenerate\generate_pose_data\';
save_dir = 'D:\GuanXJ\3D_FP\3D_PtPoseDataGenerate\generate_pose_data\generate\';

delta_z = 3;
dpi = 500;
brightness = 0.6;
edge = 30;

data_path = fullfile( data_dir, '*.mat' );
data_list = dir(data_path);



angs = [];

i = 0;
while i < length(data_list)
    i = i + 1;
    fprintf([num2str(i),' / ', num2str(length(data_list)),'\n']);
    data = load([data_dir, data_list(i).name]);
    points = data.points;
    depth = data.depth;
    
    img_name = data_list(i).name;
    save_name = img_name(1:end-4);
    
    save_path = [save_dir, save_name, '/'];
    if(~exist(save_path,'dir')), mkdir(save_path); end
    
    num = 30;
    blank = zeros(1,num);
    rolls = [(30 * rand(1,num) + 15) .* sign(rand(1,num)-0.5),blank];
    pitchs = [blank,(20 * rand(1,num) + 10) .* sign(rand(1,num)-0.5)];
    yaw = 0;
%     yaws = [blank,blank,blank];

    for j = 1:size(rolls,2)
        fprintf([num2str(i),':  ',num2str(j),' / ', num2str(size(rolls,2)),'\n']);
        roll = rolls(j);
        pitch = pitchs(j);
%         yaw = yaws(j);
        angs(j,1:3) = [roll, pitch, yaw];
        M = GetEulerMatrix(roll, pitch, yaw);
        flat_img = GeneratePoseData(M, delta_z, points, depth, dpi, edge, brightness);
        imwrite( flat_img, [save_path, save_name, '_', num2str(j), '.png']);
    end
    
    
   
    dlmwrite([save_path, save_name, '_', num2str(j), '.txt'], angs,'delimiter','\t','newline','pc');
    
end

