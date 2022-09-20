data_dir = '/mnt/data1/hefei_data/classified/NORMAL/';
stl_path = fullfile( data_dir, '*.stl' );
stl_list = dir(stl_path);
save_dir = '/mnt/data1/hefei_data/3d_processed/pt_stl2mat/NORMAL/';
if(~exist(save_dir,'dir')), mkdir(save_dir); end
i = 0;
while i < length(stl_list)
    i = i + 1;
    fprintf([num2str(i),' / ', num2str(length(stl_list)),'\n']);
    [f,~,~] = stlread([data_dir,stl_list(i).name]);
    points = f.Points;
    save_path = [save_dir, stl_list(i).name(1:end-4),'.mat'];
    save(save_path,'points');
    
end

