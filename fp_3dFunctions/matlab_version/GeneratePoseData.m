function flat_img = GeneratePoseData(M,delta_z, points,depth,dpi,edge,brightness)
gridlen = 25.4 / dpi;
points = points * M';

minz = min(points(:,3));
depth = depth(points(:,3) < minz + delta_z ,:);
points = points(points(:,3) < minz + delta_z,:);

[img, img_pos, ~, pre_mask] = VisualizeVerts(points, depth, gridlen, edge);
img2 = imdivide(img,brightness);
% figure,imshow(img2);

[flat_img,~] = ImgPos2Flat(img2,img_pos,img_pos,pre_mask,gridlen,edge);


end