function [img, img_pos, img_vec, pre_mask, varargout] = VisualizeVerts(verts, surface_depth, gridlen, edge, varargin)
% VisualizeObjectPoints  - 点云可视化
% ---------------------
%   以 XOY 平面的 -Z 视角投影可视化点云(图片坐标分别对应X，-Y)
%   对于图像中的空洞点以邻域均值填充
%
%
% Syntax
% ------
%   [img, img_pos, img_vec, pre_mask] = VisualizeObjectPoints(verts, surface_depth, gridlen, edge)
%   可视化点云
%
%   [img, img_pos, img_vec, pre_mask, img_pos_gt] = VisualizeObjectPoints(verts, surface_depth, gridlen, edge，verts_gt)
%   可视化点云,以 verts_gt 作为可视图对应的三维真值
%
%   [img, img_pos, img_vec, pre_mask] = VisualizeObjectPoints(verts, surface_depth, gridlen, edge, col, row, size_c, size_r);
%   指定点云中每个点在图片的位置，使得配准的点云在可视化图片中也能一一对应
%
%
% Examples
% --------
%   对已配对点云生成图片，并令图片对的像素一一对应
%     grid_verts1 = round(round(match_verts1 / gridlen) + 0.1); % 防止-0的出现
%     grid_verts2 = round(round(match_verts2 / gridlen) + 0.1);
%     grid_verts = [grid_verts1;grid_verts2];
%     min_col = min(grid_verts(:, 1));
%     min_row = min(grid_verts(:, 2));
%     max_col = max(grid_verts(:, 1)) - min_col + 2 * edge;
%     max_row = max(grid_verts(:, 2)) - min_row + 2 * edge;
%     col1 = grid_verts1(:, 1) - min_col + edge;
%     row1 = grid_verts1(:, 2) - min_row + edge;
%     col2 = grid_verts2(:, 1) - min_col + edge;
%     row2 = grid_verts2(:, 2) - min_row + edge;
%     [img1, img_pos1, img_vec1, pre_mask1] = VisualizeObjectPoints(match_verts1, surface_depth1, gridlen, edge, col1, row1, max_col, max_row);
%     [img2, img_pos2, img_vec2, pre_mask2] = VisualizeObjectPoints(match_verts2, surface_depth2, gridlen, edge, col2, row2, max_col, max_row);
%
%
% Input Arguments
% ---------------
%   verts           - 点云 (Nx3)
%   surface_depth   - 每个点对应的表面深度，Nx1
%   gridlen         - 离散化每个像素对应的尺度 (可以使用 0.05)
%   edge            - 图像边界的空白 (可以使用 30)
%   varargin        - 如果有两个坐标对齐的点云，需要在图片中也对齐显示时，添加的额外参数
%                   col             - 离散化的宽度
%                   row             - 离散化的高度
%                   size_c          - 图像宽度
%                   size_r          - 图像高度
%
%
% Output Arguments
% ----------------
%   img             - 可视化图像，RxCx3
%   img_pos         - 可视化图像中每一像素点对应的位置，RxCx3
%   img_vec         - 可视化图像中每一点对应的法向量，RxCx3
%   pre_mask        - 可视化图像的前景遮罩
%   img_pos_gt      - 希望对应的某种坐标位置，RxCx3
%
%
% GuanXiongJun , 2021-02-24

warning('off');


% 降采样
pc=pcdownsample(pointCloud(verts),'gridAverage',0.5); %0.5

% 使用 delaunayTriangulation 函数对球体进行三角剖分
DT = delaunayTriangulation(pc.Location);

% 查找三角剖分的自由边界面，并使用它们在曲面上创建二维三角剖分
[T,Xb] = freeBoundary(DT);
TR = triangulation(T,Xb);

% 计算 TR 中每个三角面的中心和顶点/面法向量
P = incenter(TR);
V = vertexNormal(TR);
F = faceNormal(TR);

% % 绘制可见部分的三角剖分以及中心和面法线
% figure;
% trisurf(T,Xb(:,1),Xb(:,2),Xb(:,3),'FaceColor','cyan','FaceAlpha',0.8);
% axis equal; hold on;
% quiver3(P(:,1),P(:,2),P(:,3),F(:,1),F(:,2),F(:,3),0.5,'color','r');
% quiver3(Xb(:,1),Xb(:,2),Xb(:,3),V(:,1),V(:,2),V(:,3),0.5,'Color','b');

% 只有与 -Z 小于 80 度的点可见
T = T(F(:,3)<-0.17,:);
V = V(unique(T(:)),:);
Xb = Xb(unique(T(:)),:);


% 插值估计可见部分深度，与估计深度近似的点作为可见点
F_Xb = scatteredInterpolant(Xb(:,1),Xb(:,2),Xb(:,3),'linear','none');
interpz = F_Xb(verts(:,1),verts(:,2));
F_V1 = scatteredInterpolant(Xb(:,1),Xb(:,2),V(:,1),'linear','none');
F_V2 = scatteredInterpolant(Xb(:,1),Xb(:,2),V(:,2),'linear','none');
F_V3 = scatteredInterpolant(Xb(:,1),Xb(:,2),V(:,3),'linear','none');
interpv = [F_V1(verts(:,1),verts(:,2)),F_V2(verts(:,1),verts(:,2)),F_V3(verts(:,1),verts(:,2))];

visibility = (abs(interpz - verts(:,3)) < 2.5);


% 只使用可见部分
verts = verts(visibility,:);
% normal_vec = normal_vec(visibility, :);
surface_depth = surface_depth(visibility, :);
interpv = interpv(visibility, :);

% 图像参数初始化
if nargin == 4
    grid_verts = round(round(verts / gridlen) + 0.1);
    col = grid_verts(:, 1) - min(grid_verts(:, 1)) + edge;
    row = grid_verts(:, 2) - min(grid_verts(:, 2)) + edge;
    size_c = max(col) + edge;
    size_r = max(row) + edge;
elseif nargin == 5
    verts_gt = varargin{1}(visibility,:);
    grid_verts = round(round(verts / gridlen) + 0.1);
    col = grid_verts(:, 1) - min(grid_verts(:, 1)) + edge;
    row = grid_verts(:, 2) - min(grid_verts(:, 2)) + edge;
    size_c = max(col) + edge;
    size_r = max(row) + edge;
else
    col = varargin{1}(visibility);
    row = varargin{2}(visibility);
    size_c = varargin{3};
    size_r = varargin{4};
end

% 插值得到平面可视化值
[mx,my]=meshgrid(1:size_c,1:size_r);
F = scatteredInterpolant(col,row,surface_depth,'linear','none');
img_arr = F(mx,my);

F_P1 = scatteredInterpolant(col,row,verts(:,1),'linear','none');
F_P2 = scatteredInterpolant(col,row,verts(:,2),'linear','none');
F_P3 = scatteredInterpolant(col,row,verts(:,3),'linear','none');
img_pos = cat(3,F_P1(mx,my),F_P2(mx,my),F_P3(mx,my));

if nargin == 5
    F_PGT1 = scatteredInterpolant(col,row,verts_gt(:,1),'linear','none');
    F_PGT2 = scatteredInterpolant(col,row,verts_gt(:,2),'linear','none');
    F_PGT3 = scatteredInterpolant(col,row,verts_gt(:,3),'linear','none');
    varargout{1} = cat(3,F_PGT1(mx,my),F_PGT2(mx,my),F_PGT3(mx,my));
end

F_V1 = scatteredInterpolant(col,row,interpv(:,1),'linear','none');
F_V2 = scatteredInterpolant(col,row,interpv(:,2),'linear','none');
F_V3 = scatteredInterpolant(col,row,interpv(:,3),'linear','none');
img_vec = cat(3,F_V1(mx,my),F_V2(mx,my),F_V3(mx,my));


% 图像前景区域
pre_mask = ~isnan(img_arr);

% 可视化图像归一化，按捺指纹脊线是接触部位因此为深色，非接触指纹由于光照导致脊线更亮
img_max = max(img_arr(pre_mask));
img_min = min(img_arr(pre_mask));
img_arr = round((1 - ((img_arr - img_min) / (img_max - img_min))) * 255);
img_data = img_arr(pre_mask);
img_arr(~pre_mask) = 255;
img_arr(img_arr == 0) = 1;


% 直方图均衡
hist_mask = hist(img_data,1:256);
cdf = cumsum(hist_mask);
cdf = (cdf - cdf(1)) * 255 / (cdf(end) - 1);
cdf = round(cdf / max(cdf) * 255);
img = cdf(img_arr)/255;

img = uint8(img*255);

end
