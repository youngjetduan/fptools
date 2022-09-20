function [flat_img, flat_posGT,varargout] = ImgPos2Flat(img,img_posGT,img_pos,pre_mask,gridlen, varargin)
if nargin == 6
    edge = varargin{1};
elseif nargin == 8
    if strcmp(varargin{1},'SetFigSize')
        size_c = varargin{2};
        size_r = varargin{3};
    end
end
[fx,fy] = gradient(img_pos(:,:,3) / gridlen);
mask = pre_mask;
mask(isnan(fx)) = 0;
mask(isnan(fy)) = 0;
mask=imdilate(imerode(mask,strel('disk',30)),strel('disk',30));
mask=imerode(mask,strel('disk',10));
fx(mask==0) = 0;
fy(mask==0) = 0;

[U,V,min_h,min_w]=curve_flat(fx,fy,mask);

% uu=double(U+min_w);
% vv=double(V+min_h);
% [h,w]=size(img_arr);
% [mx,my]=meshgrid(1:w,1:h);

uu=double(U+min_w);
vv=double(V+min_h);

if nargin == 6
    uu = uu - min(uu(:)) + edge;
    vv = vv - min(vv(:)) + edge;
    w = max(uu(:)) + edge;
    h = max(vv(:)) + edge;
else
    w = size_c;
    h = size_r;
end

[mx,my]=meshgrid(1:w,1:h);

img_arr = img(:,:,1);

F = scatteredInterpolant(uu(mask>0),vv(mask>0),double(img_arr(mask>0)),'linear','none');
flat_img = F(mx,my);
flat_img(isnan(flat_img)) = 255;
flat_img = uint8(flat_img);

img_posGT1 = img_posGT(:,:,1);
img_posGT2 = img_posGT(:,:,2);
img_posGT3 = img_posGT(:,:,3);
FGT1 = scatteredInterpolant(uu(mask>0),vv(mask>0),double(img_posGT1(mask>0)),'linear','none');
FGT2 = scatteredInterpolant(uu(mask>0),vv(mask>0),double(img_posGT2(mask>0)),'linear','none');
FGT3 = scatteredInterpolant(uu(mask>0),vv(mask>0),double(img_posGT3(mask>0)),'linear','none');
flat_posGT = cat(3,FGT1(mx,my),FGT2(mx,my),FGT3(mx,my));

varargout{1} = uu;
varargout{2} = vv;
end