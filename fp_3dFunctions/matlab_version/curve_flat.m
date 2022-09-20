function [U,V,min_h,min_w]=curve_flat(fx,fy,mask)
[h,w]=size(mask);
ff=sqrt(fx.^2+fy.^2);
mask2=imerode(mask,strel('disk',20));
ff2=ff.*double(mask2>0);ff2(mask2==0)=nan;
[min_ff,min_ind]=min(ff2(:));
[min_h,min_w]=ind2sub([h,w],min_ind);
ux=sqrt(1+fx.^2);
uy=sqrt(1+fy.^2);
%% u
%% y-x
vv=zeros(h,1);
mask_vv=mask(:,min_w);
vv(mask_vv==0)=nan;
h1=ux(:,1:min_w);
h1=-fliplr(h1);
hh1=cumtrapz(h1')';
hh1=fliplr(hh1);
h2=ux(:,min_w:end);
hh2=cumtrapz(h2')';
hh=[hh1,hh2(:,2:end)];
U1=bsxfun(@plus,hh,vv);

v1=uy(1:min_h,min_w);
v1=-flipud(v1);
vv1=cumtrapz(v1);
vv1=flipud(vv1);
v2=uy(min_h:end,min_w);
vv2=cumtrapz(v2);
vv=[vv1;vv2(2:end)];
vv(mask_vv==0)=nan;
hh=zeros(h,w);
V1=bsxfun(@plus,hh,vv);
%% x-y
h1=ux(min_h,1:min_w);
h1=-fliplr(h1);
hh1=cumtrapz(h1);
hh1=fliplr(hh1);
h2=ux(min_h,min_w:end);
hh2=cumtrapz(h2);
hh=[hh1,hh2(2:end)];
mask_hh=mask(min_h,:);
hh(mask_hh==0)=nan;
vv=zeros(h,w);
U2=bsxfun(@plus,hh,vv);

hh=zeros(1,w);
mask_hh=mask(min_h,:);
hh(mask_hh==0)=nan;
v1=uy(1:min_h,:);
v1=-flipud(v1);
vv1=cumtrapz(v1);
vv1=flipud(vv1);
v2=uy(min_h:end,:);
vv2=cumtrapz(v2);
vv=[vv1;vv2(2:end,:)];
V2=bsxfun(@plus,hh,vv);
%%
mask_12=(~isnan(U1))&(~isnan(U2))&(mask>0);
mask_1=(~isnan(U1))&(isnan(U2))&(mask>0);
mask_2=(isnan(U1))&(~isnan(U2))&(mask>0);
U1(isnan(U1))=0;U2(isnan(U2))=0;
V1(isnan(V1))=0;V2(isnan(V2))=0;
U=U1.*mask_12+U1.*mask_1+U2.*mask_2;
% U=U1;
U=U.*mask;
V=V1.*mask_12+V1.*mask_1+V2.*mask_2;
% V=V1;
V=V.*mask;
end