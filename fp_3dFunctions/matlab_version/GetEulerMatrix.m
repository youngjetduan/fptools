function M = GetEulerMatrix(roll, pitch, yaw) 
% GetEulerMatrix - 从旋转角获得旋转矩阵
% --------------
%   得到欧拉矩阵，转动顺序为yaw -> pitch -> roll
%   设3x1向量 V，旋转关系为 V2 = M x V
% 
% 
% Syntax
% ------
%   M = GetEulerMatrix(roll, pitch, yaw)
% 
% 
% Input Arguments
% ---------------
%   roll           - 绕X轴旋转角(角度制)
%   pitch          - 绕Y轴旋转角(角度制)
%   yaw            - 绕Z轴旋转角(角度制)
%
% 
% Output Arguments
% ----------------
%   M              - 旋转矩阵
%   
%
% GuanXiongJun , 2021-01-22

roll = roll * pi / 180;
pitch = pitch * pi / 180;
yaw = yaw * pi / 180;

Mx = [1 0 0;0 cos(pitch) -sin(pitch);0 sin(pitch) cos(pitch)];
My = [cos(roll) 0 sin(roll); 0 1 0; -sin(roll) 0 cos(roll)];
Mz = [cos(yaw) -sin(yaw) 0;sin(yaw) cos(yaw) 0;0 0 1];

M = My * Mx * Mz;

end