function [ output_args ] = plot_side_uv( point1,point2 )
%PLOT_HAND_JOINTS 此处显示有关此函数的摘要
%   此处显示详细说明
%side=[1 2;2,3]
xc=[point1(1),point2(1)];
yc=[point1(2),point2(2)];
%zc=[point1(3),point2(3)];
plot(xc,yc,'-r','LineWidth',1.5);
end

