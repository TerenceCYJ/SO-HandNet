function [ output_args ] = plot_uv_sides( points )
%PLOT_HAND_SIDES 此处显示有关此函数的摘要
%   此处显示详细说明
sides=[1 2;2 3;3 4;1 5;5 6;6 7;1 8;8 9;9 10;1 11;11 12;12 13;1 14;14 15;15 16;];
for i = 1:size(sides,1)
%     disp(sides(i,:));
    plot_side_uv(points(sides(i,1),:),points(sides(i,2),:));
    hold on;
end
end

