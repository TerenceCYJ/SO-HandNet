clearvars; close all; clc; rng(0);
%%
sides=[1 2;2 3;1 3];
points=[0,0,0;1,1,1;1,0,1];
for i = 1:size(sides,1)
    disp(sides(i,:));
    plot_side(points(sides(i,1),:),points(sides(i,2),:));
    hold on;
end

