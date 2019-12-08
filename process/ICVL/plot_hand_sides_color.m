function [ output_args ] = plot_hand_sides_color( points )
%PLOT_HAND_SIDES 此处显示有关此函数的摘要
%   此处显示详细说明
sides=[1 2;2 3;3 4;1 5;5 6;6 7;1 8;8 9;9 10;1 11;11 12;12 13;1 14;14 15;15 16;];
for i = 1:size(sides,1)
%     disp(sides(i,:));
    %plot_side(points(sides(i,1),:),points(sides(i,2),:));
    point1=points(sides(i,1),:);
    point2=points(sides(i,2),:);
    xc=[point1(1),point2(1)];
    yc=[point1(2),point2(2)];
    zc=[point1(3),point2(3)];
    if i<4
        plot3(xc,zc,yc,'-r','LineWidth',3);
        scatter3(points(2:4,1), points(2:4,3), points(2:4,2), 50, 'r', 'Fill','LineWidth', 2);
    elseif i<7
        plot3(xc,zc,yc,'-y','LineWidth',3); 
        scatter3(points(5:7,1), points(5:7,3), points(5:7,2), 50, 'y', 'Fill','LineWidth', 2);
    elseif i<10
        plot3(xc,zc,yc,'-g','LineWidth',3);
        scatter3(points(8:10,1), points(8:10,3), points(8:10,2), 50, 'g', 'Fill','LineWidth', 2);
    elseif i<13
        plot3(xc,zc,yc,'-c','LineWidth',3);
        scatter3(points(11:13,1), points(11:13,3), points(11:13,2), 50, 'c', 'Fill','LineWidth', 2);
    else
        plot3(xc,zc,yc,'-b','LineWidth',3);
        scatter3(points(14:16,1), points(14:16,3), points(14:16,2), 50, 'b', 'Fill','LineWidth', 2);
    end
    hold on;
end

end

