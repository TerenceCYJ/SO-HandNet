% Jonathan Tompson, New York University - 8/28/2014
% This is a simple script to visualize a training example
clearvars; close all; clc; rng(0);

dataset_dir = 'F:\DeepLearningDataset\ICVL\old\Testing\Depth\';
txt_dir = 'F:\DeepLearningDataset\ICVL\old\Testing\Depth\1.txt';
index=1;
subject_names={'test_seq_1','test_seq_2'};

image_index = 50;

%% Load joints coordinates
All_joints = importdata(txt_dir);

jnt_pixel=squeeze(All_joints(image_index+1,3:50));
jnt_pixel=(reshape(jnt_pixel,3,size(jnt_pixel,2)/3))';
%% Load and display a depth example
image_name=All_joints(image_index,2);
filename_prefix = sprintf('%04d',image_name);
depth = imread([dataset_dir,subject_names{index},'\', 'image_', filename_prefix, '.png']);

figure;
imshow(depth, [0, max(depth(:))]);


jnt_uvd=jnt_pixel;
jnt_colors = rand(size(jnt_uvd,1), 3);
hold on;
scatter(jnt_uvd(:,1), jnt_uvd(:,2), 20, jnt_colors, 'filled');

%% Visualize the depth in 3D
uvd = convert_depth_to_uvd(depth);
xyz = convert_uvd_to_xyz(uvd);
xyz1=xyz(:,:,1);
xyz2=xyz(:,:,2);
xyz3=xyz(:,:,3);
% Decimate the image (otherwise rendering will be too slow)
decimation = 2;
xyz_decimated = xyz(1:decimation:end, 1:decimation:end, :);
points = reshape(xyz_decimated, size(xyz_decimated,1)*size(xyz_decimated,2), 3);
%points = reshape(xyz, size(xyz,1)*size(xyz,2), 3);
% Collect the points within the AABBOX of the non-background points
body_points = points(find(points(:,3) < 2000),:);
axis_bounds = [min(body_points(:,1)) max(body_points(:,1)) ...
  min(body_points(:,3)) max(body_points(:,3)) ...
  min(body_points(:,2)) max(body_points(:,2))];
% Visualize the entire point cloud
figure;
plot3(body_points(:,1), body_points(:,3), body_points(:,2), '.', 'MarkerSize', 1.5);
axis(axis_bounds);
view(45,20);
set(gcf,'renderer','opengl'); axis vis3d; axis equal; hold on; grid on;

%% Visualize the hand and the joints in 3D
for i = 1:2
  figure;
  if i == 1
    uvd = convert_depth_to_uvd(depth);
  else
    uvd = convert_depth_to_uvd(synthdepth);
  end
  xyz = convert_uvd_to_xyz(uvd);
  points = reshape(xyz, size(xyz,1)*size(xyz,2), 3);
  %colors = reshape(rgb, size(rgb,1)*size(rgb,2), 3);
  hand_points = squeeze(convert_uvd_to_xyz(reshape(jnt_uvd, 1, size(jnt_uvd,1), 3)));
  % Collect the points within the AABBOX of the hand
  axis_bounds = [min(hand_points(:,1)) max(hand_points(:,1)) ...
    min(hand_points(:,3)) max(hand_points(:,3)) ...
    min(hand_points(:,2)) max(hand_points(:,2))];
  axis_bounds([1 3 5]) = axis_bounds([1 3 5]) - 20;
  axis_bounds([2 4 6]) = axis_bounds([2 4 6]) + 20;
  ipnts = find(points(:,1) >= axis_bounds(1) & points(:,1) <= axis_bounds(2) & ...
    points(:,2) >= axis_bounds(5) & points(:,2) <= axis_bounds(6) & ...
    points(:,3) >= axis_bounds(3) & points(:,3) <= axis_bounds(4));
  points = points(ipnts, :);
  %colors = double(colors(ipnts, :))/255;
  plot3(points(:,1), points(:,3), points(:,2), '.', 'MarkerSize', 1.5); 
  set(gcf,'renderer','opengl'); axis vis3d; axis equal; hold on; grid on;
  scatter3(hand_points(:,1), hand_points(:,3), hand_points(:,2), 50, jnt_colors, 'Fill','LineWidth', 0.5);
  axis(axis_bounds);
  %view(90,180)
end
