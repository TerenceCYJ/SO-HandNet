% Jonathan Tompson, New York University - 8/28/2014
% This is a simple script to visualize a training example
clearvars; close all; clc; rng(0);

dataset_dir = 'F:\DeepLearningDataset\ICVL\old\Training\Depth\raw\';
txt_dir = 'F:\DeepLearningDataset\ICVL\old\Training\Depth\raw\201403121140.txt';
subject_names={'201403121135','201403121140','201405151126','201406030937','201406031456','201406031503','201406031747','201406041509','201406181554','201406181600','201406191014','201406191044'};
image_index =1;
index=2;
SAMPLE_NUM = 1024;JOINT_NUM = 16;

%% Load joints coordinates
All_joints = importdata(txt_dir);
jnt_pixel=squeeze(All_joints(image_index+1,3:50));
jnt_pixel=(reshape(jnt_pixel,3,size(jnt_pixel,2)/3))';
%% Load and display a depth example
image_name=All_joints(image_index,2);
if image_name>9999
	filename_prefix = sprintf('%05d',image_name);
else
	filename_prefix = sprintf('%04d',image_name);
end
depth = imread([dataset_dir,subject_names{index},'\', 'image_', filename_prefix, '.png']);

% % jnt_uvd1=jnt_pixel(:,1)*320/0.7351+160;
% jnt_uvd2=jnt_pixel(:,2)*241.42/1.004+120;
% jnt_uvd1=jnt_pixel(:,1)*320+160;
% %jnt_uvd2=jnt_pixel(:,2)*240+120;
% jnt_uvd3=jnt_pixel(:,3)*500;
% jnt_uvd=[jnt_uvd1 jnt_uvd2 jnt_uvd3];
% % jnt_uvd=convert_xyz_to_uvd(jnt_pixel);
%%
jnt_uvd=jnt_pixel;
jnt_xyz=convert_jntuvd_to_jntxyz(jnt_uvd);%%%
jnt_colors = rand(size(jnt_uvd,1), 3);
figure;
imshow(depth, [0, max(depth(:))]);
hold on;
%scatter(jnt_uvd(:,1), jnt_uvd(:,2), 20, jnt_colors, 'filled');
scatter(jnt_uvd(:,1), jnt_uvd(:,2), 50,'r','filled');
plot_uv_sides(jnt_uvd);

%% Visualize the depth in 3D
uvd = convert_depth_to_uvd(depth);
xyz = convert_uvd_to_xyz(uvd);
xyz1=xyz(:,:,1);
xyz2=xyz(:,:,2);
xyz3=xyz(:,:,3);
% Decimate the image (otherwise rendering will be too slow)
% decimation = 2;
% xyz_decimated = xyz(1:decimation:end, 1:decimation:end, :);
% points = reshape(xyz_decimated, size(xyz_decimated,1)*size(xyz_decimated,2), 3);
points = reshape(xyz, size(xyz,1)*size(xyz,2), 3);
% Collect the points within the AABBOX of the non-background points
body_points = points(find(points(:,3) < 2000),:);
axis_bounds = [min(body_points(:,1)) max(body_points(:,1)) ...
  min(body_points(:,3)) max(body_points(:,3)) ...
  min(body_points(:,2)) max(body_points(:,2))];
% Visualize the entire point cloud
figure;
plot3(body_points(:,1), body_points(:,3), body_points(:,2), '.', 'MarkerSize', 1.5);
%axis(axis_bounds);
view(5,5);
set(gcf,'renderer','opengl'); axis vis3d; axis equal;axis off; hold on; %grid on;

%% Visualize the hand and the joints in 3D
figure;
uvd = convert_depth_to_uvd(depth);
xyz = convert_uvd_to_xyz(uvd);
points = reshape(xyz, size(xyz,1)*size(xyz,2), 3);
%colors = reshape(rgb, size(rgb,1)*size(rgb,2), 3);
hand_points = squeeze(convert_uvd_to_xyz(reshape(jnt_uvd, 1, size(jnt_uvd,1), 3)));
% Collect the points within the AABBOX of the hand
axis_bounds = [min(hand_points(:,1)) max(hand_points(:,1)) min(hand_points(:,3)) max(hand_points(:,3)) min(hand_points(:,2)) max(hand_points(:,2))];
axis_bounds([1 3 5]) = axis_bounds([1 3 5]) - 50;
axis_bounds([2 4 6]) = axis_bounds([2 4 6]) + 50;
ipnts = find(points(:,1) >= axis_bounds(1) & points(:,1) <= axis_bounds(2) & points(:,2) >= axis_bounds(5) & points(:,2) <= axis_bounds(6) & points(:,3) >= axis_bounds(3) & points(:,3) <= axis_bounds(4));
points = points(ipnts, :);
%colors = double(colors(ipnts, :))/255;
%%
plot3(points(:,1), points(:,3), points(:,2), '.', 'MarkerSize', 1.5); 
set(gcf,'renderer','opengl'); axis vis3d; axis equal; hold on; %grid on;
%scatter3(hand_points(:,1), hand_points(:,3), hand_points(:,2), 50, jnt_colors, 'Fill','LineWidth', 0.5);
% scatter3(hand_points(:,1), hand_points(:,3), hand_points(:,2), 50, 'r', 'Fill','LineWidth', 0.5);
% hold on;
% plot_hand_sides(hand_points);
%set(gca,'layer','top');
view(0,0);hold on;

%axis(axis_bounds);
rand_ind = randperm(size(points,1),SAMPLE_NUM);
points_sampled00 = points(rand_ind,:);
figure;
plot3(points_sampled00(:,1), points_sampled00(:,3), points_sampled00(:,2), '.', 'MarkerSize', 3); 
view(0,0);hold on;
%% create OBB
[coeff,score,latent] = pca(points);
if coeff(2,1)<0
	coeff(:,1) = -coeff(:,1);
end
if coeff(3,3)<0
	coeff(:,3) = -coeff(:,3);
end
coeff(:,2)=cross(coeff(:,3),coeff(:,1));
ptCloud = pointCloud(points);
points_rotate = points*coeff;   
%figure;
%plot3(points_rotate(:,1), points_rotate(:,3), points_rotate(:,2), '.', 'MarkerSize', 1.5);
%% 2.4 sampling
if size(points,1)<SAMPLE_NUM
	tmp = floor(SAMPLE_NUM/size(points,1));
	rand_ind = [];
 	for tmp_i = 1:tmp
        rand_ind = [rand_ind 1:size(points,1)];
    end
	rand_ind = [rand_ind randperm(size(points,1), mod(SAMPLE_NUM, size(points,1)))];
else
	rand_ind = randperm(size(points,1),SAMPLE_NUM);
end
points_sampled = points(rand_ind,:);
points_rotate_sampled = points_rotate(rand_ind,:);
% figure;
% plot3(points_rotate_sampled(:,1), points_rotate_sampled(:,3), points_rotate_sampled(:,2), '.', 'MarkerSize', 1.5);
figure;
plot3(points_sampled(:,1), points_sampled(:,3), points_sampled(:,2), '.', 'MarkerSize', 5);
% set(gcf,'renderer','opengl'); axis vis3d; axis equal; hold on; grid on;
axis(axis_bounds);
%% 2.5 compute surface normal
normal_k = 30;
normals = pcnormals(ptCloud, normal_k);
normals_sampled = normals(rand_ind,:);
sensorCenter = [0 0 0];
for k = 1 : SAMPLE_NUM
    p1 = sensorCenter - points_sampled(k,:);
	% Flip the normal vector if it is not pointing towards the sensor.
	angle = atan2(norm(cross(p1,normals_sampled(k,:))),p1*normals_sampled(k,:)');
	if angle > pi/2 || angle < -pi/2
        normals_sampled(k,:) = -normals_sampled(k,:);
	end
end
normals_sampled_rotate = normals_sampled*coeff;%1024*3
%% 2.6 Normalize Point Cloud
x_min_max = [min(points_rotate(:,1)), max(points_rotate(:,1))];
y_min_max = [min(points_rotate(:,2)), max(points_rotate(:,2))];
z_min_max = [min(points_rotate(:,3)), max(points_rotate(:,3))];
scale = 1.2;
bb3d_x_len = scale*(x_min_max(2)-x_min_max(1));
bb3d_y_len = scale*(y_min_max(2)-y_min_max(1));
bb3d_z_len = scale*(z_min_max(2)-z_min_max(1));
max_bb3d_len = bb3d_x_len;

points_normalized_sampled = points_rotate_sampled/max_bb3d_len;
if size(hand_points,1)<SAMPLE_NUM
	offset = mean(points_rotate)/max_bb3d_len;
else
	offset = mean(points_normalized_sampled);
end
points_normalized_sampled = points_normalized_sampled - repmat(offset,SAMPLE_NUM,1);
%% 2.8 ground truth
pc = [points_normalized_sampled normals_sampled_rotate];
jnt_xyz_normalized = (jnt_xyz*coeff)/max_bb3d_len;
jnt_xyz_normalized = jnt_xyz_normalized - repmat(offset,JOINT_NUM,1);
%plot normalized sampled point cloud
figure;
%plot3(points_normalized_sampled(:,1), points_normalized_sampled(:,3), points_normalized_sampled(:,2), '.', 'MarkerSize', 5);
set(gcf,'renderer','opengl'); axis vis3d; axis equal; hold on; %grid on;
scatter3(points_normalized_sampled(:,1), points_normalized_sampled(:,3), points_normalized_sampled(:,2), 10, 'g', 'Fill','LineWidth', 0.1);
%scatter3(jnt_xyz_normalized(:,1), jnt_xyz_normalized(:,3), jnt_xyz_normalized(:,2), 50, jnt_colors, 'Fill','LineWidth', 0.5);
%plot_hand_sides(jnt_xyz_normalized);
%scatter3(jnt_xyz_normalized(:,1), jnt_xyz_normalized(:,3), jnt_xyz_normalized(:,2), 50, 'r', 'Fill','LineWidth', 0.5);
%view(180,190);
view(0,0);
%plot joints
figure;
set(gcf,'renderer','opengl'); 
%set(gca,'XColor','white');
%set(gca,'YColor','white'); 
%set(gca,'ZColor','white');
axis vis3d; axis equal; 
axis off;
hold on; grid on;
%plot_hand_sides(jnt_xyz_normalized);
%scatter3(jnt_xyz_normalized(:,1), jnt_xyz_normalized(:,3), jnt_xyz_normalized(:,2), 50, 'r', 'Fill','LineWidth', 2);
jnt_xyz_normalized1 = [jnt_xyz_normalized(:,2),jnt_xyz_normalized(:,1), jnt_xyz_normalized(:,3)];
plot_hand_sides_color(jnt_xyz_normalized1);
%scatter3(jnt_xyz_normalized(:,2), jnt_xyz_normalized(:,3), jnt_xyz_normalized(:,1), 50, 'r', 'Fill','LineWidth', 2);
view(350,10);

