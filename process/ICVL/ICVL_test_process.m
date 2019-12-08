% create point cloud from depth image
% dataset: ICVL
% author: Yujin Chen
clc;clear;close all;clc; rng(0);

dataset_dir = 'F:\DeepLearningDataset\ICVL\old\Testing\Depth\';
save_dir = 'F:\DeepLearningDataset\ICVL\old\Testing\process_data\';
subject_names={'test_seq_1','test_seq_2'};
% image_indexs = 10000;
% start_index=70000;
% image_indexs = 2757;
indexs = 2;
SAMPLE_NUM = 1024;
JOINT_NUM = 16;

% jnt_pixel=squeeze(All_joints(image_index+1,3:50));
% jnt_pixel=(reshape(jnt_pixel,3,size(jnt_pixel,2)/3))';
%% Process depth images
for index = 1:indexs
    % Load joints coordinates
    txt_dir=[dataset_dir int2str(index) '.txt'];
    All_joints = importdata(txt_dir);
    image_indexs=size(All_joints,1);
    Point_Cloud_FPS = zeros(image_indexs,SAMPLE_NUM,6);
	Volume_rotate = zeros(image_indexs,3,3);
	Volume_length = zeros(image_indexs,1);
 	Volume_offset = zeros(image_indexs,3);
	Volume_GT_XYZ = zeros(image_indexs,JOINT_NUM,3);
    for image_index = 1:image_indexs
        if mod(image_index,100)==0
            fprintf('Processing set-%d: image %d of %d \n',index,image_index,image_indexs);
        end
        image_name=All_joints(image_index,2);
        filename_prefix = sprintf('%04d',image_name);
        %Load depth image
        depth = imread([dataset_dir,subject_names{index},'\', 'image_', filename_prefix, '.png']);

        %Get joints information
        jnt_pixel = squeeze(All_joints(image_index, 3:50));
        jnt_pixel=(reshape(jnt_pixel,3,size(jnt_pixel,2)/3))';
        jnt_uvd=jnt_pixel;
        jnt_xyz=convert_jntuvd_to_jntxyz(jnt_uvd);%%%
        jnt_colors = rand(size(jnt_uvd,1), 3);
%         figure;
%         imshow(depth, [0, max(depth(:))]);
%         hold on;
%         
%         scatter(jnt_uvd(:,1), jnt_uvd(:,2), 20, jnt_colors, 'filled');      
        %% Visualize the hand and the joints in 3D
%         figure;
        uvd = convert_depth_to_uvd(depth);
        xyz = convert_uvd_to_xyz(uvd);
        points = reshape(xyz, size(xyz,1)*size(xyz,2), 3);
        
        % Collect the points within the AABBOX of the non-background points
        body_points = points(find(points(:,3) < 2000),:);
        axis_bounds = [min(body_points(:,1)) max(body_points(:,1)) ...
          min(body_points(:,3)) max(body_points(:,3)) ...
          min(body_points(:,2)) max(body_points(:,2))];
        % Visualize the entire point cloud
%         figure;
%         plot3(body_points(:,1), body_points(:,3), body_points(:,2), '.', 'MarkerSize', 1.5);
%         axis(axis_bounds);
%         view(45,20);
%         set(gcf,'renderer','opengl'); axis vis3d; axis equal; hold on; grid on;
        % Visualize the hand and the joints in 3D
        
        % Collect the points within the AABBOX of the hand
        hand_points = squeeze(convert_uvd_to_xyz(reshape(jnt_uvd, 1, size(jnt_uvd,1), 3)));
        % Collect the points within the AABBOX of the hand
        axis_bounds = [min(hand_points(:,1)) max(hand_points(:,1)) min(hand_points(:,3)) max(hand_points(:,3)) min(hand_points(:,2)) max(hand_points(:,2))];
        axis_bounds([1 3 5]) = axis_bounds([1 3 5]) - 20;
        axis_bounds([2 4 6]) = axis_bounds([2 4 6]) + 20;
        ipnts = find(points(:,1) >= axis_bounds(1) & points(:,1) <= axis_bounds(2) & points(:,2) >= axis_bounds(5) & points(:,2) <= axis_bounds(6) & points(:,3) >= axis_bounds(3) & points(:,3) <= axis_bounds(4));
        points = points(ipnts, :);
        
%         plot3(points(:,1), points(:,3), points(:,2), '.', 'MarkerSize', 1.5); 
%         set(gcf,'renderer','opengl'); axis vis3d; axis equal; hold on; grid on;
%         scatter3(hand_points(:,1), hand_points(:,3), hand_points(:,2), 50, jnt_colors, 'Fill','LineWidth', 0.5);
%         axis(axis_bounds);

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
%         figure;
%         plot3(points_rotate(:,1), points_rotate(:,3), points_rotate(:,2), '.', 'MarkerSize', 1.5);
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
%         figure;
%         plot3(points_rotate_sampled(:,1), points_rotate_sampled(:,3), points_rotate_sampled(:,2), '.', 'MarkerSize', 1.5);
%         figure;
%         plot3(points_sampled(:,1), points_sampled(:,3), points_sampled(:,2), '.', 'MarkerSize', 5);
%         set(gcf,'renderer','opengl'); axis vis3d; axis equal; hold on; grid on;
%         axis(axis_bounds);
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
%         figure;
%         plot3(points_normalized_sampled(:,1), points_normalized_sampled(:,3), points_normalized_sampled(:,2), '.', 'MarkerSize', 5);
%         set(gcf,'renderer','opengl'); axis vis3d; axis equal; hold on; grid on;
%         scatter3(jnt_xyz_normalized(:,1), jnt_xyz_normalized(:,3), jnt_xyz_normalized(:,2), 50, jnt_colors, 'Fill','LineWidth', 0.5);
%         view(180,190);
        X1=sqrt(power((jnt_xyz(1,1)-jnt_xyz(2,1)),2)+power((jnt_xyz(1,2)-jnt_xyz(2,2)),2)+power((jnt_xyz(1,3)-jnt_xyz(2,3)),2));
        X2=sqrt(power((jnt_xyz_normalized(1,1)-jnt_xyz_normalized(2,1)),2)+power((jnt_xyz_normalized(1,2)-jnt_xyz_normalized(2,2)),2)+power((jnt_xyz_normalized(1,3)-jnt_xyz_normalized(2,3)),2));
        Point_Cloud_FPS(image_index,:,:) = pc;
        Volume_rotate(image_index,:,:) = coeff;
        Volume_length(image_index) = max_bb3d_len;
        Volume_offset(image_index,:) = offset;
        Volume_GT_XYZ(image_index,:,:) = jnt_xyz_normalized;
    end
    %% save files
    save_k_dir=[save_dir  num2str(index)];
%     save([save_k_dir '_Test_Point_Cloud.mat'],'Point_Cloud_FPS');
%     save([save_k_dir '_Test_Volume_rotate.mat'],'Volume_rotate');
%     save([save_k_dir '_Test_Volume_length.mat'],'Volume_length');
%     save([save_k_dir '_Test_Volume_offset.mat'],'Volume_offset');
%     save([save_k_dir '_Test_Volume_GT_XYZ.mat'],'Volume_GT_XYZ');
%     save([save_gesture_dir '/valid.mat'],'valid'); 
end
    

