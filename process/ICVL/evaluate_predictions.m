% Jonathan Tompson, New York University - 8/28/2014
% This is a simple example script to evaluate our convnet predictions
clearvars; close all; clc; rng(0);

dataset_dir = 'dataset\test\';
load([dataset_dir, 'joint_data.mat']);
load([dataset_dir, 'test_predictions.mat']);

% Not all joints are tracked by the convnet.  Print which ones are.
disp('Joints tracked by convnet:')
disp(conv_joint_names);
disp('Joints in database:')
disp(joint_names);

nimgs = size(joint_uvd, 2);

% Plot some examples
figure;
set(gcf, 'Position', [200, 200, 800, 600]);
nplots = 4;
examples = randi(nimgs, 1, nplots);
i_nonzero = find(pred_joint_uvconf(1,1,:,1) ~= 0);
cmap = jet(length(i_nonzero)); % Make 1000 colors.
for i = 1:nplots
  subplot(floor(sqrt(nplots)),ceil(sqrt(nplots)),i);
  pred_jnt_uv = squeeze(pred_joint_uvconf(1,examples(i),:,1:2));
  gt_jnt_uv = squeeze(joint_uvd(1,examples(i),:,1:2));
  scatter(pred_jnt_uv(i_nonzero,1), pred_jnt_uv(i_nonzero,2), 50, cmap, 'o'); 
  hold on;
  scatter(gt_jnt_uv(i_nonzero,1), gt_jnt_uv(i_nonzero,2), 50, cmap, 'x');
  title(['Frame ', num2str(examples(i))]);
  hLegend = legend({'pred','gt'}, 'FontSize',8);
  hMarkers = findobj(hLegend, 'type', 'patch');
  set(hMarkers, 'MarkerEdgeColor','k', 'MarkerFaceColor','k');
  axis([min(gt_jnt_uv(:,1))-40, max(gt_jnt_uv(:,1))+40, ...
    min(gt_jnt_uv(:,2))-40, max(gt_jnt_uv(:,2))+40]);
  grid on;
end

% Plot the performance for each joint
thresholds = logspace(-1,log10(20),50);
accuracy = zeros(length(i_nonzero), length(thresholds));
uv_err = squeeze(sqrt(sum((pred_joint_uvconf(1,:,:,1:2) - joint_uvd(1,:,:,1:2)).^2, 4)));
% TODO: This could probably be vectorized...
for t = 1:length(thresholds)
  for j = 1:length(i_nonzero)
    accuracy(j,t) = 100 * sum(uv_err(:,i_nonzero(j)) <= thresholds(t)) / nimgs;
  end
end

figure;
plot(thresholds, accuracy);
grid on;
set(gcf, 'Position', [300, 300, 800, 600]);
set(gca, 'FontSize', 12);
xlabel('Threshold (pix)');
ylabel('Percentage of examples within Threshold');
legend(joint_names(i_nonzero), 'Location', 'SouthEast', 'FontSize', 8, 'Interpreter', 'none');
