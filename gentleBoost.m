clear all
close all

addpath('car/training/');
classnum = 7;
class_code = [64 0 128];

num_boost = 700;
downsample_for_knn = 2;
downsample_for_training = 5;
K = 100;
num_texton_filters = 100;

imageindices = 1:30;

cform = makecform('srgb2lab');
fvs = [];
recompute_kmeans = 1;

training_set_size = 0;
for i = 1:length(imageindices)
    imgname = strcat(num2str(classnum),'_',num2str(imageindices(i)),'_s.bmp');
    if exist(imgname)
        i
        img_rgb = imread(imgname,'bmp');
        img = applycform(img_rgb,cform);
        fvs = [fvs; getFeatureVectorsinImage(img, downsample_for_knn)];
        training_set_size = training_set_size+1;
    end
end

if recompute_kmeans==1
    opts = statset('Display','iter');
    [IDX, C] = kmeans(fvs,K,'Options',opts,'start','cluster','MaxIter',200);
else
    load('kmeans_car.mat');
end

texton_map = zeros(size(img,1),size(img,2),training_set_size);
class_map = zeros(size(img,1),size(img,2),training_set_size);
j = 1;
% get the textop map for each of the images
for i = 1:length(imageindices)
    imgname = strcat(num2str(classnum),'_',num2str(imageindices(i)),'_s.bmp');
    img_GT_name = strcat(num2str(classnum),'_',num2str(imageindices(i)),'_s_GT.bmp');
    if exist(imgname)
        img_rgb = imread(imgname,'bmp');
        img = applycform(img_rgb,cform);
        texton_map(:,:,j) = getTextonMap(img, C);
        
        img_GT_rgb = imread(img_GT_name,'bmp');
        class_map_tmp = double((img_GT_rgb(:,:,1)==class_code(1))&(img_GT_rgb(:,:,2)==class_code(2))&(img_GT_rgb(:,:,3)==class_code(3)));
        class_map_tmp(class_map_tmp==0) = -1;
        class_map(:,:,j) = class_map_tmp;
        
        j = j+1;
    end
end

% Generate  num_texton_filters filters in a -100:100 vicinity of a pixel.
% Vary the size of the filter from 5:20 in each direction. The size and the
% location of the filter are chosen from a uniform distribution
[filter_locs,filter_sizes] = generateTextonFilter(num_texton_filters,[-40,40],[5,20]); % filter_sizes = [downwards,rightwards]

% Generate the grid coordintes for the training samples across all the
% images. X is downwards, Y is rightwards
[Ycod Xcod Zcod] = meshgrid(1:downsample_for_training:size(img,2),1:downsample_for_training:size(img,1),1:training_set_size);

% Weights for all the training samples. Initialized with one
w = ones(size(Ycod,1)*size(Ycod,2)*training_set_size,1);

% Classes of the training samples in all training images
z = class_map(sub2ind([size(img,1) size(img,2), training_set_size], Xcod(:), Ycod(:), Zcod(:)));

cnt = 1;
min_filter_idx = 0;
min_texton = 0;
min_theta = 0;
min_a = 0;
min_b = 0;


% Start the learning process
for boost_round = 1:num_boost
    random_feature_sel_size = 80;
    random_filter_idx = round(random('uniform',1,num_texton_filters,random_feature_sel_size,1));
    random_texton = round(random('uniform',1,K,random_feature_sel_size,1));
    % Loop over all the texton filters over all the training samples
    min_J = inf;
    for feature_idx = 1:random_feature_sel_size
        filter_idx = random_filter_idx(feature_idx);
        texton = random_texton(feature_idx);
        % Loop over all the textons in the current filter over all the training samples
        %         for texton = 1:K
        fprintf('K = %d, filter_idx = %d, round = %d\n',texton,filter_idx,boost_round);
        
        %                 tic
        % Get the Bounding box indices for the current texton filter in
        % all the training images
        filter_tl_pos_x = min(max(filter_locs(filter_idx,1) + Xcod,1),size(img,1)) ;
        filter_tl_pos_y = min(max(filter_locs(filter_idx,2) + Ycod,1),size(img,2)) ;
        filter_tr_pos_x = min(max(filter_locs(filter_idx,1) + Xcod,1),size(img,1));
        filter_tr_pos_y = min(max(filter_locs(filter_idx,2) + filter_sizes(filter_idx,2) + Ycod,1),size(img,2));
        filter_bl_pos_x = min(max(filter_locs(filter_idx,1) + filter_sizes(filter_idx,1) + Xcod,1),size(img,1));
        filter_bl_pos_y = min(max(filter_locs(filter_idx,2) + Ycod,1),size(img,2));
        filter_br_pos_x = min(max(filter_locs(filter_idx,1) + filter_sizes(filter_idx,1) + Xcod,1),size(img,1));
        filter_br_pos_y = min(max(filter_locs(filter_idx,2) + filter_sizes(filter_idx,2) + Ycod,1),size(img,2));
        
        % Compute the integral of the texton for the current texton
        integral_texton_map = cumsum(cumsum(texton_map==texton,1),2);
        
        % Count the number of textons in the current filter in all the
        % training images
        int_tl = integral_texton_map(sub2ind([size(img,1) size(img,2), training_set_size], filter_tl_pos_x(:),filter_tl_pos_y(:),Zcod(:)));
        int_tr = integral_texton_map(sub2ind([size(img,1) size(img,2), training_set_size], filter_tr_pos_x(:),filter_tr_pos_y(:),Zcod(:)));
        int_bl = integral_texton_map(sub2ind([size(img,1) size(img,2), training_set_size], filter_bl_pos_x(:),filter_bl_pos_y(:),Zcod(:)));
        int_br = integral_texton_map(sub2ind([size(img,1) size(img,2), training_set_size], filter_br_pos_x(:),filter_br_pos_y(:),Zcod(:)));
        
        % Compute the ratio of textons in the filter
        texton_ratio = (int_br - int_tr - int_bl + int_tl)/(filter_sizes(filter_idx,1)*filter_sizes(filter_idx,2));
        
        % Loop over all the thresholds in the current texton and filter over all the training samples
        for theta = linspace(0,0.75*max(texton_ratio),20)
            % Compute a and b parameters
            if((sum(texton_ratio>theta)==0))
                break;
            end
            b = sum(w.*z.*(texton_ratio<=theta))/sum(w.*(texton_ratio<=theta));
            a = sum(w.*z.*(texton_ratio>theta))/sum(w.*(texton_ratio>theta)) - b;
            
            % Weak learner prediction
            h = a*(texton_ratio>theta) + b;
            
            % Compute the cost
            J = sum(w.*((z - h).^2));
            
            if(J<=min_J)
                if (theta>0)
                    theta = theta;
                end
                min_J = J;
                min_filter_idx = filter_idx;
                min_texton = texton;
                min_theta = theta;
                min_a = a;
                min_b = b;
                min_w_factor = exp(-z.*h);
            end
            %                 toc
            %             end
        end
    end
    final_filter_idx(boost_round) = min_filter_idx;
    final_texton(boost_round) = min_texton;
    final_theta(boost_round) = min_theta;
    final_a(boost_round) = min_a;
    final_b(boost_round) = min_b;
    
    % Update the weights
    w = w.*min_w_factor;
    
    fprintf('max_w = %d,  avg_w = %d pc_fail = %d\n',max(w),mean(w),length(find(w>=1))/length(w))
end

% save the learnt feature which is used while classification 
save(strcat('training_data_','car'),'filter_locs','filter_sizes','final_filter_idx','final_texton','final_theta','final_a','final_b')
save('kmeans_car','IDX','C')