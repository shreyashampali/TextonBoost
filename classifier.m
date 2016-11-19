% Texton based classifier
% Uses trained data from gentleBoost.m

clear all
close all
if(1)
    % Unary classification using textons
    % Requires (refer to gentleBoost where these files are created) :
    % 1. kmeans_<class_name>.mat file containing the cluster centers after
    %    kmeans clustering. These centers are also used in training for
    %    generating the texton maps. Same cluster centers should be used here
    % 2. training_data_<class_name>.mat containing learnt features such as
    %    texton filter sizes learnt thresholds and coefficients
    cform = makecform('srgb2lab');
    fvs = [];
    recompute_kmeans = 0;
    classnum = 5;        %class code
    class_name = 'cow';  %class name
    set_name = 'testing';%testing or training directory
    class_code = [0 0 128];
    
    % Get the image names
    imageindices = 1:5;
    testing_set_size = 0;
    for i = 1:length(imageindices)
        imgname = strcat(class_name,'/',set_name,'/',num2str(classnum),'_',num2str(imageindices(i)),'_s.bmp');
        if exist(imgname)
            testing_set_size = testing_set_size+1;
        end
    end
    
    % Load the texton cluster centers
    load(strcat('kmeans_',class_name,'.mat'));
    
%     texton_map = zeros(size(img,1),size(img,2),testing_set_size);
%     class_map = zeros(size(img,1),size(img,2),testing_set_size);
    % Read the images
    j = 1;
    for i = 1:length(imageindices)
        imgname = strcat(class_name,'/',set_name,'/',num2str(classnum),'_',num2str(imageindices(i)),'_s.bmp');
        img_GT_name = strcat(class_name,'/',set_name,'/',num2str(classnum),'_',num2str(imageindices(i)),'_s_GT.bmp');
        if exist(imgname)
            img_rgb(:,:,:,j) = imread(imgname,'bmp');
            img = applycform(img_rgb(:,:,:,j),cform);
            texton_map(:,:,j) = getTextonMap(img, C);
            
            img_GT_rgb = imread(img_GT_name,'bmp');
            class_map_tmp = double((img_GT_rgb(:,:,1)==class_code(1))&(img_GT_rgb(:,:,2)==class_code(2))&(img_GT_rgb(:,:,3)==class_code(3)));
            class_map_tmp(class_map_tmp==0) = -1;
            class_map(:,:,j) = class_map_tmp;
            
            j = j+1;
        end
    end
    
    % Load the learnt features
    load(strcat('training_data_',class_name));
    
    % Generate the grid coordintes for the training samples across all the
    % images. X is downwards, Y is rightwards
    [Ycod Xcod Zcod] = meshgrid(1:1:size(img,2),1:1:size(img,1),1:testing_set_size);
    E = 0;
    
    % Generate unary confidence
    for boost_round = 1:length(final_theta)
        % Loop over all the texton filters over all the training samples
        filter_idx = final_filter_idx(boost_round);
        texton = final_texton(boost_round);
        theta = final_theta(boost_round);
        a = final_a(boost_round);
        b = final_b(boost_round);
        fprintf('K = %d, filter_idx = %d, round = %d\n',texton,filter_idx,boost_round);
        % Loop over all the thresholds in the current texton and filter over all the training samples
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
        int_tl = integral_texton_map(sub2ind([size(img,1) size(img,2), testing_set_size], filter_tl_pos_x(:),filter_tl_pos_y(:),Zcod(:)));
        int_tr = integral_texton_map(sub2ind([size(img,1) size(img,2), testing_set_size], filter_tr_pos_x(:),filter_tr_pos_y(:),Zcod(:)));
        int_bl = integral_texton_map(sub2ind([size(img,1) size(img,2), testing_set_size], filter_bl_pos_x(:),filter_bl_pos_y(:),Zcod(:)));
        int_br = integral_texton_map(sub2ind([size(img,1) size(img,2), testing_set_size], filter_br_pos_x(:),filter_br_pos_y(:),Zcod(:)));
        
        % Compute the ratio of textons in the filter
        texton_ratio = (int_br - int_tr - int_bl + int_tl)/(filter_sizes(filter_idx,1)*filter_sizes(filter_idx,2));
        
        % Weak learner prediction
        E = E + a*(texton_ratio>theta) + b;
        
        %temp
        %             tmpmat = reshape(a*(texton_ratio>theta) + b,size(img,1),size(img,2),testing_set_size);
        %             wlforimg(:,:,boost_round) = tmpmat;
    end
    
    confidence_map = reshape(E,size(img,1),size(img,2),testing_set_size);
    cm = (class_map+1)/2;
    sum(sum(sum((confidence_map>0)==cm)))/prod(size(class_map))
    
else
    % Bypass (for debugging only)
    load car_classifier;
end

% Incorporate the unary classified results in the CRF. 
% Perform alpha expansion to compute the new labelling.
figure
for i = 1:testing_set_size
    i
    ini_label = confidence_map(:,:,i)>0;
    tmpmat = confidence_map(:,:,i);
    labeliing(:,:,i) = alphaExpansion(ini_label(:), [0 1], [tmpmat(:), -tmpmat(:)], img_rgb(:,:,:,i));
    tmpmat = img_rgb(:,:,:,i);
    tmpmat(find(labeliing(:,:,i)==1)) = 255;
    figure(1),imshow(uint8(tmpmat));
    pause
end

% Print some stats
cm = (class_map+1)/2;
sum(sum(sum((labeliing==cm))))/prod(size(class_map))

[sum(labeliing(find(cm==1))==1)/length(find(cm==1)),sum(labeliing(find(cm==1))==0)/length(find(cm==1));...
    sum(labeliing(find(cm==0))==1)/length(find(cm==0)),sum(labeliing(find(cm==0))==0)/length(find(cm==0))]*100