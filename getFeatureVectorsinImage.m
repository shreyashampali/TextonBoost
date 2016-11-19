% Feature vectors for each pixel obtained by convolving with 17 different
% Filters. The filterd image is downsampled by a factor of
% 'downsample_fact' in each direction.
function fvs = getFeatureVectorsinImage(img, downsample_fact)
filt_size = 5; %5x5 filtes
sigma_gauss = [1,2,4,8];
% Generate the 17 gaussian filters with different variances
filters = generateFilters(filt_size, sigma_gauss);   

% img_rgb = imread('test.JPG','jpg');
% cform = makecform('srgb2lab');
% img = applycform(img_rgb,cform);

feature_vector_img = zeros(size(img,1), size(img,2), 17);   % 17D feature vectors
for i = 1:3
    feature_vector_img(:,:,(i-1)*3+1) = imfilter(double(img(:,:,1)),filters{i});
    feature_vector_img(:,:,(i-1)*3+2) = imfilter(double(img(:,:,2)),filters{i});
    feature_vector_img(:,:,(i-1)*3+3) = imfilter(double(img(:,:,3)),filters{i});
end

for i = 10:17
    feature_vector_img(:,:,i) = imfilter(double(img(:,:,1)),filters{i-6});
end

feature_vector_img = feature_vector_img(1:downsample_fact:size(img,1),1:downsample_fact:size(img,2),:);
fvs = reshape(permute(feature_vector_img,[3,2,1]),17,size(feature_vector_img,1)*size(feature_vector_img,2))';
end
% [IDX, C] = kmeans(fvs,10);
% texton_map = reshape(IDX, size(img,2), size(img,1));
% texton_map = texton_map';
% figure, imshow(uint8(texton_map*10))