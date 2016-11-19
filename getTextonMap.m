% Generates the texton map for the input image 'img' using the cluster
% centers provided by 'C'
function texton_map = getTextonMap(img, C)
    fvs = getFeatureVectorsinImage(img, 1); % get the features for each pixel(17 filter responses per pixel)
    [IDX, D] = knnsearch(C, fvs);
    texton_map = reshape(IDX, size(img,2), size(img,1));
    texton_map = texton_map';
end