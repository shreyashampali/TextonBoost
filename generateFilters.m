% Generate 11 Texton filters
function texton_filters = generateFilters(filt_size, sigma_gauss)
texton_filters = cell(11,1);

[X Y] = meshgrid(-2:2, -2:2);
% 3 Gaussian (sigma = 1,2,4) applied to L,a,b
for i = 1:3
    texton_filters{i} = fspecial('gaussian', [filt_size, filt_size], sigma_gauss(i));
end

% 4 derivatives of Gaussian (sigma 2,4) appied to L
for i = 2:3
    texton_filters{i+2} = fspecial('gaussian', [filt_size, filt_size], sigma_gauss(i)) .* ...
        (-2*X/(2*sigma_gauss(i)^2));
    texton_filters{i+2} = texton_filters{i+2}/sqrt(sum(sum(texton_filters{i+2}.^2)));
    
    texton_filters{i+4} = fspecial('gaussian', [filt_size, filt_size], sigma_gauss(i)) .* ...
        (-2*Y/(2*sigma_gauss(i)^2));
    texton_filters{i+4} = texton_filters{i+4}/sqrt(sum(sum(texton_filters{i+4}.^2)));
end

% Laplacian of Gauassian applied to L
for i = 1:4
    texton_filters{i+7} = fspecial('log', [filt_size, filt_size], sigma_gauss(i));
    texton_filters{i+7} = texton_filters{i+7}/sqrt(sum(sum(texton_filters{i+7}.^2)));
end
end

