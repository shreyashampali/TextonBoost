function labeliing = alphaExpansion(initializer, classes, unary, img)
% initializer : Initial labelling. Array of length num_pixels
% classes : an array of length C+1 representing C class labels. Label 0 represents not
% belonging to any of the classes
% unary : A matrix of size (C+1) x num_pixels representing unary potential
% values
% img : image. Used for getting pairwise portentials.
%
% labeling : Final labelling

% The code assumes 4 grid connection for pairwise potentials

%Include the matlab wrapper folder for alpha expansion
addpath('Bk_matlab\');

a1 = 45;
a2 = 0;
img = double(img);
unary = unary';



% Set the expansion move pattern to [classes classes] i.e., move twice over
% all the classes in the input order
expans_mov_pattern = [classes classes];

% Compute the pairwise potentials assuming pott's model
up_diff = sum(((img(2:end,:,:) - img(1:end-1,:,:)).^2),3);
beta = 1./(2*mean(up_diff(:)));
pott_up = a1*exp(-beta*up_diff)+a2;

left_diff = sum(((img(:,2:end,:) - img(:,1:end-1,:)).^2),3);
beta = 1./(2*mean(left_diff(:)));
pott_left = a1*exp(-beta*left_diff)+a2;

% Try each expansion in the expans_mov_pattern order
curr_state = reshape(double(initializer),size(img,1),size(img,2));
cnt = 1;
for alpha = expans_mov_pattern
    
    % Create the handle for the mex object
    h = BK_Create();

    % Create the nodes. = number of pixels in the image
    BK_AddVars(h,size(img,1)*size(img,2));
    
    % Compute the unary potential for 0 state and 1 state and store them
    % rowise in a matrix. Set the unary potenials using the API
    dc = unary(sub2ind([size(unary,1),size(unary,2)],curr_state(:)+1,(1:size(unary,2))'))';
    dc = [dc; unary(sub2ind([size(unary,1),size(unary,2)],repmat(alpha+1,size(unary,2),1),(1:size(unary,2))'))'];
    BK_SetUnary(h,dc);
    
    % The pairwise potentials are input using a upper triangular matrix.
    % Hence, updating the pairwise potential for bottom and right neighbors
    % for every pixel.
    tmpmat = ones(size(img,1),size(img,2));
    tmpmat(end,:) = 0;
    down_indx = find(tmpmat>0);
    tmpmat00 = abs(curr_state(2:end,:)-curr_state(1:end-1,:))>0;
    tmpmat01 = abs(alpha-curr_state(1:end-1,:))>0;
    tmpmat10 = abs(curr_state(2:end,:)-alpha)>0;
    D1 = [down_indx,down_indx+1,tmpmat00(:).*pott_up(:),tmpmat01(:).*pott_up(:),tmpmat10(:).*pott_up(:),zeros(size(down_indx,1),1)];
    
    tmpmat = ones(size(img,1),size(img,2));
    tmpmat(:,end) = 0;
    right_indx = find(tmpmat>0);
    tmpmat00 = abs(curr_state(:,2:end)-curr_state(:,1:end-1))>0;
    tmpmat01 = abs(alpha-curr_state(:,1:end-1))>0;
    tmpmat10 = abs(curr_state(:,2:end)-alpha)>0;
    D2 = [right_indx,right_indx+size(img,1),tmpmat00(:).*pott_left(:),tmpmat01(:).*pott_left(:),tmpmat10(:).*pott_left(:),zeros(size(right_indx,1),1)];
    
    D = [D1;D2];
    
    BK_SetPairwise(h,D);
    
    % Run the minimization
    e(cnt) = BK_Minimize(h);
    
    alphaexp = double(BK_GetLabeling(h))-1;
    
    % update the expansion
    curr_state(find(alphaexp==1)) = alpha;
    
    labeliing = uint8(reshape(curr_state,size(img,1),size(img,2)));
    
    BK_Delete(h);
    
    cnt = cnt + 1;
end

