% verification of Eqn.(1) in paper 'Fast Feature Pyramids for Object Detection'
% Here, take 614 positive samples of INIRA for example
% Compute each image's sum of gradients and its counterpart's.
% Then, check if the ratio of two sum is round the upsample factor

INRIA_path = '/Users/Coldmoon/Datasets/INRIAPerson/Train/pos.lst';
k = 5;
filelist = importdata(INRIA_path);
ratios = zeros(length(filelist));
parfor i=1:length(filelist)
    fprintf('Processing the %dth image\n', i);
    I = imread(strcat('/Users/Coldmoon/Datasets/INRIAPerson/',filelist{i,1}));
    I = rgb2gray(I);
    J = imresize(I, k);
    [Gmag_I, Gdir_I] = imgradient(I,'prewitt');
    [Gmag_J, Gdir_J] = imgradient(J,'prewitt');
    Gmag_I_sum = sum(sum(Gmag_I));
    Gmag_J_sum = sum(sum(Gmag_J));
    ratios(i) = Gmag_J_sum / Gmag_I_sum;
end
histogram(ratios(:,1));