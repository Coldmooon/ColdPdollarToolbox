function [lambdas,as,scales,fs] = chnsScaling( pChns, Is, show )
% Compute lambdas for channel power law scaling.
%
% For a broad family of features, including gradient histograms and all
% channel types tested, the feature responses computed at a single scale
% can be used to approximate feature responses at nearby scales. The
% approximation is accurate at least within an entire scale octave. For
% details and to understand why this unexpected result holds, please see:
%   P. Doll?r, R. Appel, S. Belongie and P. Perona
%   "Fast Feature Pyramids for Object Detection", PAMI 2014.
%
% This function computes channels at multiple image scales and plots the
% resulting power law scaling. The purpose of this function is two-fold:
% (1) compute lambdas for fast approximate channel computation for use in
% chnsPyramid() and (2) provide a visualization of the power law channel
% scaling described in the BMVC2010 paper.
%
% chnsScaling() takes two main inputs: the parameters for computing image
% channels (pChns), and an image or set of images (Is). The images are
% cropped to the dimension of the smallest image for simplicity of
% computing the lambdas (and fairly high resolution images are best). The
% computed lambdas will depend on the channel parameters (e.g. how much
% smoothing is performed), but given enough images (>1000) the computed
% lambdas should not depend on the exact images used.
%
% USAGE
%  [lambdas,as,scales,fs] = chnsScaling( pChns, Is, [show] )
%
% INPUTS
%  pChns          - parameters for creating channels (see chnsCompute.m)
%  Is             - [nImages x 1] cell array of images (nImages may be 1)
%  show           - [1] figure in which to display results
%
% OUTPUTS
%  lambdas        - [nTypes x 1] computed lambdas
%  as             - [nTypes x 1] computed y-intercepts
%  scales         - [nScales x 1] vector of actual scales used
%  fs             - [nImages x nScales x nTypes] array of feature means
%
% EXAMPLE
%  sDir = 'data/Inria/train/neg/';
%  Is = fevalImages( @(x) {x}, {}, sDir, 'I', 'png', 0, 200 );
%  p = chnsCompute(); lambdas = chnsScaling( p, Is, 1 );
%
% See also chnsCompute, chnsPyramid, fevalImages
%
% Piotr's Computer Vision Matlab Toolbox      Version 3.25
% Copyright 2014 Piotr Dollar & Ron Appel.  [pdollar-at-gmail.com]
% Licensed under the Simplified BSD License [see external/bsd.txt]

% get additional input arguments
if(nargin<3 || isempty(show)), show=1; end

% construct pPyramid (don't pad, concat or appoximate)
pPyramid=chnsPyramid(); 
pPyramid.pChns=pChns; 
pPyramid.concat=0;
pPyramid.pad=[0 0]; 
pPyramid.nApprox=0; 
pPyramid.smooth=0;
pPyramid.minDs(:)=max(8,pChns.shrink*4);

% crop all images to smallest image size
ds=[inf inf]; 
nImages=numel(Is);
for i=1:nImages, 
    ds=min(ds,[size(Is{i},1) size(Is{i},2)]); 
end
% ds=round(ds/pChns.shrink)*pChns.shrink;
ds=floor(ds/pChns.shrink)*pChns.shrink; % for INRIA train pos. --by liyang 
for i=1:nImages, 
    Is{i}=Is{i}(1:ds(1),1:ds(2),:); 
end

% compute fs [nImages x nScales x nTypes] array of feature means
% ------------------------------
% Compute the mean features for each image, each scale, and each type of feature.
% Imagine that the images lie on the first dimension, the scales lie on
% the second dimension, and the feature types lie on the third dimension.
% This structure is similar to Caffe. We can treat scales and feature types
% as a virtual "image", where scales are row and feature types are column,
% and each element in this structure is a mean of features.
% -- by liyang.
P=chnsPyramid(Is{1},pPyramid); 
scales=P.scales'; 
info=P.info;
nScales=P.nScales; 
nTypes=P.nTypes; 
fs=zeros(nImages,nScales,nTypes);
parfor i=1:nImages, 
    P=chnsPyramid(Is{i},pPyramid); 
    for j=1:nScales
        for k=1:nTypes, 
            fs(i,j,k)=mean(P.data{j,k}(:)); 
        end; 
    end; 
end

% remove fs with fs(:,1,:) having small values
% --------------------------------------------
% Continue to imagine. There are 28 layers in the structure described
% above. Each layer is corresponding to a scale. For the original scale,
% that is scales(1), we can get the first layer's features for all images
% and all the types of features. 'kp' is used to do this. Therefore,
% 'fs(:, 1, :)' means all the elements in the first layer. We are
% interested in the max elements in the first layer. There are three max 
% elements corresponding to three columns respectively. Next, we intend to
% compare each element in the first layer with the max element for each
% type of feature, that is column comparisons. 'kp(ones(1,nImages),1,:)'
% make 201 copys of the max elements, which are corresponding to the
% elements in the first one by one, then divided by 50, and finally compare
% with the elements in the first layer.
% -- by liyang.
kp=max(fs(:,1,:)); 
max_layer = kp(ones(1,nImages),1,:)/50;
% For the first layer, find the elements which are not less than 50% of max
% values. -- by liyang
kp=fs(:,1,:) > max_layer; 
% As long as there is a element which is less than the 50% max value, the image
% is excluded. Just as a saying in China, a smelly meat brings the entire
% pot bad. -- by liyang.
kp=min(kp,[],3); 
% Only keep the images whose elements are bigger than 50% max.-- by liyang.
fs=fs(kp,:,:); 
% Count the number of images we kept. -- by liyang.
nImages=size(fs,1);

% compute ratios, intercepts and lambdas using least squares
scales1=scales(2:end); 
nScales=nScales-1; 
O=ones(nScales,1);
% ----------------------------------------------------------------------
% fs(:,2:end,:) means getting the second to the last layer in the feature
% pyramid. fs(:,O,:) means expending the first layer to 27 layers which are
% corresponding to the 2:end layer by layer. Therefore, rs is the ratio
% between the first layer and the remaining layers.
% mean(rs, 1) averages the 201 images, which results in a mean average feature
% map. -- by liyang.
rs=fs(:,2:end,:)./fs(:,O,:); 
mus=permute(mean(rs,1),[2 3 1]); % Corresponding to Eqn.(5) of paper. 
% ----------------------------------------------------------------------
% There are a lot of stories in the next line. First, the symbol "\" has
% already contained the least squares computation, see "help \". So, no need
% to manually compute the least squares. Second, what is the "O" in 
% [O -log2(scales1)]? Note that in [O -log2(scales1)], the symbol "-" is 
% not the meaning of minus. There is a space between O and -log2(scales1), 
% therefore [O -log2(scales1)] means making a matrix by O and
% -log2(scales1). Finally, why is there a O in [O -log2(scales1)] ? To
% answer this, we need to review the paper. In page 1537, it is said that 
% 
% u_s = a_ * s^(-lambda)  =>  log2(u_s) = log2(a_) -lambda*log2(s).
% 
% If change this formula to the form of matrix, we get:
% 
% log2(u_s) = [1  -lambda] * [log2(a_)  log2(s)]. 
% 
% See? This is exactly the thing that the next line means. By this formula,
% 'a_' and 'lambda' can be computed together.
% -- by liyang.
out=[O -log2(scales1)]\log2(mus); 
as=2.^out(1,:); 
lambdas=out(2,:);
if(0), 
    lambdas=-log2(scales1)\log2(mus); 
    as(:)=1; 
end
if(show==0), 
    return; 
end

% compute predicted means and errors for display purposes
% -------------------------------------------------------
% musp: the best-fit reates of features at the different scales using the
%       computed 'lamda' and 'as'. The rates are mean value since the lamda
%       and as are computed over all the images.
%
% errsFit: $\xi$ in Eqn.(4), the deviation from the power law. See the
% paragraph beginning with "There is strong agreement between the resulting
% best-fit lines and the observations.". You will find a equation:
% \left | E[\xi] \right | = \left | \mu_s - a_{\Omega}s^{-\lambda_{\Omega} } \right |
%
% mus: The observation or the real rates of features at the different scales.
%
% errsFits is corresponding to the section 4.2 and used to verify the
% expection of $\xi$ is very small and close to zero. stds is corresponding
% to the section 4.3 and used to show the deviation from power law for a
% single image.
% -- by liyang.
musp=as(O,:).*scales1(:,ones(1,nTypes)).^-lambdas(O,:);
errsFit=mean(abs(musp-mus)); 
stds=permute(std(rs,0,1),[2 3 1]);

% plot results
if(show<0), 
    show=-show; 
    clear=0; 
else
    clear=1; 
end
figureResized(.75,show); 
if(clear), 
    clf; 
end
lp={'LineWidth',2}; 
tp={'FontSize',12};
for k=1:nTypes
  % plot ratios
  subplot(2,nTypes,k); 
  set(gca,tp{:});
  % In Fig.3 of paper, we can see " ... for 20 randomly selected pedestrian
  % images are shown as faint gray linees. ..." -- by liyang.
  for i=round(linspace(1,nImages,20))
    % used to plot the \textbf{real} rate of features v.s. the rate of scale 
    % for selected 20 images. -- by liyang.
    loglog(1./scales1,rs(i,:,k),'Color',[1 1 1]*.8); hold on; 
  end
  % used to plot the real mean rates over all the images. -- liyang.
  h0=loglog(1./scales1,mus(:,k),'go',lp{:});
  % used to plot the estimated mean rates. You will find that it is a
  % straight line! This is because we estimate the rates of features at the
  % different scales by a power relationship. -- by liyang.
  h1=loglog(1./scales1,musp(:,k),'b-',lp{:});
  title(sprintf('%s\n\\lambda = %.03f,  error = %.2e',...
    info(k).name,lambdas(k),errsFit(k)));
  legend([h0 h1],{'real','fit'},'location','ne');
  xlabel('log2(scale)');
  ylabel('\mu (ratio)'); 
  axis tight;
  ax=axis; 
  ax(1)=1; ax(3)=min(.9,ax(3)); ax(4)=max(2,ax(4)); 
  axis(ax);
  set(gca,'ytick',[.5 1 1.4 2 3 4],'YMinorTick','off');
  set(gca,'xtick',2.^(-10:.5:10),'XTickLabel',10:-.5:-10);
  % plot variances
  subplot(2,nTypes,k+nTypes); 
  set(gca,tp{:});
  semilogx(1./scales1,stds(:,k),'go',lp{:}); 
  hold on;
  xlabel('log2(scale)'); 
  ylabel('\sigma (ratio)'); 
  axis tight;
  ax=axis; ax(1)=1; ax(3)=0; ax(4)=max(.5,ax(4));
  axis(ax);
  set(gca,'xtick',2.^(-10:.5:10),'XTickLabel',10:-.5:-10);
end

end
