function vbbPlayer( s, v )
% Simple GUI to play annotated videos (seq files)
%
% Uses the seqPlayer to display a video (seq file). See help for the
% seqPlayer for navigation options (playing forward/backward at variable
% speeds). Note that loading a new video via the file menu will not work
% (as the associated annotation is not loaded). The location of the videos
% and associated annotations (vbb files) are specified using dbInfo. To
% actually alter the annotations, use the vbbLabeler.
%
% USAGE
%  vbbPlayer( [s], [v] )
%
% INPUTS
%  s      - set index (randomly generated if not specified)
%  v      - vid index (randomly generated if not specified)
%
% OUTPUTS
%
% EXAMPLE
%  vbbPlayer
%
% See also SEQPLAYER, DBINFO, VBB, VBBLABELER, DBBROWSER
%
% Caltech Pedestrian Dataset     Version 3.2.1
% Copyright 2014 Piotr Dollar.  [pdollar-at-gmail.com]
% Licensed under the Simplified BSD License [see external/bsd.txt]

% -------------------------------------------------------
% dbInfo will return information of the specfied dataset.
% pth: the path of dataset
% setIds: the set ID of the specfied dataset. For example, if 'inriatest'
%         is specfied, the setIds will be 1 since INRIA has only two sets
%         in which set00 is the training set and set01 is the test set.
%         setIds = 1 means that the test set ( set01 ) is used.
% vidIds: the index of video file.
% -- by liyang.
[pth,setIds,vidIds] = dbInfo;

% randomly generate 's' and 'v' if running without parameters. The code below is used
% to check if there are one or two input parameters.
% -- by liyang.
if(nargin<1 ||isempty(s)), s=randint2(1,1,[1 length(setIds)]); end
if(nargin<2 ||isempty(v)), v=randint2(1,1,[1 length(vidIds{s})]); end
assert(s>0 && s<=length(setIds)); assert(v>0 && v<=length(vidIds{s}));

% s: the index of data to play. For example, set00 of INRIA is the 
% training set.
% v: the index of video file to play.
% -- by liyang.
vStr = sprintf('set%02i/V%03i',setIds(s),vidIds{s}(v));
fprintf('displaying vbb for %s\n',vStr);
A = vbb('vbbLoad', [pth '/annotations/' vStr] );
vbb( 'vbbPlayer', A, [pth '/videos/' vStr] );

end
