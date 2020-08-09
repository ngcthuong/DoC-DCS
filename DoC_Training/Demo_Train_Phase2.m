% The training data is generated by '[imdb] = generatepatches;' in line 126 of 'DnCNN_train_dag.m'.
addpath('D:\matconvnet-1.0-beta25\matlab\mex');
addpath('D:\matconvnet-1.0-beta25\matlab\simplenn');
% addpath('D:\matconvnet-1.0-beta25\matlab');

rng('default')
addpath('utilities');
%-------------------------------------------------------------------------
% Configuration
%-------------------------------------------------------------------------
global noLayer subRate blk_size   channel epoch
noLayer               = 5; 
subRate               = 0.2;
blk_size              = 16 ;
channel               = 4; 
epoch                 = 100;

opts.learningRate     = [logspace(-3,-3,50) logspace(-3,-3,50)./2 logspace(-3,-3,40)./4 logspace(-3,-3,40)./8];% you can change the learning rate
opts.batchSize        = 32; % 
opts.gpus             = [1]; 
opts.numSubBatches    = 2;
opts.modelName        = ['W2_DoG_CSNet' num2str(noLayer) '_r' num2str(subRate) '_blk' num2str(blk_size) '_mBat' num2str(opts.batchSize)]; %%% model name

% solver
opts.solver           = 'Adam'; % global
opts.derOutputs       = {'objective',1} ;

opts.backPropDepth    = Inf;
%-------------------------------------------------------------------------
%   Initialize model
%-------------------------------------------------------------------------

net  = feval(['DoG_CSNet_Init_Phase2']);

%-------------------------------------------------------------------------
%   Train
%-------------------------------------------------------------------------

[net, info] = MWCNN_CSNet_train_dag(net,  ...
    'learningRate',opts.learningRate, ...
    'derOutputs',opts.derOutputs, ...
    'numSubBatches',opts.numSubBatches, ...
    'backPropDepth',opts.backPropDepth, ...
    'solver',opts.solver, ...
    'batchSize', opts.batchSize, ...
    'modelname', opts.modelName, ...
    'gpus',opts.gpus) ;





