function net = DoG_CSNet_Init_Phase1()
% function net = CSNet_Init()
global subRate blk_size channel epoch
% by Kai Zhang (1/2018)
% cskaizhang@gmail.com
% https://github.com/cszn

% Create DAGNN object
net = dagnn.DagNN();

% conv + relu
blockNum = 1;
inVar0    = 'input';
% channel  = 2; % grayscale image
dims     = [3,3,channel,64];
pad      = [1,1];
stride   = [1,1];
lr       = [1,1];


% 1st layer sampling with convolution
%% 1. Gaussian blurr filering 

if channel == 2, blurKernel  = [2, 0.001]; end
if channel == 3, blurKernel  = [4, 2, 0.001]; end
if channel == 4, blurKernel  = [8, 4, 2, 0.001]; end
dimAll         = [3, 3, 1, 1; 5, 5, 1, 1; 7 7 1 1; 9 9 1 1];
strideAll      = [1, 1; 1 1; 1 1; 1 1];
padAll    = [1, 1; 2 2; 3 3; 4 4 ]; 

% First layer 
inVarConcatMeas = cell(1);
inVarConcat     = cell(1); 

noMeas   = round(subRate * blk_size^2);


i = 1;      % Convolution 
[net, inVar, blockNum] = addConv(net, blockNum, 'input', dimAll(i, :),...         % Convolution 
                                         padAll(i, :), strideAll(i, :), [1, 0]);
inVarConcat{i}         = inVar;
[net, inVar, blockNum] = addSubTract(net, blockNum, {'input', inVar}); 
inVarConcatMeas{i}     = inVar;

i = 2;    % Residual of convolution 
[net, inVar, blockNum] = addConv(net, blockNum, 'input', dimAll(i, :),...         % Convolution 
                                         padAll(i, :), strideAll(i, :), [1, 0]);
inVarConcat{i}         = inVar;
[net, inVar, blockNum] = addSubTract(net, blockNum, {inVarConcat{2:-1:1}}); 
inVarConcatMeas{i}     = inVar;


i = 3;    % Residual of convolution 
[net, inVar, blockNum] = addConv(net, blockNum, 'input', dimAll(i, :),...         % Convolution 
                                         padAll(i, :), strideAll(i, :), [1, 0]);
inVarConcat{i}         = inVar;
[net, inVar, blockNum] = addSubTract(net, blockNum, {inVarConcat{3:-1:2}});     
inVarConcatMeas{i}     = inVar;

i = 4;   % Final residual layer 
[net, inVar, blockNum] = addConv(net, blockNum, 'input', dimAll(i, :),...         % Convolution 
                                         padAll(i, :), strideAll(i, :), [1, 0]);
% [net, inVar, blockNum] = addSum(net, blockNum, {inVarConcat{:}}); 
% [net, inVar, blockNum] = addSubTract(net, blockNum, {'input', inVar}); 
inVarConcatMeas{i}     = inVar;

% Concat all convolution layer of DoC 
[net, inVar, blockNum] = addConcat(net, blockNum, {inVarConcatMeas{:}});


%% Sampling accross 4 layers 

[net, inVar, blockNum] = addConv(net, blockNum, inVar, [blk_size, blk_size, 4, noMeas],  ...   % Sampling
                                    [0 0], [blk_size, blk_size], [1, 0]);
									
%% Initial reconstruction 
[net, inVar, blockNum] = addConv(net, blockNum, inVar, [1, 1, noMeas, blk_size^2], ...      % Init reconstruct 1x1 Conv.
                                    [0 0], stride, [1, 0]);
[net, inVar, blockNum] = addReshapeConcat(net, blockNum, inVar, blk_size);

%% Copy data from Phase 1
%% ---- Copy previous data and set learning rate to 0 ----------------- 
data = load(['data/model/W1_DoG_CSNet5_r' num2str(subRate) '_blk' num2str(blk_size) '_mBat32-epoch-' num2str(epoch) '.mat']);
net1 = dagnn.DagNN.loadobj(data.net) ;
net = copyParDagNet(net, net1, [], [], []); 

%------------------------ Enhance Reconstruction ------------------
dims = [3, 3, 1, 64];
[net, inVar, blockNum] = addConv(net, blockNum, inVar, dims, pad, stride, lr);
[net, inVar, blockNum] = addReLU(net, blockNum, inVar);

dims   = [3,3,64,64];
[net, inVar, blockNum] = addConv(net, blockNum, inVar, dims, pad, stride, lr);
[net, inVar, blockNum] = addReLU(net, blockNum, inVar);

[net, inVar, blockNum] = addConv(net, blockNum, inVar, dims, pad, stride, lr);
[net, inVar, blockNum] = addReLU(net, blockNum, inVar);

[net, inVar, blockNum] = addConv(net, blockNum, inVar, dims, pad, stride, lr);
[net, inVar, blockNum] = addReLU(net, blockNum, inVar);

dims   = [3,3,64,1];
[net, inVar, blockNum] = addConv(net, blockNum, inVar, dims, pad, stride, lr);
% -------------------- Completed reconstruction each Scale -----------
% %% ---- Copy previous data and set learning rate to 0 ----------------- 
% data = load(['data/model/W2_P_CSNet5_r' num2str(subRate) '_blk' num2str(blk_size) '_mBat32-epoch-' num2str(epoch) '.mat']);
% net1 = dagnn.DagNN.loadobj(data.net) ;
% net = copyParDagNet(net, net1, [], [], []); 
% 
% %% MWCNN
% sigma 	 = 15; 
% data = load(['../../MWCNN/models/MWCNN_Haart_GDSigma' num2str(sigma) '.mat']);
% net2 = dagnn.DagNN.loadobj(data.net) ;
% 
% % Concatinate two network
% concatDagNet( net, net2, []);

outputName = 'prediction';
net.renameVar(inVar,outputName)

% loss
net.addLayer('loss', dagnn.Loss('loss','L2'), {'prediction','label'}, {'objective'},{});
net.vars(net.getVarIndex('prediction')).precious = 1;

end


% Add a subtraction layer
function [net, inVar, blockNum] = addSubTract(net, blockNum, inVar)

outVar   = sprintf('subtract%d', blockNum);
layerCur = sprintf('subtract%d', blockNum);

block    = dagnn.Subtract();
net.addLayer(layerCur, block, inVar, {outVar},{});

inVar    = outVar;
blockNum = blockNum + 1;
end



function [net, inVar, blockNum] = addDWT2HD(net, blockNum, inVar, level)
outVar   = sprintf('LIEVE0%d_haartx', level);
layerCur = sprintf('Haart_LIEVE0%d', level);

block = dagnn.DWT2HD();
net.addLayer(layerCur, block, inVar, {outVar},{});

inVar = outVar;
blockNum = blockNum + 1;
end

function [net, inVar, blockNum] = addIWT2HD(net, blockNum, inVar, level)
outVar   = sprintf('Haart_LIEVE0%d', level);
layerCur = sprintf('IHaart_LIEVE0%d', level);

block = dagnn.IWT2HD();
net.addLayer(layerCur, block, inVar, {outVar},{});

inVar = outVar;
blockNum = blockNum + 1;
end


% Add a reshape and concat layers
function [net, inVar, blockNum] = addReshapeConcat(net, blockNum, inVar, blk_size)
% global blk_size

outVar   = sprintf('ReshapeConcat%d', blockNum);
layerCur = sprintf('ReshapeConcat%d', blockNum);

block    = dagnn.ReshapeConcat();
block.dim = [blk_size, blk_size];

net.addLayer(layerCur, block, inVar, {outVar},{});
inVar    = outVar;
blockNum = blockNum + 1;
end

% Add a Concat layer
function [net, inVar, blockNum] = addConcat(net, blockNum, inVar)
outVar   = sprintf('concat%d', blockNum);
layerCur = sprintf('concat%d', blockNum);

block = dagnn.Concat('dim',3);
net.addLayer(layerCur, block, inVar, {outVar},{});

inVar = outVar;
blockNum = blockNum + 1;
end


% Add a loss layer
function [net, inVar, blockNum] = addLoss(net, blockNum, inVar)

outVar   = 'objective';
layerCur = sprintf('loss%d', blockNum);

block    = dagnn.Loss('loss','L2');
net.addLayer(layerCur, block, inVar, {outVar},{});

inVar    = outVar;
blockNum = blockNum + 1;
end


% Add a sum layer
function [net, inVar, blockNum] = addSum(net, blockNum, inVar)

outVar   = sprintf('sum%d', blockNum);
layerCur = sprintf('sum%d', blockNum);

block    = dagnn.Sum();
net.addLayer(layerCur, block, inVar, {outVar},{});

inVar    = outVar;
blockNum = blockNum + 1;
end


% Add a relu layer
function [net, inVar, blockNum] = addReLU(net, blockNum, inVar)

outVar   = sprintf('relu%d', blockNum);
layerCur = sprintf('relu%d', blockNum);

block    = dagnn.ReLU('leak',0);
net.addLayer(layerCur, block, {inVar}, {outVar},{});

inVar    = outVar;
blockNum = blockNum + 1;
end

% Add a relu layer
function [net, inVar, blockNum] = addReLU_w_Name(net, blockNum, inVar, outVar, layerCur)

%outVar   = sprintf('relu%d', blockNum);
%layerCur = sprintf('relu%d', blockNum);

block    = dagnn.ReLU('leak',0);
net.addLayer(layerCur, block, {inVar}, {outVar},{});

inVar    = outVar;
blockNum = blockNum + 1;
end


% Add a bnorm layer
function [net, inVar, blockNum] = addBnorm(net, blockNum, inVar, n_ch)

trainMethod = 'adam';
outVar   = sprintf('bnorm%d', blockNum);
layerCur = sprintf('bnorm%d', blockNum);

params={[layerCur '_g'], [layerCur '_b'], [layerCur '_m']};
net.addLayer(layerCur, dagnn.BatchNorm('numChannels', n_ch), {inVar}, {outVar},params) ;

pidx = net.getParamIndex({[layerCur '_g'], [layerCur '_b'], [layerCur '_m']});
b_min                           = 0.025;
net.params(pidx(1)).value       = clipping(sqrt(2/(9*n_ch))*randn(n_ch,1,'single'),b_min);
net.params(pidx(1)).learningRate= 1;
net.params(pidx(1)).weightDecay = 0;
net.params(pidx(1)).trainMethod = trainMethod;

net.params(pidx(2)).value       = zeros(n_ch, 1, 'single');
net.params(pidx(2)).learningRate= 1;
net.params(pidx(2)).weightDecay = 0;
net.params(pidx(2)).trainMethod = trainMethod;

net.params(pidx(3)).value       = [zeros(n_ch,1,'single'), 0.01*ones(n_ch,1,'single')];
net.params(pidx(3)).learningRate= 1;
net.params(pidx(3)).weightDecay = 0;
net.params(pidx(3)).trainMethod = 'average';

inVar    = outVar;
blockNum = blockNum + 1;
end

function [net, inVar, blockNum] = addBnorm_w_Name(net, blockNum, inVar, n_ch, layerCur, outVar)

trainMethod = 'adam';
%outVar   = sprintf('bnorm%d', blockNum);
%layerCur = sprintf('bnorm%d', blockNum);

params={[layerCur '_g'], [layerCur '_b'], [layerCur '_m']};
net.addLayer(layerCur, dagnn.BatchNorm('numChannels', n_ch), {inVar}, {outVar},params) ;

pidx = net.getParamIndex({[layerCur '_g'], [layerCur '_b'], [layerCur '_m']});
b_min                           = 0.025;
net.params(pidx(1)).value       = clipping(sqrt(2/(9*n_ch))*randn(n_ch,1,'single'),b_min);
net.params(pidx(1)).learningRate= 1;
net.params(pidx(1)).weightDecay = 0;
net.params(pidx(1)).trainMethod = trainMethod;

net.params(pidx(2)).value       = zeros(n_ch, 1, 'single');
net.params(pidx(2)).learningRate= 1;
net.params(pidx(2)).weightDecay = 0;
net.params(pidx(2)).trainMethod = trainMethod;

net.params(pidx(3)).value       = [zeros(n_ch,1,'single'), 0.01*ones(n_ch,1,'single')];
net.params(pidx(3)).learningRate= 1;
net.params(pidx(3)).weightDecay = 0;
net.params(pidx(3)).trainMethod = 'average';

inVar    = outVar;
blockNum = blockNum + 1;
end


% add a ConvTranspose layer
function [net, inVar, blockNum] = addConvt(net, blockNum, inVar, dims, crop, upsample, lr)
opts.cudnnWorkspaceLimit = 1024*1024*1024*2; % 2GB
convOpts = {'CudnnWorkspaceLimit', opts.cudnnWorkspaceLimit} ;
trainMethod = 'adam';

outVar      = sprintf('convt%d', blockNum);

layerCur    = sprintf('convt%d', blockNum);

convBlock = dagnn.ConvTranspose('size', dims, 'crop', crop,'upsample', upsample, ...
    'hasBias', true, 'opts', convOpts);

net.addLayer(layerCur, convBlock, {inVar}, {outVar},{[layerCur '_f'], [layerCur '_b']});

f  = net.getParamIndex([layerCur '_f']) ;
sc = sqrt(2/(dims(1)*dims(2)*dims(4))) ; %improved Xavier
net.params(f).value        = sc*randn(dims, 'single');
net.params(f).learningRate = lr(1);
net.params(f).weightDecay  = 1;
net.params(f).trainMethod  = trainMethod;

f = net.getParamIndex([layerCur '_b']) ;
net.params(f).value        = zeros(dims(3), 1, 'single');
net.params(f).learningRate = lr(2);
net.params(f).weightDecay  = 1;
net.params(f).trainMethod  = trainMethod;

inVar    = outVar;
blockNum = blockNum + 1;

end


% add a Conv layer
function [net, inVar, blockNum] = addConv(net, blockNum, inVar, dims, pad, stride, lr)
opts.cudnnWorkspaceLimit = 1024*1024*1024*2; % 2GB
convOpts    = {'CudnnWorkspaceLimit', opts.cudnnWorkspaceLimit} ;
trainMethod = 'adam';

outVar      = sprintf('conv%d', blockNum);
layerCur    = sprintf('conv%d', blockNum);

convBlock   = dagnn.Conv('size', dims, 'pad', pad,'stride', stride, ...
    'hasBias', true, 'opts', convOpts);

net.addLayer(layerCur, convBlock, {inVar}, {outVar},{[layerCur '_f'], [layerCur '_b']});

f = net.getParamIndex([layerCur '_f']) ;
sc = sqrt(2/(dims(1)*dims(2)*max(dims(3), dims(4)))) ; %improved Xavier
net.params(f).value        = sc*randn(dims, 'single') ;
net.params(f).learningRate = lr(1);
net.params(f).weightDecay  = 1;
net.params(f).trainMethod  = trainMethod;

f = net.getParamIndex([layerCur '_b']) ;
net.params(f).value        = zeros(dims(4), 1, 'single');
net.params(f).learningRate = lr(2);
net.params(f).weightDecay  = 1;
net.params(f).trainMethod  = trainMethod;

inVar    = outVar;
blockNum = blockNum + 1;
end


function [net, inVar, blockNum] = addConv_w_Name(net, blockNum, inVar, dims, pad, stride, lr, layerCur, outVar)
opts.cudnnWorkspaceLimit = 1024*1024*1024*2; % 2GB
convOpts    = {'CudnnWorkspaceLimit', opts.cudnnWorkspaceLimit} ;
trainMethod = 'adam';

%outVar      = sprintf('conv%d', blockNum);
%layerCur    = outVar; %sprintf('conv%d', blockNum);

convBlock   = dagnn.Conv('size', dims, 'pad', pad,'stride', stride, ...
    'hasBias', true, 'opts', convOpts);

net.addLayer(layerCur, convBlock, {inVar}, {outVar},{[layerCur '_f'], [layerCur '_b']});

f = net.getParamIndex([layerCur '_f']) ;
sc = sqrt(2/(dims(1)*dims(2)*max(dims(3), dims(4)))) ; %improved Xavier
net.params(f).value        = sc*randn(dims, 'single') ;
net.params(f).learningRate = lr(1);
net.params(f).weightDecay  = 1;
net.params(f).trainMethod  = trainMethod;

f = net.getParamIndex([layerCur '_b']) ;
net.params(f).value        = zeros(dims(4), 1, 'single');
net.params(f).learningRate = lr(2);
net.params(f).weightDecay  = 1;
net.params(f).trainMethod  = trainMethod;

inVar    = outVar;
blockNum = blockNum + 1;
end

% add a Conv layer
function [net, inVar, blockNum] = addConvInit(net, blockNum, inVar, dims, pad, stride, lr, blurKernel)
opts.cudnnWorkspaceLimit = 1024*1024*1024*2; % 2GB
convOpts    = {'CudnnWorkspaceLimit', opts.cudnnWorkspaceLimit} ;
trainMethod = 'adam';

outVar      = sprintf('conv%d', blockNum);
layerCur    = sprintf('conv%d', blockNum);

convBlock   = dagnn.Conv('size', dims, 'pad', pad,'stride', stride, ...
    'hasBias', true, 'opts', convOpts);

net.addLayer(layerCur, convBlock, {inVar}, {outVar},{[layerCur '_f'], [layerCur '_b']});

f = net.getParamIndex([layerCur '_f']) ;
sc = sqrt(2/(dims(1)*dims(2)*max(dims(3), dims(4)))) ; %improved Xavier
H        				   = sc*randn(dims, 'single') ;
for i = 1:1:size(H, 4)
    H(:, :, 1,i)           = fspecial('Gaussian',dims(1:2), blurKernel); %imshow(H,[]);
end

net.params(f).value        = single(H);
net.params(f).learningRate = lr(1);
net.params(f).weightDecay  = 1;
net.params(f).trainMethod  = trainMethod;

f = net.getParamIndex([layerCur '_b']) ;
net.params(f).value        = zeros(dims(4), 1, 'single');
net.params(f).learningRate = lr(2);
net.params(f).weightDecay  = 1;
net.params(f).trainMethod  = trainMethod;

inVar    = outVar;
blockNum = blockNum + 1;
end


function A = clipping(A,b)
A(A>=0&A<b) = b;
A(A<0&A>-b) = -b;
end
