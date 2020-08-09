function concatDagNet( net, net2, blockNum)
%% This function is to concat net2 to net
%   Parameter name is maintain from net2 to net


if isempty(blockNum)
    blockNum = length(net.layers); 
end
blockNum    = blockNum + 1;

% Concat the first layer
inVar       = net.vars(end).name; 
net         = addLayer(net, net2, 1, blockNum, inVar);

% Concat the rest layer 
for i = 2:1:length(net2.layers)
    net = addLayer(net, net2, i, blockNum, []); 
end

end


function net = addLayer(net, net2, i, blockNum, inVar)
layerCur = net2.layers(i).name;
block    = net2.layers(i).block;
outVar   = net2.layers(i).outputs;
if isempty(inVar)
    inVar    = net2.layers(i).inputs;
end

switch class(net2.layers(i).block)
    case 'dagnn.Conv'   % Convolution
        net.addLayer(layerCur, block, inVar, outVar,{[char(net2.layers(i).params(1))], [char(net2.layers(i).params(2))]});
        for j = 1:1:2
            f = net.getParamIndex([char(net2.layers(i).params(j))]) ;
            f2 = net2.getParamIndex([char(net2.layers(i).params(j))]) ;
            net.params(f).value        = net2.params(f2).value;
            net.params(f).learningRate = net2.params(f2).learningRate;
            net.params(f).weightDecay  = net2.params(f2).weightDecay ;
            net.params(f).trainMethod  = net2.params(f2).trainMethod;
        end
        %inVar    = outVar;
        blockNum = blockNum + 1;
        
    case 'dagnn.BatchNorm'   % Batch norm
        f2 = net2.getParamIndex([char(net2.layers(i).params(1))]) ;
        n_ch = length(net2.params(f2).value);
        params = {[char(net2.layers(i).params(1))], [char(net2.layers(i).params(2))], [char(net2.layers(i).params(3))]};
        net.addLayer(layerCur, dagnn.BatchNorm('numChannels', n_ch), inVar, outVar, params) ;
        for j = 1:1:3
            f = net.getParamIndex([char(net2.layers(i).params(j))]) ;
            f2 = net2.getParamIndex([char(net2.layers(i).params(j))]) ;
            net.params(f).value        = net2.params(f2).value;
            net.params(f).learningRate = net2.params(f2).learningRate;
            net.params(f).weightDecay  = net2.params(f2).weightDecay ;
            net.params(f).trainMethod  = net2.params(f2).trainMethod;
        end
        %inVar    = outVar;
        blockNum = blockNum + 1;
        
    otherwise
        net.addLayer(layerCur, block, inVar, outVar,{});
        %inVar    = outVar;
        blockNum = blockNum + 1;
end
end