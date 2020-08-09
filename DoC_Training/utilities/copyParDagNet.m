function to_net = copyParDagNet( to_net, from_net, learningRate, weightDecay, trainMethod)
% This function will copy parameter of net2 to net1,

noParam = length(from_net.params); 
for f = 1:1:noParam
    to_net.params(f).value        = from_net.params(f).value;
    if isempty(learningRate)
        to_net.params(f).learningRate = from_net.params(f).learningRate;
    else 
        to_net.params(f).learningRate = learningRate;
    end
    if isempty(weightDecay)
        to_net.params(f).weightDecay  = from_net.params(f).weightDecay ;
    else
        to_net.params(f).weightDecay = weightDecay;
    end
    if isempty(trainMethod)
        to_net.params(f).trainMethod  = from_net.params(f).trainMethod;
    else
        to_net.params(f).trainMethod  = trainMethod;
    end
end