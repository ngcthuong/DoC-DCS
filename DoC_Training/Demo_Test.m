warning('off','all')


%run '.\matconvnet-1.0-beta23\matlab\vl_setupnn.m'
addpath('D:\matconvnet-1.0-beta25\matlab\mex');

addpath('.\utilities');

testSetName = {'Set5' 'Set14' 'Classic13_512'};

for kkk = 1:1:3
    
folderTest  = testSetName{kkk}

showResult  = 0;
writeRecon  = 1;
blkSize     = 16;
noLayer     = 5;
featureSize = 64;
batSize     = 32;
isLearnMtx = [1, 0];
channel     = 4; 

for epoch       = 1:200
    
    for samplingRate = [0.3]
        modelName   = ['W3_DoG_CSNet' num2str(noLayer) '_r' num2str(samplingRate) '_blk' num2str(blkSize) '_mBat' num2str(batSize)]; %%% model name
        
        %load(['.\model\Org\sampling' num2str(samplingRate) '.mat']);
        % net = vl_simplenn_move(net, 'gpu') ;
        
        try
        org = 0;
        if org == 1
            load(['..\..\model\Org\sampling' num2str(samplingRate) '.mat']);
            net      = dagnn.DagNN.fromSimpleNN(net);
            net.renameVar('x0', 'input');
            net.renameVar('x12', 'prediction');
        else
            data = load(fullfile('data\model',[modelName,'-epoch-',num2str(epoch),'.mat']));
            net  = dagnn.DagNN.loadobj(data.net);
           % net.removeLayer('objective') ;
        end
        
        catch
            continue
        end
        net.mode = 'test';
        net.move('gpu');
        
        
        %%% read images
        ext         =  {'*.jpg','*.png','*.bmp', '*.pgm', '*.tif'};
        filePaths   =  [];
        for i = 1 : length(ext)
            filePaths = cat(1,filePaths, dir(fullfile('testsets',folderTest,ext{i})) );
        end
        
        PSNRs_CSNet = zeros(1,length(filePaths));
        SSIMs_CSNet = zeros(1,length(filePaths));
        
        count = 1;
        allName = cell(1);
        
        for i = 1:length(filePaths)
            
            %%% read images
            image = imread(fullfile('testsets', folderTest, filePaths(i).name));
            [~,nameCur,extCur] = fileparts(filePaths(i).name);
            allName{count} = nameCur;
            if size(image,3) == 3
                image = modcrop(image,32);
                image = rgb2ycbcr(image);
                image = image(:,:,1);
            end
            label = im2single(image);
            if mod(size(label, 1), blkSize) ~= 0 || mod(size(label, 2), blkSize) ~= 0
                continue
            end
            
            input = label;
            input = gpuArray(input);
            %res    = vl_simplenn(net,input,[],[],'conserveMemory',true,'mode','test');
            net.eval({'input', input}) ;
            out1 = net.getVarIndex('prediction') ;
            output = gather(squeeze(gather(net.vars(out1).value)));
            
            %output = res(end).x;
            output = gather(output);
            input  = gather(input);
            %%% calculate PSNR and SSIM
            [PSNRCur_CSNet, SSIMCur_CSNet] = Cal_PSNRSSIM(im2uint8(label),im2uint8(output),0,0);
            if showResult
                %imshow(cat(2,im2uint8(label),im2uint8(output)));
                %title([filePaths(i).name,'    ',num2str(PSNRCur_CSNet,'%2.2f'),'dB','    ',num2str(SSIMCur_CSNet,'%2.4f')])
                %drawnow;
                display(['        ' filePaths(i).name,'        ',num2str(PSNRCur_CSNet,'%2.2f'),'dB','    ',num2str(SSIMCur_CSNet,'%2.3f')])
            end
            
            PSNRs_CSNet(i) = PSNRCur_CSNet;
            SSIMs_CSNet(i) = SSIMCur_CSNet;
            
            % save results for current image
            if writeRecon
                folder  = ['Results\2Image_CSNet' num2str(noLayer) '_' num2str(featureSize) ...
                    '_blk' num2str(blkSize) '_mBat' num2str(batSize) ...
                    '_' num2str(isLearnMtx(1)) '_' num2str(isLearnMtx(2)) '_epoch' num2str(100)];
                if ~exist(folder), mkdir(folder); end
                fileName = [folder '\' folderTest '_' allName{count} '_subrate' num2str(samplingRate) '.png'];
                imwrite(im2uint8(output), fileName );
                
                count = count + 1;
            end
            
        end
        % save results for current image
        folder  = ['Results\1Text_CSNet' num2str(noLayer) '_' num2str(featureSize) ...
            '_blk' num2str(blkSize) '_mBat' num2str(batSize) ...
            '_' num2str(isLearnMtx(1)) '_' num2str(isLearnMtx(2)) '_epoch' num2str(100)];
        if ~exist(folder), mkdir(folder); end
        imgName = [folderTest ];
        fileName = [folder '\' imgName '_subrate' num2str(samplingRate) '.txt'];
        write_txt(fileName, allName, samplingRate, PSNRs_CSNet, SSIMs_CSNet );
        
        disp(['Epoch ' num2str(epoch) ', subrate ' num2str(samplingRate) ': ' num2str(mean(PSNRs_CSNet), '%2.2f') 'dB, SSIM: ', num2str(mean(SSIMs_CSNet), '%2.3f')]);
    end
end
end; 