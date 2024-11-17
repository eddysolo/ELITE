clc
clear classes
clear
close all
cd '/home/eddysolomon/code/matlab/github_prep/'
addpath(genpath(pwd));

breastData = 1;
breastType = 'Malig';
aif_label = 'aif_avrg_RS'; % avrg AIF with original rise time

loadingCaseSlice

for jj = 1: length(caseInx)
    tt = tic;

    loadingGraspData

    for sl = sliceInx(jj)

        disp(['Proc. ',num2str(sl),'/',num2str(nsl)]);
        kdata = kdata_all(:,:,sl,:);

        [nx,ntviews,~,nc]=size(kdata);

        loadingGraspParams

        if breastData
            % reading tissue mask
            load ([pathName '/' 'BC' num2str(caseInx(jj),'%02.f') '_mask_sl' num2str(sl)])
        end

        if flags.highRes
            disp('reading low resolution recon by ResNet')
            file = ['/high-res/lowResPredict_data_BC' num2str(caseInx(jj),'%02.f') '_slice' num2str(sl) '_' aif_label '_rept50_randRiseTimeAIF_valid_test_new2_v7_2_learningRate5e-05_batch_size1_RESNET_epoch=200.mat'];

            afterCNN = load ([pathName file]) ;
            recon_basis = afterCNN.result;
            recon_basis = rot90(recon_basis,3);
            recon_basis(recon_basis==0) = 1e-5;

        end
        for kNum=1:length(K) % number of K
            for segNum=1:length(KM) % number of segments
                addpath(genpath(pwd))
                if flags.highRes
                    [GRASP_Pro_kArr,kArr,idxHighResImg] = run_GRASP_Pro_seg_kArr_roi_crop_highRes(kdata,Traj,nline,K(kNum),lambHigh,numIter,flags,KM(segNum),nite,breastType,mask,recon_basis,caseInx(jj));
                else
                    [GRASP_Pro_kArr,kArr,idxHighResImg,cost_f_val_GRASP_Pro_seg] = run_GRASP_Pro_seg_kArr_roi_crop(kdata,Traj,nline,K(kNum),Matrix,lambLow,lambHigh,numIter,flags,KM(segNum),nite,mask,breastType);
                end

                GRASP_Pro_kArr = GRASP_Pro_kArr / max(GRASP_Pro_kArr(:));
                %                 figure,imshow3Dfull(rot90(squeeze(GRASP_Pro_kArr)),[0 1])
            end
        end

    end

    s = toc(tt);
    h = fix(s/3600);
    s = mod(s,3600);
    m = fix(s/60);
    s = mod(s,60);

    fprintf (['\n done case N' num2str(caseInx(jj)) '/' ' after %dh:%dm:%2.0fs\n'], h,m,s);

end
