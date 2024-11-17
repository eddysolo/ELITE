clc
disp ('.............................')
disp (['processing N',num2str(caseInx(jj))])

if breastData
    pathName = ['/home/eddysolomon/code/matlab/github_prep/data/'];

    fName = dir([pathName,'/','*.dat']);
    disp(pathName)

    fileName = fName.name;

    twixfile = mapVBVD([pathName,'/',fileName]);

    try
        centerpar = max(twixfile{2}.image.centerPar);
        partitions   =twixfile{2}.image.NPar;
        imagesPerSlab=twixfile{2}.hdr.Meas.lImagesPerSlab;
        data=squeeze(twixfile{2}.image{''});
    catch
        centerpar = max(twixfile.image.centerPar);% esolo no Brace indexing
        partitions   =twixfile.image.NPar;
        imagesPerSlab=twixfile.hdr.Meas.lImagesPerSlab;
        data=squeeze(twixfile.image{''}); % esolo no Brace ind
    end


    disp('Loaded image')

    if breastData
        data=data(:,:,:,:,2);
    end

    data=permute(data,[1,3,4,2,5]);

    orig_size = size(data);
    disp('kspace orig size');
    disp(orig_size);

    data=zerofill_Kz(data,3,partitions,imagesPerSlab,centerpar,true);


    nPoints=size(data,1);
    nCoils=size(data,4);
    nSpokes=size(data,2);
    nPar=size(data, 3);
    nRep=size(data, 5);

    %%
    kdata_all = zeros(size(data));

    %%
    fprintf('Performing iFFT in z-direction \n')
    dim = 3;
    kdata_all = fftshift(ifft(fftshift(data,dim),[],dim),dim);


    kdata_all = flip(kdata_all,3);
    disp('After concat, kdata size');
    disp(size(kdata_all));

    [nx,ntviews,nsl,nc]=size(kdata_all);

    if nRep>1
        disp(['Traj: ',num2str(ntviews),'multiplying ',num2str(ntviews/nRep),'times']);
        Traj = Trajectory_GoldenAngle_GROG(round(ntviews/nRep),nx);
        Traj = repmat(Traj,[1,nRep]);
    else
        Traj = Trajectory_GoldenAngle_GROG(ntviews,nx);
    end
end
