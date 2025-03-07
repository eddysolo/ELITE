clc
disp ('.............................')
disp (['processing N',num2str(caseInx(jj))])

if breastData
    pathName = ['ELITE/data/'];

    cd (pathName)
    
    fileList = dir(fullfile(pathName, '*.h5'));
    filename = {fileList(1).name};
    filename = char(filename);
%     h5disp(filename)
    tmp = h5read(filename,'/kspace');
    data = complex(tmp(:,:,:,:,1),tmp(:,:,:,:,2));
    data = squeeze(data);
    data = permute(data,[3,4,1,2]);
    orig_size = size(data);
    disp('kspace orig size');
    disp(orig_size);
    
    partitions = str2num(h5readatt(filename,'/kspace','partitions'));size(data,3);
    imagesPerSlab = str2num(h5readatt(filename,'/kspace','imagesPerSlab'));
    centerpar = str2num(h5readatt(filename,'/kspace','centerPartition'));

    data=zerofill_Kz(data,3,partitions,imagesPerSlab,centerpar,true);

    nPoints=size(data,1);
    nCoils=size(data,4);
    nSpokes=size(data,2);
    nPar=size(data, 3);
    nRep=size(data, 5);

    kdata_all = zeros(size(data));
    
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
