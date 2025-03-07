clc;clear;

CaseID.ID='sub7';

%twixfile = mapVBVD('GRASP_BBB#SPrismaCBI#F20599#M1263#D110319#T160631#AXIALGRASP_YGB.dat');
filePath = ['/gpfs/scratch/baej05/GRASP_Pro_Brain/Data/',CaseID.ID,'/'];
fileName = dir([filePath,'*.dat']);
twixfile = mapVBVD([filePath,fileName.name]);
disp('Opend');
centerpar = max(twixfile.image.centerPar);
partitions   =twixfile.image.NPar;
imagesPerSlab=twixfile.hdr.Meas.lImagesPerSlab;

imagesPerSlab=52;

data=squeeze(twixfile.image{''});
%data=data(:,:,:,:,2);
disp(size(data));

data=permute(data,[1,3,4,2,5]);
%data=zerofill_Kz(data,3,partitions,imagesPerSlab,centerpar,true);
kdata = data(:,1:100,:,:,3);
clear data
[nPoints,nSpokes,nPar,nCoil] = size(kdata);

data2 = zerofill_Kz(squeeze(kdata),3,partitions,imagesPerSlab,centerpar,true);

for m=1:1:nCoil
    disp(m);
    for k=1:1:nSpokes
        %disp(k)
        %temp=squeeze(data2(n,m,k,:));
        temp=squeeze(data2(:,k,:,m));
        [~,idx] = max(temp(:));

        temp = temp*exp(-angle(temp(idx))*j);

        for n=1:1:nPoints 
            temp(n,:) = fftshift(ifft(fftshift(temp(n,:))));
        end

        %temp = temp*exp(-angle(temp(idx))*j);
        temp_RawFID(:,k,:,m)=temp;
        %temp_RawFID(n,m,k,:,1)=temp;
        clear temp
    end
end

save([filePath,'temp_RawFID.mat'],'temp_RawFID');

for ii = 1:imagesPerSlab
    %cd('/Users/baej05/Desktop/Research/GRASP_Pro_Brain/Code');
%clc;clearvars -except img ii;
%dataPath = '/home/baej05/GRASPPro_Brain/Data/sub7/sub3_sl';
%dataPath = ['/home/baej05/GRASPPro_Brain/Data/',CaseID,'/',CaseID,'_Rep03_Par'];

disp(ii);
%load([dataPath,num2str(ii,'%.3d'),'.mat']);
kdata = temp_RawFID(:,:,ii,:);
%cd('/home/baej05/GRASPPro_Brain/Code')

[nx,orig_ntviews,~,nCoils] = size(kdata);
%orig_ntviews = size(kdata,2);

%Traj = Trajectory_GoldenAngle_GROG(orig_ntviews/3,size(kdata,1));
%Traj = repmat(Traj,[1,3]);
Traj = Trajectory_GoldenAngle_GROG(3050,size(kdata,1));

%kdata = reshape(kdata,[nx,orig_ntviews,1,nCoils]);
%kdata = kdata(:,end-99:end,:,:);
Traj = Traj(:,1:100);

disp('kdata-size:');
disp(size(kdata));
disp('Traj-size:');
disp(size(Traj));


nline = size(kdata,2);
bas=size(kdata,1)/2;

[Gx,Gy] = GROG.get_Gx_Gy(kdata,Traj);

G = GROG.init(kdata,Traj,Gx,Gy,0);
kref = GROG.interp(kdata,G,1);
ref=squeeze(ifft2c_mri(kref));
b1=adapt_array_2d(ref);clear ref
b1=single(b1/max(abs(b1(:))));
%b1 = gpuArray(b1);

%Calculate the DCF for nyquist sampling
%Nqu=floor(bas/2*pi);
G = GROG.init(kdata,Traj,Gx,Gy,0);
DCF=reshape(G.weight,[sqrt(size(G.weight,1)),sqrt(size(G.weight,1))]);
[nx,ntviews,nt,nc] = size(kdata);
DCF=CropImg(DCF,nx,nx);

%sort the data
ntviews = size(kdata,2);
nt=floor(ntviews/nline); %number of dynamic frames
Traj=reshape(Traj(:,1:nt*nline),[nx,nline,nt]);
kdata=reshape(kdata(:,1:nt*nline,:,:),[nx,nline,nt,nc]);
[nx,ntviews,nt,nc] = size(kdata);

%calculat weighting for iteration
G = GROG.init(kdata,Traj,Gx,Gy,0);
DCF_U=reshape(G.weight,[sqrt(size(G.weight,1)),sqrt(size(G.weight,1)),nt]);
DCF_U=CropImg(DCF_U,nx,nx);
DCF=repmat(DCF,[1,1,nt,nc]);
DCF_U=repmat(DCF_U,[1,1,1,nc]);
%Weighting=gpuArray(single(DCF./DCF_U));
Weighting=DCF./DCF_U;

%grog
kdata = GROG.interp(kdata,G,1);
mask=single(kdata~=0);

param.E=Emat_GROG2D(mask,b1,single(Weighting));
clear mask b1 DCF DCF_U kref 

param.y=single(kdata).*sqrt(single(Weighting));
%param.y=gpuArray(kdata_sel.*sqrt(Weighting));
%clear kdata
%clear mask Weighting b1

recon_cs=param.E'*param.y;
data=abs(single(gather(recon_cs/max(recon_cs(:)))));
data=CropImg(data,bas,bas);
img(:,:,ii) = data;


end
save([filePath,'slice_check.mat'],'img');
