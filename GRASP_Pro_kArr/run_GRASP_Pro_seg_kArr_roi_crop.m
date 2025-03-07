function [GRASP_Pro_recon,kArr,idxHighResImg,cost_f_val] = ...
    run_GRASP_Pro_seg_kArr_roi_crop(kdata,Traj,nline,K,Matrix,lambLow,lambHigh, numIter,flags,KM,nite,mask_tissue,breastType)
% kdata: kspace-data
% Traj: Trajectory for kspace data
% nline: # of spokes per frame
% K: # of basis function to include.
% (You might want to take a look at each basis function and corresponding eig. values before choosing this)
% Matrix: You need to specify 'dimension' for low-res. GRASP image (for PCA)
% Generally, I used 1/4 size of original matrix. In H/N data, I think the image matrix is 256x256
% So you would want to have the low-res GRASP image to be 64x64. Then specify Matrix.Sub = 128
% lambda: Lambda value for Full-res GRASP recon
% numIter: # of Iteration for Full-res GRASP recon
% flags.GPU: Running on GPU

kdata_ori = kdata;
Traj_ori = Traj;

[nx,ntviews,nz,nc]=size(kdata);

kdata_base=kdata((nx-Matrix.Sub)/2+1:end-(nx-Matrix.Sub)/2,:,:,:);
Traj_base = Trajectory_GoldenAngle_GROG(ntviews,Matrix.Sub);

[nx,ntviews,nz,nc]=size(kdata_base);
bas_base=nx/2;

[Gx,Gy] = GROG.get_Gx_Gy(kdata_base,Traj_base);

%Coil sensitivities estimation
G = GROG.init(kdata_base,Traj_base,Gx,Gy,0);
kref = GROG.interp(kdata_base,G,1);
ref=squeeze(ifft2c_mri(kref));
b1=adapt_array_2d(ref);clear ref
b1=single(b1/max(abs(b1(:))));

%Calculate the DCF for nyquist sampling
Nqu=floor(bas_base/2*pi);
G = GROG.init(kdata_base(:,end-Nqu+1:end,:,:),Traj_base(:,end-Nqu+1:end),Gx,Gy,0);
DCF=reshape(G.weight,[sqrt(size(G.weight,1)),sqrt(size(G.weight,1))]);

DCF=CropImg(DCF,nx,nx);

%sort the data
[nx,ntviews,nt,nc] = size(kdata_base);

nt=floor(ntviews/nline); %number of dynamic frames
Traj_base=reshape(Traj_base(:,1:nt*nline),[nx,nline,nt]);
kdata_base=reshape(kdata_base(:,1:nt*nline,:,:),[nx,nline,nt,nc]);
[nx,ntviews,nt,nc] = size(kdata_base);

%calculat weighting for iteration
G = GROG.init(kdata_base,Traj_base,Gx,Gy,0);
DCF_U=reshape(G.weight,[sqrt(size(G.weight,1)),sqrt(size(G.weight,1)),nt]);
DCF_U=CropImg(DCF_U,nx,nx);
DCF=repmat(DCF,[1,1,nt,nc]);
DCF_U=repmat(DCF_U,[1,1,1,nc]);
Weighting=DCF./DCF_U;

%grog
kdata_base = GROG.interp(kdata_base,G,1);
mask=single(kdata_base~=0);

warning('off','all')
if flags.GPU==1
    Weighting = gpuArray(Weighting);
    mask=gpuArray(mask);
    b1=gpuArray(b1);
    kdata_base = gpuArray(kdata_base);
end

param.y=kdata_base.*sqrt(Weighting);
param.E=Emat_GROG2D(mask,b1,Weighting);

recon_cs=param.E'*param.y;
data=abs(single(gather(recon_cs/max(recon_cs(:)))));

param.TV=TV_Temp;
Weight1=lambLow;

param.TVWeight=max(abs(recon_cs(:)))*Weight1;
param.nite = 7;param.display = 1;
for n=1:3
    recon_cs = CSL1NlCg(recon_cs,param);
end

recon_basis=abs(gather(single(recon_cs/max(recon_cs(:)))));
disp('low resolution recon done')


%Base Calculation
tmp=CropImg(recon_basis,bas_base,bas_base);
[nx,ny,nt]=size(tmp);
% figure,imshow3Dfull(rot90(squeeze(tmp)),[0 1])

tissue_array_zoom = zeros([2*nx 2*ny]);
tissue_array_crop = zeros([nx ny]);

for tiss_inc = 1 : 5
    if tiss_inc == 1
        if strcmp(breastType,'Malig')
            tmp_zoom = HardZoom(mask_tissue.malignant,0.5);
        elseif strcmp(breastType,'Benign')
            tmp_zoom = HardZoom(mask_tissue.benign,0.5);
        end
    elseif tiss_inc == 2
        tmp_zoom = HardZoom(mask_tissue.glandular,0.5);
        tmp_zoom(162,199)=1;
        tmp_zoom(158,204)=1;
        tmp_zoom(166,200)=1;
    elseif   tiss_inc == 3
        tmp_zoom = HardZoom(mask_tissue.muscle,0.5);
    elseif  tiss_inc == 4
        tmp_zoom = HardZoom(mask_tissue.skin,0.5);
    elseif  tiss_inc == 5
        tmp_zoom = HardZoom(mask_tissue.heart,0.5);
    end

    tmp_crop = CropImg(tmp_zoom,bas_base,bas_base);
    tmp_zoom = double(tmp_zoom);
    tmp_crop = double(tmp_crop);
    if ~isempty(tmp_zoom) && ~isempty(tmp_crop)
        clear C_zoom C
        C = bitand(imbinarize(tissue_array_crop),imbinarize(tmp_crop));
        C_zoom = bitand(imbinarize(tissue_array_zoom),imbinarize(tmp_zoom));

        if isempty(nonzeros(C_zoom)) && isempty(nonzeros(C))
            tissue_array_crop = tissue_array_crop+(tmp_crop.*tiss_inc);
            tissue_array_zoom = tissue_array_zoom+(tmp_zoom.*tiss_inc);
        else
            tmp_crop(C>0) = 0;
            tmp_zoom(C_zoom>0) = 0;
            tissue_array_crop      = tissue_array_crop+(tmp_crop.*tiss_inc);
            tissue_array_zoom = tissue_array_zoom+(tmp_zoom.*tiss_inc);
        end
    else
        disp(['missing tissue num ',num2str(tiss_inc), '!!!']);
    end
end
if max(tissue_array_zoom(:)) ~= 5 || max(tissue_array_crop(:)) ~= 5
    disp(['missing tissue num ',num2str(tiss_inc), '!!!']);
    pause
end

idxLowResImg_zoom = flipud(fliplr(rot90(tissue_array_zoom)));
idxLowResImg_crop = flipud(fliplr(rot90(tissue_array_crop)));


idxArr_crop = repmat(idxLowResImg_crop,[1 1 nt]);
idxArr_zoom = repmat(idxLowResImg_zoom,[1 1 nt]);


cIndex = round(linspace(1,256,KM+1));
kkInx = 1;

% figure(11)
% c = parula;
% cMap = c(cIndex,:);

for kk=0: KM    
    Data_Seq_msk = zeros(size(tmp));
    Data_Seq_msk(idxArr_crop==kk)=1;
    Data_Seq = tmp.*Data_Seq_msk;
    clear Data_Seq_new
    for ip=1: size(Data_Seq,3)
        Data_Seq_new(:,ip) = nonzeros(Data_Seq(:,:,ip));
    end

    % Orig Base Selection
    covariance=cov(Data_Seq_new);
    [PC, V] = eig(covariance);
    V = diag(V);
    [junk, rindices] = sort(-1*V);
    V = V(rindices);
    V_perc= V/sum(V);

%     plot(V_perc,'-o', 'LineWidth', 2,'Color',cMap(kkInx,:));
%     axis([0 size(V_perc,1) 0 0.07]);
%     set(gca,'FontSize',12);
%     hold on;

    kArr(kk+1) = K;
    basis(:,:,kk+1) = PC(:,rindices);
    kkInx = kkInx +1;
end

kArr(1) = 1; % background signal
% title('singular values decay patterns per segment')
% legend('background','malignant', 'glandular', 'muscle', 'skin' ,'heart');

%%

[nx,ntviews,nz,nc]=size(kdata);
bas=nx/2;

[Gx,Gy] = GROG.get_Gx_Gy(kdata,Traj);

%Coil sensitivities estimation
G = GROG.init(kdata,Traj,Gx,Gy,0);
kref = GROG.interp(kdata,G,1);
ref=squeeze(ifft2c_mri(kref));
b1=adapt_array_2d(ref);clear ref
b1=single(b1/max(abs(b1(:))));

%Calculate the DCF for nyquist sampling
Nqu=floor(bas/2*pi);

G = GROG.init(kdata,Traj,Gx,Gy,0);


DCF=reshape(G.weight,[sqrt(size(G.weight,1)),sqrt(size(G.weight,1))]);
DCF=CropImg(DCF,nx,nx);

%sort the data
nt=floor(ntviews/nline);
Traj=reshape(Traj(:,1:nt*nline),[nx,nline,nt]);
kdata=reshape(kdata(:,1:nt*nline,:,:),[nx,nline,nt,nc]);
[nx,ntviews,nt,nc] = size(kdata);

%calculat weighting for iteration
G = GROG.init(kdata,Traj,Gx,Gy,0);
DCF_U=reshape(G.weight,[sqrt(size(G.weight,1)),sqrt(size(G.weight,1)),nt]);
DCF_U=CropImg(DCF_U,nx,nx);
DCF=repmat(DCF,[1,1,nt,nc]);
DCF_U=repmat(DCF_U,[1,1,1,nc]);
Weighting=(DCF./DCF_U);

%grog
kdata = GROG.interp(kdata,G,1);
mask=single(kdata~=0);

% warning('off','all')
if flags.GPU==1
    Weighting = gpuArray(Weighting);
    mask=gpuArray(mask);
    b1=gpuArray(b1);
    kdata = gpuArray(kdata);
end
% Orig Base Selection
clear DCF DCF_U kdata_base kdata_base

param.y=kdata.*sqrt(Weighting);
idxHighResImg = zeros([nx nx]);
for kinx = 0: KM
    Data_tmp = zeros([nx nx]);
    Data_tmp= imresize(squeeze(idxArr_zoom(:,:,1)==kinx),[nx nx]);
    Data_tmp = logical(Data_tmp);
    Data_tmp = Data_tmp*(kinx+1);
    idxHighResImg = idxHighResImg + Data_tmp;
    idxHighResImg(idxHighResImg>(kinx+1)) = (kinx+1);
end
% [rIdx,cIdx] = find(~idxHighResImg);
myfun6 = @basisfun_v6;
idxHighResImg = blockproc(idxHighResImg, [3, 3], myfun6, 'BorderSize', [1, 1]);

Data_idxArr = repmat(idxHighResImg,[1 1 nt]);
KM = KM + 1;
PCA=TempPCASub(Data_idxArr,kArr,basis,KM);
param.PCA=PCA;


clear kdata
param.E=Emat_GROG2DSub(mask,b1,Weighting,PCA);
recon_cs=param.E'*param.y;
tmp=abs(param.PCA'*(gather((single(gather(recon_cs/max(recon_cs(:))))))));
data2(:,:,:,1)=CropImg(tmp/max(tmp(:)),bas,bas);
param.TV=TV_Temp;

Weight1=lambHigh;
param.TVWeight=max(abs(recon_cs(:)))*Weight1;
param.nite = nite;param.display = 1;
clear Weighting mask

tt = tic;
%{
n = 1;
bStop=0;
error = 1;
while ~bStop
    [recon_cs, recon_fval] = CSL1NlCgSub(recon_cs,param);
    cost_f_val(:,n) = recon_fval;

    if n>=2
        error = (cost_f_val(end,n)-cost_f_val(end,n-1))/cost_f_val(end,n-1);
    end

    if abs(error)<2.5/100
        bStop = 1;
    end
    n = n+1;
end
%}

for n=1:numIter
    [recon_cs, cost_f_val] = CSL1NlCgSub(recon_cs,param);
end


disp('high resolution reco done')

s = toc(tt);
h = fix(s/3600);
s = mod(s,3600);
m = fix(s/60);
s = mod(s,60);

fprintf ('\n CS done after %dh:%dm:%2.0fs\n', h,m,s);


tmp=abs(param.PCA'*(gather((single(gather(recon_cs/max(recon_cs(:))))))));
data2(:,:,:,2)=CropImg(tmp/max(tmp(:)),bas,bas);

GRASP_Pro_p0 = data2(:,:,:,1);
GRASP_Pro_recon = squeeze(data2(:,:,:,2));