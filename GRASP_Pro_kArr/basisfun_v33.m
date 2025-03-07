function subImgBasis = basisfun_v33(data,Data_idxArr,basisArrCell,kArr,KM)
[nx, ny, nt] = size(data);

for seg_inx =1: KM  % cluster num
    basis = basisArrCell(:,:,seg_inx);
    K = kArr(seg_inx); % xxxxxxx
    basis_sel = basis(:,1:K);
    Kmax = max(max(kArr));
    Data_tmp = zeros(size(Data_idxArr));
    Data_tmp(Data_idxArr==seg_inx)=1;

    Data_tmp_frame =Data_tmp(:,:,1);
    [rIdx,cIdx] = find(0<(Data_tmp_frame));
    tmp = data.*Data_tmp;

    tmp_new = zeros([size(nonzeros(tmp(:,:,1)),1), nt]);
    for ip=1: size(tmp,3)
        tmp_new(:,ip) = nonzeros(tmp(:,:,ip));
    end


    %     clear proj_sub
    proj_sub = tmp_new*basis_sel;


    parfor k_inc = 1:K % K Kmax
        proj_sub_all = zeros([nx ny]);
        val = proj_sub(:,k_inc)';
        x = rIdx(:,1);
        y = cIdx(:,1);
        idx = sub2ind(size(proj_sub_all),x,y) ;
        proj_sub_all(idx) = val ;
        proj_sub_all_final(:,:,k_inc,seg_inx) = proj_sub_all;
    end
end
subImgBasis = zeros([nx,ny,Kmax]);

for ig =1:size(proj_sub_all_final,4)
    subImgBasis = subImgBasis + proj_sub_all_final(:,:,:,ig);
end
