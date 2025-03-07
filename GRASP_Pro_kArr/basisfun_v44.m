function subImg = basisfun_v44(data,Data_idxArr,basisArrCell,kArr,KM)
[nx, ny, nt] = size(Data_idxArr);

for seg_inx =1: KM % cluster num
    basis = basisArrCell(:,:,seg_inx);
    Kmax = max(max(kArr));
    basis_sel = basis(:,1:Kmax);
    Data_tmp = zeros(size(Data_idxArr));
    Data_tmp(Data_idxArr==seg_inx)=1;

    Data_tmp_frame =Data_tmp(:,:,1);
    [rIdx,cIdx] = find(0<(Data_tmp_frame));
    Data_tmp = repmat(Data_tmp_frame,[1 1 Kmax]);
    tmp = data.*Data_tmp;

    tmp_new = zeros([size(nonzeros(tmp(:,:,1)),1), Kmax]);
    for ip=1: size(tmp,3)
        if ~isempty(nonzeros(tmp(:,:,ip)))
            tmp_new(:,ip) = nonzeros(tmp(:,:,ip));
        else
            tmp_new(:,ip) = zeros(size(tmp_new,1),1);
        end
    end


    %     clear proj_sub
    proj_sub = tmp_new*basis_sel';


    parfor t_inc = 1:nt % time points num
        proj_sub_all = zeros([nx ny]);
        val = proj_sub(:,t_inc)';
        x = rIdx(:,1);
        y = cIdx(:,1);
        idx = sub2ind(size(proj_sub_all),x,y) ;
        proj_sub_all(idx) = val ;
        proj_sub_all_final(:,:,t_inc,seg_inx) = proj_sub_all;
    end
end
subImg = zeros([nx,ny,size(proj_sub_all_final,3)]);

for ig =1:size(proj_sub_all_final,4)
    subImg = subImg + proj_sub_all_final(:,:,:,ig);
end
