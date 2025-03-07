function  res = Emat_GROG2DSub(mask,b1,W1,PCA)

res.adjoint = 0;
res.mask = mask;
res.b1=repmat(permute(b1,[1,2,4,3]),1,1,size(mask,3),1);
res.W1=sqrt(W1);
res.PCA=PCA;
res = class(res,'Emat_GROG2DSub');

