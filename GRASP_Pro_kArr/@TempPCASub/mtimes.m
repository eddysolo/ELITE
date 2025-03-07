function res = mtimes(a,b)

if isa(a,'TempPCASub') == 0
    error('In  A*B only A can be TempPCASub operator');
end
if a.adjoint
    [nx,ny,nt]=size(b); % b is C [nx nx K]
    res = basisfun_v44(b,a.Data_idxArr,a.Phi,a.kArr,a.KM);% CB - projected to image space


else
    [nx,ny,nt]=size(b); % b is image [nx nx nt]
    res = basisfun_v33(b,a.Data_idxArr,a.Phi,a.kArr,a.KM); % C - image projected to sub_space


end

