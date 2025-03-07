function res= TempPCASub(Data_idxArr,kArr,Phi,KM)
% 
% implements a temporal FFT along the dim dimenison
%

res.adjoint = 0;
res.Phi=Phi;
res.Data_idxArr = Data_idxArr; %esolo
res.KM = KM; % number of segments
res.kArr = kArr; %esolo
res = class(res,'TempPCASub');

