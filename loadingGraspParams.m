
flags.highRes = 1;

if breastData
    Matrix.Sub = 320;
    Patch = 10;
    KM = 5;

    nline = 8;

    if flags.highRes
        nline = 2;
    end

end

flags.GPU = 1;
flags.GRASP = 0;
flags.Nufft = 0;
flags.GRASP_Pro = 1;

lamb = 0.01;    % regular GRASP
lambLow = 0.06;  % GRASP-Pro, GRASP-seg, GRASP-patch, low-res
lambHigh = 0.01; % GRASP-Pro, GRASP-seg, GRASP-patch, high-res

numIter = 3;   %  regular temp' res (nline = 2) outside CS loop
nite = 7;     %  regular temp' res (nline = 2) inside CS loop

K = 6; % limit the number of PC's

energySeg = 0.98; % not relevant in the code
energyPatch = 0.98; % not relevant in the code (see /home/eddysolomon/code/matlab/T1B1/GRASP_Pro_Patch/basisfun_v22.m)