
flags.highRes = 1;

if breastData
    Matrix.Sub = 320;
    KM = 5; % number of tissue segments
    nline = 8; % 8 radial views, equavalent to time resolution of 4.2 Sec

    if flags.highRes
        nline = 2; % 2 radial views, equavalent to time resolution of 1.0 Sec
    end

end

if flags.highRes
    flags.GPU = 0;
else
    flags.GPU = 1;
end

lambLow = 0.06;  % lambda, low-res
lambHigh = 0.01; % lambda, high-res

numIter = 3;   %  regular temp' res (nline = 2) outside CS loop
nite = 7;     %  regular temp' res (nline = 2) inside CS loop

K = 6; % limit the number of PC's
