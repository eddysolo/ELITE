function tmp_new = basisfun_v6(block_struct)


startInd = block_struct.location;
tmp = block_struct.data;
[rIdx,cIdx,V] = find(tmp==0);
tmp_new = tmp;

if sum(sum(V))~=0
    M = mode(tmp);
    tmp_new(rIdx,cIdx) = M(1);
end

    