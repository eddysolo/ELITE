function res = mtimes(a,b)

if a.adjoint
    %res = zeros(size(b));
    res = adjDx(b(:,:,:,1))+adjDy(b(:,:,:,2));
    
    
    %{
    for i = 1:size(b,3)
        g = double(b(:,:,i));
        %kernel = [-1 1 0];
        %diffImageLeft = imfilter(g, kernel);
        kernel = [0 1 -1];
        diffImageRight = imfilter(g, kernel);
        %kernel = [-1 1 0]';
        %diffImageTop = imfilter(g, kernel);
        kernel = [0 1 -1]';
        diffImageBottom = imfilter(g, kernel);
        %TV_Image = sqrt(diffImageLeft.^2 + diffImageRight.^2 + diffImageTop.^2 + diffImageBottom.^2);
        res(:,:,i) = sqrt(diffImageRight.^2 + diffImageBottom.^2);
    end
    %}
else
    
    Dx = b([2:end,end],:,:)-b;
    Dy = b(:,[2:end,end],:)-b;
    res = cat(4,Dx,Dy);
    
    %{
    res = [];
    for i = 1:size(b,3)
        g = double(b(:,:,i));
        kernel = [-1 1 0];
        diffImageLeft = imfilter(g, kernel);
        %kernel = [0 1 -1];
        %diffImageRight = imfilter(g, kernel);
        kernel = [-1 1 0]';
        diffImageTop = imfilter(g, kernel);
        %kernel = [0 1 -1]';
        %diffImageBottom = imfilter(g, kernel);
        %TV_Image = sqrt(diffImageLeft.^2 + diffImageRight.^2 + diffImageTop.^2 + diffImageBottom.^2);
        res(:,:,i) = sqrt(diffImageLeft.^2 + diffImageTop.^2);
    end
    %}
end

function out = adjDy(x)
out = x(:,[1,1:end-1],:) - x;
out (:,1,:) = -x(:,1,:);
out (:,end,:) = x(:,end-1,:);

function out = adjDx(x)
out = x([1,1:end-1],:,:) - x;
out (1,:,:) = -x(1,:,:);
out (end,:,:) = x(end-1,:,:);