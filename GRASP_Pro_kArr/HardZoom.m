function OutPicture = KM(InPicture, ZoomFactor)
      % Si el factor de escala es 1, no se hace nada
      if ZoomFactor == 1
          OutPicture = InPicture;
          return;
      end
      % Se obtienen las dimensiones de las imágenes
      ySize = size(InPicture, 1);
      xSize = size(InPicture, 2);
      zSize = size(InPicture, 3);
      yCrop = floor(ySize / 2 * abs(ZoomFactor - 1));
      xCrop = floor(xSize / 2 * abs(ZoomFactor - 1));
      % Si el factor de escala es 0 se devuelve una imagen en negro
      if ZoomFactor == 0
          OutPicture = uint8(zeros(ySize, xSize, zSize));
          return;
      end
      % Se reescala y se reposiciona en en centro
      zoomPicture = imresize(InPicture, ZoomFactor);
      ySizeZ = size(zoomPicture, 1);
      xSizeZ = size(zoomPicture, 2);      
      if ZoomFactor > 1
          OutPicture = zoomPicture( 1+yCrop:yCrop+ySize, 1+xCrop:xCrop+xSize, :);
      else
          OutPicture = uint8(zeros(ySize, xSize, zSize));
          OutPicture( 1+yCrop:yCrop+ySizeZ, 1+xCrop:xCrop+xSizeZ, :) = zoomPicture;
      end
end