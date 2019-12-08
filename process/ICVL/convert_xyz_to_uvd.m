function [ uvd ] = convert_xyz_to_uvd( xyz )
halfResX = 320/2;
halfResY = 240/2;
coeffX = 241.42;
coeffY = 241.42;

uvd = zeros(size(xyz));
uvd(:,:,1) = coeffX * xyz(:,:,1) ./ xyz(:,:,3) + halfResX;
uvd(:,:,2) = halfResY - coeffY * xyz(:,:,2) ./ xyz(:,:,3);
uvd(:,:,3) = xyz(:,:,3);

end

