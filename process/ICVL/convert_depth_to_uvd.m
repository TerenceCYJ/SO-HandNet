function [ uvd ] = convert_depth_to_uvd( depth )
[V, U] = ndgrid(1:size(depth, 1), 1:size(depth, 2));
uvd = cat(3, uint16(U), uint16(V), depth);
end

