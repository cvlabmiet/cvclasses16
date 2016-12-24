clear
clc
close all
num_images = 25 * 23;
num_images_without_object = 25 * 2;

videoName = 'Video.avi';
videoObjectName = 'Object.avi';
angle_step = 0.13;

im = imread('Bryansk.png');
im_object = imread('circle.png');

im_object(im_object ~= 255) = 127;

[M N ~] = size(im);
[M_Object N_Object ~] = size(im_object);

radius = 150;
angle = 0;

im_with_object = im;
im_segm_object = zeros(M, N, 3);

writer = VideoWriter(videoName, 'Uncompressed AVI');
writer.FrameRate = 25;
% writer.VideoCompressionMethod = 'Motion JPEG';
open(writer);
writerObject = VideoWriter(videoObjectName, 'Uncompressed AVI');
writerObject.FrameRate = 25;
% writerObject.VideoCompressionMethod = 'Motion JPEG';
open(writerObject);
for i = 1:num_images_without_object
    writeVideo(writer, uint8(im_with_object));
    writeVideo(writerObject, uint8(im_segm_object));
end

for i = (num_images_without_object + 1):(num_images + num_images_without_object)
    im_with_object = im;
    im_segm_object = zeros(M, N, 3);
    
    x = round(radius * cos(angle) + M / 2);
    y = round(radius * sin(angle) + N / 2);
    
    for j = 1:M_Object
        for k = 1:N_Object
            if ( sum(im_object(j, k, :) ~= 255) == 3 )
                im_with_object(x + j, y + k, :) = im_object(j, k, :);
                im_segm_object(x + j, y + k, :) = 255;
            end
        end
    end
    
%     imshow(uint8(im_with_object));
    writeVideo(writer, uint8(im_with_object)); 
    writeVideo(writerObject, uint8(im_segm_object));
    
%     radius = radius + radius_step;
    angle = angle + angle_step;
end

close(writer);
close(writerObject);