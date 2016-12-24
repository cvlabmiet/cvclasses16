clear
clc
close all

flageWritingImages = 0;

num_images = 480;
num_images_without_object = 24*3;

videoName = './1G/ImageWithObject.avi';

videoObjectName = './1G/object.avi';

radius_step = 0.4;
angle_step = 0.05;

im = imread('./1G/Rostislav.png');
im_object = imread('./1G/romb.png');
%im_object = rgb2gray(im_object);

path_im_object = './1G/image with object/';
path_object = './1G/object/';

figure()
imshow(im);
title('Исходное изображение')

figure()
imshow(im_object);
title('Изображение объекта')

[M N ~] = size(im);
[M_Object N_Object ~] = size(im_object);

radius = 0;
angle = 0;

im_with_object = im;
im_segm_object = zeros(M, N, 3);

writer = avifile(videoName, 'compression', 'None', 'fps', 24);

writerObject = avifile(videoObjectName, 'compression', 'None', 'fps', 24);

for i = 1:num_images_without_object
    writer = addframe(writer, uint8(im_with_object));
    writerObject = addframe(writerObject, uint8(im_segm_object));
    
    if(flageWritingImages == 1)
        imwrite(uint8(im_with_object), [path_im_object int2str(i) '.bmp'], 'bmp');
        imwrite(uint8(im_segm_object), [path_object int2str(i) '_object.bmp'], 'bmp');
    end
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
    
    writer = addframe(writer, uint8(im_with_object)); 
    writerObject = addframe(writerObject, uint8(im_segm_object));
    
    if flageWritingImages == 1
        imwrite(uint8(im_with_object), [path_im_object int2str(i) '.bmp'], 'bmp');
        imwrite(uint8(im_segm_object), [path_object int2str(i) '_object.bmp'], 'bmp');
    end

    radius = radius + radius_step;
    angle = angle + angle_step;
end

writer = close(writer);
writerObject = close(writerObject);