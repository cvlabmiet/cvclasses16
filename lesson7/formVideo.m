clear
clc
close all

fps = 24;

num_sec_video = 30;%in seconds
num_sec_without_obj = 0;%in seconds
num_sec_stop = 10;%in seconds

num_im = fps * num_sec_video;
num_im_without_obj = fps * num_sec_without_obj;
num_im_stop_obj_bot = fps * round(((num_sec_video - num_sec_stop) / 2)); 
num_im_stop_obj_upp = fps * round(((num_sec_video + num_sec_stop) / 2)); 

videoName = './Surveillance System/ImageWithObject.avi';

radius_step = 0.4;
angle_step = 0.05;

im = imread('./Surveillance System/Rostislav.png');
im_object = imread('./Surveillance System/romb.png');
%im_object = rgb2gray(im_object);

[M N ~] = size(im);
[M_Object N_Object ~] = size(im_object);

radius = 0;
angle = 0;

x = round(radius * cos(angle) + M / 2);
y = round(radius * sin(angle) + N / 2);

writer = avifile(videoName, 'compression', 'None', 'fps', fps);

for i = 1:num_im
    
    im_with_object = im;
    
    if(i > num_im_without_obj)
        
        for j = 1:M_Object
            for k = 1:N_Object
                if ( sum(im_object(j, k, :) ~= 0) == 3 )
                    im_with_object(x + j, y + k, :) = im_object(j, k, :);
                end
            end
        end
        
        if(( i <= num_im_stop_obj_bot )||( i > num_im_stop_obj_upp  ))
            
            radius = radius + radius_step;
            angle = angle + angle_step;
            
            x = round(radius * cos(angle) + M / 2);
            y = round(radius * sin(angle) + N / 2);
        end
       
    end
    
    writer = addframe(writer, uint8(im_with_object));
end

writer = close(writer);