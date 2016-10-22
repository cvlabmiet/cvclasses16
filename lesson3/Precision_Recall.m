clear
clc
close all

Name = './Scharr_vs_LoG/Precision_Recall.txt';
f = fopen(Name, 'rt');

if f == -1 
    error('Invalid file name'); 
end 

[value, ~] = fscanf(f,'%d %f %f\n', [3 10]);

fclose(f);


f1 = figure();
hold on
grid on
xlabel('Precision')
ylabel('Recall')
title('Recall / Precision')

for i = 1:size(value,2)
    if(value(1,i) == 80)
        index = i;
        break;
    end
end

plot(value(2,1:index), value(3,1:index),'-*r');
plot(value(2,(index + 1):end), value(3,(index + 1):end),'-*');

legend('Scharr 3x3','LoG 3x3');

 print(f1, '-dpng', 'figure');