clear
clc
close all

fileName = 'measures.txt';
path = './';

flageWritingImages = 1;

fId = fopen(fileName, 'rt');

if fId == -1 
    error('File is not opened'); 
end 

numFrames = 24*23;
numMetrics = 7;

[value, ~] = fscanf(fId,'%d %f %f %f %f %f %f %f\n', [(numMetrics + 1) numFrames]);

fclose(fId);

metrics_names = {'F-measure', 'Precision', 'Recall', 'Specificity','False Positive Rate','False Negative Rate', 'Percentage of Wrong Classifications'};
for i = 1:numMetrics
    f = figure();
    hold on
    grid on
    xlabel('# frame')
    ylabel(metrics_names(i))
    title(metrics_names(i))
    plot(value(1,:), value((i + 1),:));
    axis([0 (numFrames + 10) (min(value((i + 1),:)) - 0.1) (max(value((i + 1),:)) + 0.1)]);
    if(flageWritingImages == 1)
        print(f, '-dpng', [path char(metrics_names(i))]);
    end
end