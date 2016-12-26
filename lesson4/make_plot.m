function make_plot(data, name)
figure; hold on; grid on
plot(2:numel(data), data(2:end));
title(name)
xlabel('frame')
print(strcat(name, '.jpg'), '-djpeg');
end