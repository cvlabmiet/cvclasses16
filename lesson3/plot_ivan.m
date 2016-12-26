f = fopen('recprec.txt', 'r');
data= cell(1);
l = fgets(f);
sobel_rec = [];
sobel_prec = [];
dog_rec = [];
dog_prec = [];
while (l ~= -1)
    algo = sscanf(l, '%s.bmp');
    l(1:length(algo)) = [];
    algo = strtok(algo, '_');
    [rp, c] = sscanf(l, '%f');
    rec = rp(1);
    prec = rp(2);
    if (strcmp(algo, 'Sobel'))
        sobel_rec(end+1) = rec;
        sobel_prec(end+1) = prec;
    end
    if (strcmp(algo, 'DoG'))
        dog_rec(end+1) = rec;
        dog_prec(end+1) = prec;
    end
    l = fgets(f);
end

%[sobel_prec, idx] = sort(sobel_prec);
%sobel_rec = sobel_rec(idx);
%[dog_prec, idx] = sort(dog_prec);
%dog_rec = dog_rec(idx);

figure; hold on; grid on
plot(sobel_rec, sobel_prec, '*-r', 'linewidth', 2);
plot(dog_rec, dog_prec, '*-b', 'linewidth', 2);
legend('Sobel', 'DoG 3x3')
xlabel('recall')
ylabel('precision')
title('Recall and precision')
