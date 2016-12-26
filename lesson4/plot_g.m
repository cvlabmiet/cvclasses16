fd = fopen('metrics.bin', 'rb');
fseek(fd, 0, 'eof');
nbytes = ftell(fd);
fseek(fd, 0, 'bof');

recall = 0;
precision = 0;
specificity = 0;
fpr = 0;
fnr = 0;
pwc = 0;
f_m = 0;

i = 1;
while(ftell(fd) ~= nbytes)
    recall(i) = fread(fd, 1, 'double');
    precision(i) = fread(fd, 1, 'double');
    specificity(i) = fread(fd, 1, 'double');
    fpr(i) = fread(fd, 1, 'double');
    fnr(i) = fread(fd, 1, 'double');
    pwc(i) = fread(fd, 1, 'double');
    f_m(i) = fread(fd, 1, 'double');
    i = i + 1;
end

make_plot(recall, 'recall');
make_plot(precision, 'precision');
make_plot(specificity, 'specificity');
make_plot(fpr, 'fpr');
make_plot(fnr, 'fnr');
make_plot(pwc, 'pwc');
make_plot(f_m, 'f-measure');

fclose(fd);
