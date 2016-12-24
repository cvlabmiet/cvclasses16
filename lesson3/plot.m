[re_log, pre_log] = textread('C:\Users\Elena\cvclasses16\lesson3\re_pre_LoG7.txt', '%f %f')
[re_prew, pre_prew] = textread('C:\Users\Elena\cvclasses16\lesson3\re_pre_prewwit.txt', '%f %f')
figure; hold on
plot(  pre_log, re_log,'-*b');
plot(  pre_prew, re_prew, '-*r');
grid on
legend('LoG 7õ7', 'Prewitt diagonal');
ylabel 'recall'
xlabel 'precision'
plot( pre_log(1),re_log(1),  'ob');
plot( pre_prew(1),re_prew(1),  'or');


