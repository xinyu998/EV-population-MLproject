clc
clear all
data=xlsread('K-meanresults.xlsx');
x=[1:50];
y1=data(:,3);
y2=data(:,5);
%y2(41:45,:)=[23.4,23.5,25,48,16];
figure
plot(x,y1,'LineWidth',2)
hold on
plot(x,y2,'LineWidth',2)
grid on
hold off
title('Min EV range vs Max EV to CS distance')
xlabel('Charging Station area')
ylabel('Miles')
legend('Min EV range','Max EV to CS distance')
%number of EV plot
EV=data(:,1);
SC=data(:,6);
figure
plot(x,EV,'LineWidth',2)
hold on
plot(x,SC,'LineWidth',2)
grid on
hold off
title('Number of EV and Charging ports')
xlabel('Charging Station area')
ylabel('Numbers')
legend('Number of EV','Number of Charging ports in each area')