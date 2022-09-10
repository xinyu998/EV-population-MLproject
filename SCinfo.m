clc
clear all
data00=xlsread('cluster00.xlsx');
data01=xlsread('cluster01.xlsx');
data02=xlsread('cluster02.xlsx');
data03=xlsread('cluster03.xlsx');
data04=xlsread('cluster04.xlsx');
data1=xlsread('cluster1.xlsx');
data2=xlsread('cluster2.xlsx');
data3=xlsread('cluster3.xlsx');
data4=xlsread('cluster4.xlsx');
data5=xlsread('cluster5.xlsx');
center=xlsread('Chargingstation.xlsx');
%% 1st charging staion info
for i=0:4
   SCfun(data00,1,i,center) 
end
%% 2st charging staion info
for i=0:4
   SCfun(data01,2,i,center) 
end   
%% 3st charging staion info
for i=0:4
   SCfun(data02,3,i,center) 
end
%% 4st charging staion info
for i=0:4
   SCfun(data03,4,i,center) 
end
%% 5st charging staion info
for i=0:4
   SCfun(data04,5,i,center) 
end
%% 6st charging staion info
for i=0:4
   SCfun(data1,6,i,center) 
end
%% 7st charging staion info
for i=0:4
   SCfun(data2,7,i,center) 
end
%% 8st charging staion info
for i=0:4
   SCfun(data3,8,i,center) 
end
%% 9st charging staion info
for i=0:5
   SCfun(data4,9,i,center) 
end
%% 10st charging staion info
for i=0:4
   SCfun(data5,10,i,center) 
end

