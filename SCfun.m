function [sizeSC,meanrange,meandis,maxdis,NSC,minrange,result]=SCfun(data,N,i,center)
%N=number of charging station
%i=number of subcluster
if N<=5
    SC=data(data(:,10)==i,:);
else
    SC=data(data(:,9)==i,:);
end
sizeSC=size(SC);
sizeSC=sizeSC(1,1);
meanrange=mean(SC(:,8));
minrange=min(SC(:,8));
if N==1
    dis=((SC(:,4)*54.6-center(i+2,2)*54.6).^2+(SC(:,5)*69-center(i+2,3)*69).^2).^0.5;
elseif N==2
    dis=((SC(:,4)*54.6-center(i+7,2)*54.6).^2+(SC(:,5)*69-center(i+7,3)*69).^2).^0.5;
elseif N==3
    dis=((SC(:,4)*54.6-center(i+12,2)*54.6).^2+(SC(:,5)*69-center(i+12,3)*69).^2).^0.5;  
elseif N==4
    dis=((SC(:,4)*54.6-center(i+17,2)*54.6).^2+(SC(:,5)*69-center(i+17,3)*69).^2).^0.5; 
elseif N==5
    dis=((SC(:,4)*54.6-center(i+22,2)*54.6).^2+(SC(:,5)*69-center(i+22,3)*69).^2).^0.5; 
elseif N==6
    dis=((SC(:,4)*54.6-center(i+27,2)*54.6).^2+(SC(:,5)*69-center(i+27,3)*69).^2).^0.5;
elseif N==7
    dis=((SC(:,4)*54.6-center(i+32,2)*54.6).^2+(SC(:,5)*69-center(i+32,3)*69).^2).^0.5; 
elseif N==8
    dis=((SC(:,4)*54.6-center(i+37,2)*54.6).^2+(SC(:,5)*69-center(i+37,3)*69).^2).^0.5; 
elseif N==9
    dis=((SC(:,4)*54.6-center(i+42,2)*54.6).^2+(SC(:,5)*69-center(i+42,3)*69).^2).^0.5;
elseif N==10
    dis=((SC(:,4)*54.6-center(i+48,2)*54.6).^2+(SC(:,5)*69-center(i+48,3)*69).^2).^0.5; 
end
meandis=mean(dis);
maxdis=max(dis);
%number of charging station
NSC=(sizeSC*meanrange*0.34*0.5*0.3)/(20*3);
result=[sizeSC,meanrange,meandis,maxdis,NSC,minrange]
%disp(result);
end