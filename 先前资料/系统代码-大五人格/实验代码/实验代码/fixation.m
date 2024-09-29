%filename is:fixation.m

%clear
clc;clear;close all;

%prepare
picFolderName='Pics';
if ~exist(picFolderName,'dir')
    mkdir(picFolderName);
end


% %得到当前路径
% CWPath=fileparts(mfilename('fullpath'));

%制图
hFigure=figure(1);
set(hFigure,'position',[100 100 300 300]);

%坐标轴
hAxes=axes('parent',hFigure);
set(hAxes,'units','pixels','position',[1 1 300 300]);

%创建图形矩阵
imgMat=zeros(300,300,3);
imgMat(121:180,141:160,1)=1;
imgMat(121:180,141:160,2)=1;
imgMat(121:180,141:160,3)=1;

imgMat(141:160,121:180,1)=1;
imgMat(141:160,121:180,2)=1;
imgMat(141:160,121:180,3)=1;

%展示图片
imshow(imgMat,'parent',hAxes);

%构建框架
hFrame=getframe(gcf);
imgCross=hFrame.cdata;
imgCross300=imresize(imgCross,[200 200]);%调整大小

%保存
crossFileName='fixation.jpg';
fixationPathName=sprintf('%s/%s',picFolderName,crossFileName);
imwrite(imgCross300,fixationPathName,'jpg');