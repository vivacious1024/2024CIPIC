%filename is:stimuli.m

%clear
clear;clc;close all;

%prepare
foldername='Pics';
if ~exist(foldername,'dir')
    mkdir(foldername);
end

feature('DefaultCharacterSet', 'UTF8');
FilePath='C:\Users\lenovo\Desktop\本基\情绪词.txt';
WordName=importdata(FilePath);

for i=1:200
    WordDigit=WordName{i};
    
    hFigure=figure(1);
    set(hFigure,'position',[100 100 300 300]);
    
    hAxes=axes('parent',hFigure);
    set(hAxes,'units','pixels','position',[1 1 300 300]);
    
    imgMat=zeros(300,300,3);
    imshow(imgMat,'parent',hAxes);
    axis([1 300 1 300]);
    
    text(150,150,WordDigit,'HorizontalAlign','center','fontname','Microsoft Yahei','fontsize',80,'color','w');
    
    hFrame=getframe(gcf);
    imgDigit=hFrame.cdata;
    
    jpgFilename=sprintf('S%d.jpg',i);
    jpgPathname=sprintf('%s/%s',foldername,jpgFilename);
    
    %save
    imwrite(imgDigit,jpgPathname,'jpg');
    
    close(hFigure);
end


