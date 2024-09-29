%filename is:Stimuli_BiYan.m

%clear
clear;clc;close all;

foldername='Pics';
if ~exist(foldername,'dir')
    mkdir(foldername);
end

%create a figure
hFigure=figure(1);
set(hFigure,'position',[100 50 900 600],'color','k');

%create control
hText_ZhengYan=uicontrol('style','text','String','请保持睁眼，并注视中央的注视点','fontname','Microsoft YaHei','Position',[151 150 600 300],'fontsize',60,'backgroundcolor','k','foregroundcolor','w');

%抓拍
hFrame=getframe(gcf);
imgBiYan=hFrame.cdata;
jpgFileName_biyan='BiYan.jpg';
jpgPathName_biyan=sprintf('%s/%s',foldername,jpgFileName_biyan);
imwrite(imgBiYan,jpgPathName_biyan,'jpg');