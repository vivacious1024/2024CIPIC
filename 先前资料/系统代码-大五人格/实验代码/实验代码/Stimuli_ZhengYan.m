%filename is:Stimuli_ZhengYan.m

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
hText_ZhengYan=uicontrol('style','text','String','请闭眼，听到闹钟后睁眼','fontname','Microsoft YaHei','Position',[151 150 600 300],'fontsize',60,'backgroundcolor','k','foregroundcolor','w');

%抓拍
hFrame=getframe(gcf);
imgZhengYan=hFrame.cdata;
jpgFileName_zhengyan='ZhengYan.jpg';
jpgPathName_zhengyan=sprintf('%s/%s',foldername,jpgFileName_zhengyan);
imwrite(imgZhengYan,jpgPathName_zhengyan,'jpg');