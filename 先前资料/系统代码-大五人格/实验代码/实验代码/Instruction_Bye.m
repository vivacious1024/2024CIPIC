%filename is:Instruction_Bye.m

%clear
clear;clc;close all;

foldername='Pics';
if ~exist(foldername,'dir')
    mkdir(foldername);
end

%创建图像
hFigure=figure(1);
set(hFigure,'position',[100 50 900 600],'color','k');

%创建控制界面
hText_JieShu=uicontrol('style','text','String','实验结束，感谢您的参与','fontname','Microsoft Yahei','position',[151 150 600 300],'fontsize',60,'backgroundcolor','k','foregroundcolor','w');

%抓拍
hFrame=getframe(gcf);
imgBye=hFrame.cdata;
jpgFileName_bye='Instruction_Bye.jpg';
jpgPathName_bye=sprintf('%s/%s',foldername,jpgFileName_bye);
imwrite(imgBye,jpgPathName_bye,'jpg');