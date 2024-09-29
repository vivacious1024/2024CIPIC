%filename is:instruction.m

%clear
clear;clc;close all;

%prepare
foldername='Pics';
if ~exist(foldername,'dir')
    mkdir(foldername);
end

%create a figure
hFigure=figure(1);
set(hFigure,'position',[100 50 900 600],'color','k');

%create a uicontrol for zhidaoyu
hText_Kaishi=uicontrol('style','text','String','指导语','fontname','Microsoft Yahei','position',[351 500 200 60],'fontsize',36,'backgroundcolor','k','foregroundcolor','w');

%
tmpInstr=sprintf('请您根据屏幕上方呈现的词语按键选择\n只有出现名字按f键，其余不作反应');
hText_Content=uicontrol('style','text','String',tmpInstr,'fontname','Microsoft Yahei','position',[101 150 700 300],'fontsize',24,'backgroundcolor','k','foregroundcolor','w','horizontalAlign','center');

tmpGo='如果理解指导语，请按空格键开始正式实验。';
hText_Go=uicontrol('style','text','HorizontalAlign','center','String',tmpGo,'fontname','Microsoft Yahei','position',[101 150 700 40],'fontsize',24,'backgroundcolor','k','foregroundcolor','w');

%snapshot
hFrame=getframe(gcf);
imgInstruction=hFrame.cdata;

jpgFilename_instruction='Instruction_Start1.jpg';
jpgPathname_instruction=sprintf('%s/%s',foldername,jpgFilename_instruction);
imwrite(imgInstruction,jpgPathname_instruction,'jpg');