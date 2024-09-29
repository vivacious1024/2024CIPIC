%filename is:BJExp.m

%clear all
clear;clc;close all;

%prepare
picsFolderName='Pics';

trialNum=200;
for i=1:trialNum
    jpgFileName=sprintf('S%d.jpg',i);
    jpgPathName=sprintf('%s/%s', picsFolderName, jpgFileName);
    imgMat_cjCells{i} = imread(jpgPathName);
end
imgMat_cjCells = imgMat_cjCells';

try
    %filename
    jpgFileName_Instruction = 'Instruction_Start1.jpg';
    jpgPathName_Instruction = sprintf('%s/%s', picsFolderName, jpgFileName_Instruction);
    %
    imgMat_Instruction=imread(jpgPathName_Instruction);
    
    % 准备好zhengyan的图像
    zhengyanFileName = 'ZhengYan.jpg';
    zhengyanPathName = sprintf('%s/%s', picsFolderName, zhengyanFileName);
    zhengyan_imgMatrix = imread(zhengyanPathName);
    
    % 准备好biyan的图像
    biyanFileName = 'BiYan.jpg';
    biyanPathName = sprintf('%s/%s', picsFolderName, biyanFileName);
    biyan_imgMatrix = imread(biyanPathName);
    
    % 准备好fixation的图像
    fixationFileName = 'fixation.jpg';
    fixationPathName = sprintf('%s/%s', picsFolderName, fixationFileName);
    fixation_imgMatrix = imread(fixationPathName);
    
    %prepare color for background
    bkgColor=[0 0 0];
    
    %openwindow
    Screen('Preference', 'SkipSyncTests', 1);
    Screen('Preference', 'ConserveVRAM', 64);
    [wptr, rect] = Screen('OpenWindow', 0, bkgColor);
    
    % 关闭键盘监听
    ListenChar(2);
    
    %yin cang shu biao
    HideCursor;
    
%     % show
%     Screen('PutImage',wptr, imgMat_biyan);
%     Screen('Flip',wptr);
%     WaitSecs(60);
%     
%     % show
%     Screen('PutImage',wptr, imgMat_zhengyan);
%     Screen('Flip',wptr);
%     WaitSecs(1);
%     
%     % show the fixation for
%     Screen('PutImage',wptr, imgMatrix_Fixation);
%     Screen('Flip',wptr);
%     WaitSecs(60);
%     t0_all=GetSecs;
%     t0_all=clock
    
    % show the instruction
    Instruction_PTB(wptr, imgMat_Instruction);
    
    t0_all=datestr(now,'mmmm dd,yyyy HH:MM:SS.FFF AM');
    % 等待被试按键
    % KbWait;
    
    cjMatrix=generate_cjMatrix_BJ();
    
    % open a .txt file for store the data
    txtFileName_Result = 'expTimePressure_data.txt';
    fid = fopen(txtFileName_Result, 'a+');
    % LOOP: index is i
    for i = 1:length(cjMatrix)
        [rt acc] = singleTrial(wptr, cjMatrix, i, fixation_imgMatrix, imgMat_cjCells);
        if acc == 999
            % 显示鼠标
            break;
        end
        
        tmpArr = [cjMatrix(i,:) rt acc];
        time = sprintf('%d\t%d\t%d\t%d\t%.3f\t%d', tmpArr);
        fprintf(fid, '%s\r\n', time);
    end
    fclose(fid);
    % show
    biyan_imgMatrix=Screen('MakeTexture',wptr, biyan_imgMatrix);
    Screen('DrawTexture',wptr,biyan_imgMatrix);
    Screen('Flip',wptr);
    WaitSecs(3);
    
    % show the fixation for
    fixation_imgMatrix=Screen('MakeTexture',wptr, fixation_imgMatrix);
    Screen('DrawTexture',wptr,fixation_imgMatrix);
    Screen('Flip',wptr);
    t1_all=datestr(now,'mmmm dd,yyyy HH:MM:SS.FFF AM');%zhengyan stimuli
    WaitSecs(60);
    
    % show
    zhengyan_imgMatrix=Screen('MakeTexture',wptr, zhengyan_imgMatrix);
    Screen('DrawTexture',wptr,zhengyan_imgMatrix);
    Screen('Flip',wptr);
    t2_all=datestr(now,'mmmm dd,yyyy HH:MM:SS.FFF AM');%biyan stimuli
    WaitSecs(60);
    load chirp;
    sound(y,Fs);
    
%     t1_all=GetSecs;
%     t1_all=clock;
    t3_all=datestr(now,'mmmm dd,yyyy HH:MM:SS.FFF AM');
    % Bye bye
    % filename
    jpgFileName_Instruction = 'Instruction_Bye.jpg';
    jpgPathName_Instruction = sprintf('%s/%s', picsFolderName, jpgFileName_Instruction);
    imgMat_Instruction_Bye = imread(jpgPathName_Instruction);
    Instruction_PTB(wptr, imgMat_Instruction_Bye);
    
    
    
    % open a .txt file for store the data
    txtFileName_Result = 'expTimePressure_data.txt';
    fid = fopen(txtFileName_Result, 'a+');
%     t_all=etime(t1_all,t0_all);
    time = sprintf('%s\n%s\n%s\n%s\n', t0_all,t1_all,t2_all,t3_all);
    fprintf(fid, '%s\r\n', time);
    fclose(fid);
    
    ShowCursor;
    ListenChar(1);
    Screen('CloseAll');
    
catch
    ShowCursor;
    ListenChar(1);
    Screen('CloseAll');
    Priority(0);
    psychrethrow(psychlasterror);
end
