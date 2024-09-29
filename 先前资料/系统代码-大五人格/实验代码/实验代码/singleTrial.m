function [rt acc] = singleTrial(wptr, cjMatrix, ID, imgMatrix_Fixation, cjSeries)

% prepare foldernames
picFolderName = 'Pics';

% prepare colors
bkgColor = [0 0 0];

% prepare Columns
TrialID_Column = 1;
Type_Column = 2;
Word_Column=3;
CorrectResponse_Column = 4;

% prepare parameters
fixation_Interval = 1.2;  
blank_Interval_200 = 0.2;     % 200 ms
timeUpperLimit = 3;
% blank_Interval_3000 = 60;
blank_Interval_1000 = 1;

% 设置按键的准备情况
KbName('UnifyKeyNames');

% 准备好按键的向量
KeyPressArray = KbName('f');       %定义按键

    
% 开始
% 空屏1秒
Screen('FillRect', wptr,bkgColor);  %准备黑屏
Screen('Flip', wptr);        %黑屏
WaitSecs(blank_Interval_1000);    %Duration

% % show 
% Screen('PutImage',wptr, imgMat_biyan);
% Screen('Flip',wptr);
% WaitSecs(blank_Interval_3000);
% 
% % show
% Screen('PutImage',wptr, imgMat_zhengyan);
% Screen('Flip',wptr);
% WaitSecs(blank_Interval_1000);

% show the fixation for
imgMatrix_Fixation=Screen('MakeTexture',wptr, imgMatrix_Fixation);
Screen('DrawTexture',wptr,imgMatrix_Fixation);
Screen('Flip',wptr);
WaitSecs(fixation_Interval);

% show the cj
cjItem_ID = cjMatrix(ID, Word_Column);
imgMatrix_BJ = cjSeries{cjItem_ID};
imgMatrix_BJ=Screen('MakeTexture',wptr, imgMatrix_BJ);
Screen('DrawTexture',wptr,imgMatrix_BJ);
Screen('Flip',wptr);

% mark single trial
t1_single=datestr(now,'mmmm dd,yyyy HH:MM:SS.FFF AM');

% chixu shijian
if cjMatrix(ID, CorrectResponse_Column)==1
    WaitSecs(blank_Interval_200);
    rt=blank_Interval_200;
    acc=1;
else
    t0 = GetSecs;   %启动计时器
    
    while 1     %等待被试反应
        [~, t, key_Code] = KbCheck;      %监听按键
        
        %
        if key_Code(KbName('f'))
            rt = t - t0;
            acc = 1;
            break;
            
            
            % 如果按键为ESC
        elseif key_Code(KbName('ESCAPE'))  %如果按键为esc，返回值为1
            rt = 999;    %ACC为999，进行标记，用于退出程序
            acc = 999;
            break;
            
            
            % 如果不按键，超时了
        else
            tt = GetSecs;   %启动计时器
            if tt-t0>timeUpperLimit
                rt = 3;
                acc = 0;
                break;
            end
            
%         else
%             rt = t - t0;
%             acc = 0;
%             break;
        end
    end
    
end

% open a .txt file for store the data
    txtFileName_Result = 'expTime_data.txt';
    fid = fopen(txtFileName_Result, 'a+');
%     t_all=etime(t1_all,t0_all);
    time = sprintf('%s', t1_single);
    fprintf(fid, '%s\r\n', time);
    fclose(fid);
end
