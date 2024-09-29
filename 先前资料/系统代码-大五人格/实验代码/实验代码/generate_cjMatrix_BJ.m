function cjMatrix_BJ=generate_cjMatrix_BJ()

%prepare
trialNum=200;


trialID=1:trialNum;
trialID=trialID';

%+type
type1=ones(60,1);
type2=ones(60,1)*2;
type3=ones(60,1)*3;
type4=ones(20,1)*4;
type=[type1;type2;type3;type4];

%     %import textword
%     feature('DefaultCharacterSet', 'UTF8');
%     FilePath='C:\Users\lenovo\Desktop\本基\情绪词.txt';
%     WordName=importdata(FilePath);
%     WordName_change=str2num(char(WordName));
% +WordNameArr
WordNameArr=1:trialNum;
WordNameArr=WordNameArr';

%+correctResponseArr
correctResponseArr=[];
for i=1:trialNum
    if type(i,1)==4
        tmpCorrectResponseValue=2;
    else
        tmpCorrectResponseValue=1;
    end
    correctResponseArr=[correctResponseArr;tmpCorrectResponseValue];
end

coreMatrix_3col=[type WordNameArr correctResponseArr];

%rand
tmpRandArr=randperm(trialNum)';

tmpMat_addRarr=[coreMatrix_3col tmpRandArr];
tmpMat_rand=sortrows(tmpMat_addRarr,4);

%tidiao randapp
cjMatrix_r_3=tmpMat_rand(:,1:3);

%add trialIDArray
cjMatrix_BJ=[trialID cjMatrix_r_3];

end