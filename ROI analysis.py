# convert to matlab and implement in 6.ROI_analysis.py


function [planedata]=getplanedata(filename,planenumber,trialnumber)


% clear all
% planenumber=0;
% trialnumber=0;
datastem=strcat('/planes/item_',num2str(planenumber),'/trials/item_',num2str(trialnumber),'/');

imagedata=single(h5read(filename,strcat(datastem,'/images/data')));

planedata.imagedata=imagedata;

badframes=find(ismember(h5read(filename,strcat(datastem,'/images/coords/mask good frames')),'FALSE'));
goodframes=find(ismember(h5read(filename,strcat(datastem,'/images/coords/mask good frames')),'TRUE'));

planedata.badframes=badframes;
planedata.goodframes=goodframes;
% badframes=h5read(filename,strcat(datastem,'/images/coords/mask good frames'));

% clear pooledata
flatdata=squeeze(mean(single(imagedata),1));
for n=1:30
    pooledata(n,:)=squeeze(mean(flatdata([1:20]+(n-1)*20,:),1));
end
repval=median(pooledata(:,goodframes),2);
for n=badframes'
pooledata(:,n)=repval;
end

planedata.pooledata=pooledata;

times=h5read(filename,strcat(datastem,'/images/coords/Time (ms)'));

planedata.times=times-times(1);

% figure
% for n=1:30
% plot(times,zscore(pooledata(n,:))+n*2)
% hold on
% end

behaviordata=h5read(filename,strcat(datastem,'/behavior/values'))';
behaviortimes=behaviordata(:,1);
allbehavior=cumsum(behaviordata(:,2:end)')';
tailtrace=allbehavior(:,15);


planedata.behaviortimes=behaviortimes-times(1);
planedata.tailtrace=tailtrace;

% plot(behaviortimes,tailtrace/50)

protocol=h5read(filename,strcat(datastem,'/protocol/values'));
protocoltimes=protocol(1,:);
cs_start_time = protocoltimes(find(protocol(2,:)>0,1,'first'));
cs_end_time = protocoltimes(find(protocol(3,:)>0,1,'first'));
us_start_time = protocoltimes(find(protocol(4,:)>0,1,'first'));
us_end_time = protocoltimes(find(protocol(5,:)>0,1,'first'));

% line([cs_start_time cs_start_time],[-1 70],'Color',[0 1 0])
% line([cs_end_time cs_end_time],[-1 70],'Color',[0 1 0])

planedata.timings.cs_start_time=cs_start_time-times(1);
planedata.timings.cs_end_time=cs_end_time-times(1);

% if ~isempty(us_start_time)
% line([us_start_time us_start_time],[-1 70],'Color',[1 0 0])
% line([us_end_time us_end_time],[-1 70],'Color',[1 0 0])
planedata.timings.us_start_time=us_start_time-times(1);
planedata.timings.us_end_time=us_end_time-times(1);

end



function [roidata]=findRoisSinglePane20260105(alldata,thisplanecorr,roiparams)

deTrendScl=roiparams.deTrendScl;
maximumroisize=roiparams.maximumroisize;
%this controls how easily the ROIs will grow. You want to set it so that
%most cells will grow out to the max size
corrthresh=roiparams.corrthresh;
%this controls how deep you will go looking for cells
stopthresh=roiparams.stopthresh;
%this is an alternative limit to avoid making rois for ever
maxperplane=roiparams.maxperplane;
showfigs=roiparams.showfigs;


%now start the ROI detection
[maxval,maxindex]=max(thisplanecorr(:));
[i,j]= ind2sub(size(thisplanecorr),maxindex)
i=i(1);j=j(1);
thisplanecorr(i,j)=0;

% pathname,fullMatfilePath,maximumroisize,corrthresh,stopthresh,deTrendScl,frameOffset,showfigs
%roi finding parameters
    clear FplaneData 
    
planeData4ROIs=alldata.imagedata(:,:,find(alldata.maskgoodframes));

   myfilter=ones(deTrendScl,1)*(-1/deTrendScl);
   myfilter(ceil(deTrendScl/2))= myfilter(ceil(deTrendScl/2))+1;
% planeData4ROIs=zeros(h,w,length(frames2use),'single');
% framecount=0;
%read in all the data
% for n=frames2use'
%     framecount=framecount+1;
%     frame2read=n-startframe+1;
%     file2read=fullfile(savefoldermc,strcat('splitfilesmcrig',num2str(floor((frame2read-1)/1000),'%0.4i'),'.tif'));
%     thisplane=single(imread(file2read,'TIF',mod(frame2read-1,1000)+1));
%     planeData4ROIs(:,:,framecount)=thisplane;
% end

% parameters that need to be set

% fh=figure
% fi=figure

imagetoshow=zeros(size(thisplanecorr,1),size(thisplanecorr,2));

% timep=size(planeData,3);
timepfull=size(planeData4ROIs,3);

allrois=zeros(size(thisplanecorr(:)));


thisroi=false([size(thisplanecorr,1) size(thisplanecorr,2) ]);
allroioutlines=false([size(thisplanecorr,1) size(thisplanecorr,2) ]);
% thisroiEliminated=false([size(thisplanecorr,1) size(thisplanecorr,2) ]);
disp(i)
disp('here')
thisroi(i,j)=1;


thistrace=squeeze(double(planeData4ROIs(i,j,:)));
thistracef=conv(thistrace,myfilter,'same');
thistracetrue=squeeze(double(planeData4ROIs(i,j,:)));


thisavcorr=0;

thisnum=1;
corrmapval=[];
avcorrval=[];
allnums=[];
numcells=0;

while ((max(thisplanecorr(:))>stopthresh)&(numcells<maxperplane))
numcells=numcells+1;
processing=true;
iter=0;
% disp('here')
%entering processingloop
    while (processing)
    % for iter=1:100
        iter=iter+1;

        %grow roipoints
        newroipoints=imdilate(thisroi,[0 1 0 ; 1 0 1 ; 0 1 0]);
        %find new roi point indices
        newroiindices=find(newroipoints&~thisroi);

%         [newroix,newroiy]= ind2sub([size(thisplanecorr,1) size(thisplanecorr,2) ],newroiindices);
% 
%         newroiindices=sub2ind([size(thisplanecorr,1) size(thisplanecorr,2) ],newroix,newroiy); 
        
        alreadyfound=allrois(newroiindices);
        newroiindices=newroiindices(find(~alreadyfound));

%         if (isempty(newroiindices))
%             processing=false;
%         else
                [i,j]= ind2sub([size(thisplanecorr,1) size(thisplanecorr,2) ],newroiindices);
                clear alltraces thiscorr


                if (~isempty(i))
                    alltracesf=zeros(timepfull,length(i));
                    alltracestrue=zeros(timepfull,length(i));
                for n=1:length(i)

                        newtrace=squeeze(double(planeData4ROIs(i(n),j(n),:)));
                        
                        newtracef=conv(newtrace,myfilter,'same');
                        newtracetrue=squeeze(double(planeData4ROIs(i(n),j(n),:)));

              	alltracesf(:,n)=newtracef;
              	alltracestrue(:,n)=newtracetrue;
                
                thiscorr(n)=corr(alltracesf(:,n),thistracef(:));

                end
               %now we have all traces and their correlation coefficients

               %i,j,alltraces,thiscorr
               [sortcorr,sortind]=sort(thiscorr);
               num2inc=ceil(length(i)/2);
               bestind=sortind(end-num2inc+1:end);
               thiscorr=thiscorr(bestind);
               alltraces=alltracesf(:,bestind);
               alltracestrue=alltracestrue(:,bestind);
               i=i(bestind);
               j=j(bestind);

        if (iter==1)
          sigpixels=find(thiscorr>0);  
        else 
        sigpixels=find(thiscorr>corrthresh);
        end
        
        
%     alltraces=alltraces(:,sigpixels);
    alltracestrue=alltracestrue(:,sigpixels);
    tracecorrs=thiscorr(sigpixels);
    
    i=i(sigpixels);
    j=j(sigpixels);
    
        thisroi(sub2ind([size(thisplanecorr,1) size(thisplanecorr,2) ],i,j))=1;
        thisplanecorr(sub2ind([size(thisplanecorr,1) size(thisplanecorr,2) ],i,j))=0;
   
        
    thistrace=((thistrace*thisnum)+sum(alltracestrue,2))/(size(alltracestrue,2)+thisnum);
    thistracetrue=thistrace;
    thistracef=conv(thistrace,myfilter,'same');
    
    thisavcorr=((thisavcorr*(thisnum-1))+sum(tracecorrs))/(size(alltracestrue,2)+thisnum-1);
    thisnum=thisnum+size(alltracestrue,2);




        if (isempty(sigpixels))
            processing=false;
        end
   
    newroipoints=imclose(thisroi,[0 1 0 ; 1 0 1 ; 0 1 0]);
    newroiindices=find(newroipoints&~thisroi);

%     [newroix,newroiy]= ind2sub([size(thisplanecorr,1) size(thisplanecorr,2) ],newroiindices);

    alreadyfound=allrois(newroiindices);
    newroiindices=newroiindices(find(~alreadyfound));

        [i,j]= ind2sub([size(thisplanecorr,1) size(thisplanecorr,2) ],newroiindices);
        clear alltraces thiscorr
        
        if (~isempty(i))
            alltracesf=zeros(timepfull,length(i));
            alltracestrue=zeros(timepfull,length(i));
            for n=1:length(i)
                newtrace=squeeze(double(planeData4ROIs(i(n),j(n),:)));
                        newtracef=conv(newtrace,myfilter,'same');
                newtracetrue=squeeze(double(planeData4ROIs(i(n),j(n),:)));
                

                alltracesf(:,n)=newtracef;
                alltracestrue(:,n)=newtracetrue;
                
                thiscorr(n)=corr(alltracesf(:,n),thistracef(:));

%                 thiscorr(n)=corr(diff(alltraces(f2use,n)),diff(thistrace(f2use)));
            end
            

            thisroi(sub2ind([size(thisplanecorr,1) size(thisplanecorr,2) ],i,j))=1;
          %  thisroi(sub2ind(size(all[size(thiscorr,1) size(thiscorr,2) ]corrsimbig),i,j,k))=1;
            thisplanecorr(sub2ind([size(thisplanecorr,1) size(thisplanecorr,2) ],i,j))=0;

        thistrace=((thistrace*thisnum)+sum(alltracestrue,2))/(size(alltracestrue,2)+thisnum);
        thistracetrue=alltracestrue;
        thistracef=conv(thistrace,myfilter,'same');
        thisavcorr=((thisavcorr*(thisnum-1))+sum(thiscorr))/(size(alltracestrue,2)+thisnum-1);
        thisnum=thisnum+size(alltracestrue,2);
        
        
        end
        
        if (thisnum>maximumroisize)
            processing=false;
        end
         else
                    processing=false;
                end
    end
% % hold on
if (numcells==1)
    mytraces=zscore(thistrace);
%     mytracestrue=thistracetrue;
    mytracesraw=thistrace;
else
mytraces=[mytraces zscore(thistrace)];
% mytracestrue=[mytracestrue thistracetrue];
mytracesraw=[mytracesraw thistrace];
end
% plot(zscore(thistrace))

[thisx thisy ]=ind2sub([size(thisplanecorr,1) size(thisplanecorr,2) ],find(thisroi==1));
thisbound=bwboundaries(thisroi);
allboundpts=sub2ind(size(thisroi),thisbound{1}(:,1),thisbound{1}(:,2));
if (thisnum>10)
allroioutlines(allboundpts)=1;
end

allrois(sub2ind([size(thisplanecorr,1) size(thisplanecorr,2) ],thisx,thisy))=numcells;

if (showfigs&(mod(numcells,ceil(sqrt(numcells)))==0))
figure(fh)
subplot(2,1,2)
imagesc(mytraces(:,:)')
subplot(2,1,1)
imshow(reshape(allrois,h,w))


figure(fi)
imagetoshow=imagetoshow+thisroi;
imagesc(imagetoshow);
drawnow
end


% figure(fj)
% [roii,roij]= ind2sub([size(thisplanecorr,1) size(thisplanecorr,2) ],find(allrois==numcells));



allnums(numcells)=thisnum;
corrmapval(numcells)=maxval;
avcorrval(numcells)=thisavcorr;

thisroi=thisroi*0;
% thisroiEliminated=thisroiEliminated*0;
[maxval,maxindex]=max(thisplanecorr(:));
[i,j]= ind2sub([size(thisplanecorr,1) size(thisplanecorr,2) ],maxindex);
i=i(1);
j=j(1);
thisplanecorr(i,j)=0;


thisroi(i,j)=1;



thistrace=squeeze(double(planeData4ROIs(i,j,:)));
thistracef=conv(thistrace,myfilter,'same');
thistracetrue=squeeze(double(planeData4ROIs(i,j,:)));

% thistrace=squeeze(double(planeData(i,j,:)));
% thistracetrue=squeeze(double(planeData4ROIs(i,j,:)));

thisnum=1;
% if (mod(numcells,100)==0)
% save(fullfile(corrfolder,'roisave',strcat('plane',num2str(whichplane,'%0.5i'),'currentdata0.mat')),'allrois','allnums','mytraces','mytracesraw')
% else if (mod(numcells,100)==10)
% save(fullfile(corrfolder,'roisave',strcat('plane',num2str(whichplane,'%0.5i'),'currentdata1.mat')),'allrois','allnums','mytraces','mytracesraw')
%     end
% end

end

roidata.allnums=allnums;
roidata.corrmapval=corrmapval;
roidata.avcorrval=avcorrval;

roidata.mytraces=mytraces;
% mytracestrue=[mytracestrue thistracetrue];
roidata.mytracesraw=mytracesraw;
roidata.allrois=allrois;

end



function alldata=concatsingleplanefile(whichfolder,whichfile,whichplane,savedata,overwrite)

numbertrials=16; %should be determined automatically
% figure
% filename = 'L:\For Mike\20241015_03_delay_2p-9_mitfaminusminusca8e1bgcamp6s_6dpf\20241015_03_4. Activity_maps_data.h5'

filename = fullfile(whichfolder,whichfile);
savefolder= fullfile(whichfolder,'savefolder');
mkdir(savefolder);

savefilename=strcat('plane',num2str(whichplane,'%0.5i'),'.mat');
fullsavepath=fullfile(savefolder,savefilename);

if ((exist(fullsavepath,'file'))&(~overwrite))
    load(fullsavepath,'alldata');
    return
    
else

whichtrial=0;

trialhasus=zeros(numbertrials,1);
thisdata=getplanedata(filename,whichplane,whichtrial);

imagedata=thisdata.imagedata;
badframes=thisdata.badframes;
goodframes=thisdata.goodframes;
times=thisdata.times;

compressedtimes=times;
compressedtimes=compressedtimes-times(1);

interframetime=mean(diff(times));
nextrialtime=times(end)+interframetime;

behaviortimes=thisdata.behaviortimes;

trialstarttime(1)=0;
trialendtime(1)=nextrialtime;

tailtrace=thisdata.tailtrace;

lastbehaviorframe=find(behaviortimes<nextrialtime,1,'last');
if (~isempty(lastbehaviorframe));
    behaviortimes(lastbehaviorframe:end)=[];
    tailtrace(lastbehaviorframe:end)=[];
end

behinterframetime=mean(diff(behaviortimes));
nexbehtrialtime=behaviortimes(end)+behinterframetime;

compressedbehaviortimes=behaviortimes;
compressedbehaviortimes=compressedbehaviortimes-behaviortimes(1);

csflaghires=zeros(size(behaviortimes));
csflaghires(find(behaviortimes>=thisdata.timings.cs_start_time))=1;
csflaghires(find(behaviortimes>=thisdata.timings.cs_end_time))=0;

usflaghires=zeros(size(behaviortimes));
testa=thisdata.timings.us_start_time;
if (~isempty(testa))
usflaghires(find(behaviortimes>=thisdata.timings.us_start_time))=1;
usflaghires(find(behaviortimes>=thisdata.timings.us_end_time))=0;
end

csflaglores=zeros(size(times));
csflaglores(find(times>=thisdata.timings.cs_start_time))=1;
csflaglores(find(times>=thisdata.timings.cs_end_time))=0;

usflaglores=zeros(size(times));
testa=thisdata.timings.us_start_time;
if (~isempty(testa))
    trialhasus(whichtrial+1)=1;
usflaglores(find(times(1:end-1)>=thisdata.timings.us_start_time))=1;
usflaglores(find(times(1:end-1)>=thisdata.timings.us_end_time)+1)=0;
end

% for whichtrial=1:15
for whichtrial=1:numbertrials-1
    
thisdata=getplanedata(filename,whichplane,whichtrial);

badframes=cat(1,badframes,thisdata.badframes+size(imagedata,3));
goodframes=cat(1,goodframes,thisdata.goodframes+size(imagedata,3));

imagedata=cat(3,imagedata,thisdata.imagedata);


thistimes=thisdata.times;
thisbehaviortimes=thisdata.behaviortimes;
thistailtrace=thisdata.tailtrace;
nextrialtime=thistimes(end)+interframetime;


lastbehaviorframe=find(thisbehaviortimes<nextrialtime,1,'last');
if (~isempty(lastbehaviorframe));
    thisbehaviortimes(lastbehaviorframe:end)=[];
    thistailtrace(lastbehaviorframe:end)=[];
end


times=cat(1,times,thistimes);


nextrialtime=compressedtimes(end)+interframetime;
trialstarttime(whichtrial+1)=nextrialtime;
compressedtimes=cat(1,compressedtimes,thistimes-thistimes(1)+nextrialtime);

nextrialtime=compressedtimes(end)+interframetime;
trialendtime(whichtrial+1)=nextrialtime;

behaviortimes=cat(1,behaviortimes,thisbehaviortimes);
tailtrace=cat(1,tailtrace,thistailtrace);

compressedbehaviortimes=cat(1,compressedbehaviortimes,thisbehaviortimes-thisbehaviortimes(1)+nexbehtrialtime);

nexbehtrialtime=compressedbehaviortimes(end)+behinterframetime;

newcsflaghires=zeros(size(thisbehaviortimes));
newcsflaghires(find(thisbehaviortimes>=thisdata.timings.cs_start_time))=1;
newcsflaghires(find(thisbehaviortimes>=thisdata.timings.cs_end_time))=0;

newusflaghires=zeros(size(thisbehaviortimes));
testa=thisdata.timings.us_start_time;
if (~isempty(testa))
    trialhasus(whichtrial+1)=1;
%     disp('us')
newusflaghires(find(thisbehaviortimes>=thisdata.timings.us_start_time))=1;
newusflaghires(find(thisbehaviortimes>=thisdata.timings.us_end_time))=0;
end
csflaghires=cat(1,csflaghires,newcsflaghires);
usflaghires=cat(1,usflaghires,newusflaghires);

newcsflaglores=zeros(size(thistimes));
newcsflaglores(find(thistimes>=thisdata.timings.cs_start_time))=1;
newcsflaglores(find(thistimes>=thisdata.timings.cs_end_time))=0;

newusflaglores=zeros(size(thistimes));
testa=thisdata.timings.us_start_time;
if (~isempty(testa))
%     disp('us2')
%     disp(thisdata.timings.us_start_time)
%     disp(thisdata.timings.us_end_time)
%     disp(find(thistimes>=thisdata.timings.us_start_time))
%     disp(find(thistimes>=thisdata.timings.us_end_time))
%     trialhasus(whichtrial+1)=1;
newusflaglores(find(thistimes(1:end-1)>=thisdata.timings.us_start_time))=1;
newusflaglores(find(thistimes(1:end-1)>=thisdata.timings.us_end_time)+1)=0;
% plot(newusflaglores)
% drawnow
% pause
end
csflaglores=cat(1,csflaglores,newcsflaglores);
usflaglores=cat(1,usflaglores,newusflaglores);

    
end

maskgoodframes=zeros(size(imagedata,3));
maskgoodframes(goodframes)=1;

% 
alldata.maskgoodframes=maskgoodframes;
alldata.imagedata=imagedata;
alldata.badframes=badframes;
alldata.goodframes=goodframes;
alldata.times=times;
alldata.compressedtimes=compressedtimes;
alldata.behaviortimes=behaviortimes;
alldata.csflaghires=csflaghires;
alldata.usflaghires=usflaghires;
alldata.csflaglores=csflaglores;
alldata.usflaglores=usflaglores;
alldata.trialstarttime=trialstarttime;
alldata.trialendtime=trialendtime;
alldata.trialhasus=trialhasus;

%construct a CS  US and learning regressor
csRegressor=zeros(size(alldata.times));
csMask=zeros(size(alldata.times));
learnRegressor=zeros(size(alldata.times));

for n=1:16
    startcs=find((alldata.compressedtimes>alldata.trialstarttime(n))&(alldata.csflaglores>0),1,'first')
    if (alldata.trialhasus(n)==1)
    
    endcs=find((alldata.compressedtimes>alldata.trialstarttime(n))&(alldata.usflaglores>0),1,'first')
    csRegressor(startcs:endcs-1)=1;
        endflag=find((alldata.compressedtimes<alldata.trialendtime(n)),1,'last');
    csMask(endcs:endflag)=1
    learnRegressor(startcs:endcs-1)=1; 
    else
    endcs=find((alldata.compressedtimes<alldata.trialendtime(n))&(alldata.csflaglores>0),1,'last')
    csRegressor(startcs:endcs)=1;
    end
    
end

alldata.csRegressor=csRegressor;
alldata.csMask=csMask;
alldata.learnRegressor=learnRegressor;
if (savedata)
    disp('saving')
    save(fullsavepath,'alldata','-v7.3')
    
end
end




function [corrmap]=calculateCorrelationMapSinglePlane(alldata,gausswidth,deTrendScl, howmanyframes,myborder)
%  gausswidth=2;
    %detrend filter window size
%     deTrendScl=55;
    %right now i just used the first few frames to avoid memory crash. i
    %will make roi traces with all data
%     howmanyframes=200;
    allframes=size(alldata.imagedata,3);
    if (howmanyframes==0)
    howmanyframes=allframes;
    end

h=size(alldata.imagedata,1);
w=size(alldata.imagedata,2);
planeData4ROIs=alldata.imagedata(:,:,find(alldata.maskgoodframes));
zz1=size(planeData4ROIs,3);
planeData=alldata.imagedata(:,:,find(alldata.maskgoodframes(1:howmanyframes)));
zzc=size(planeData,3);

FplaneData=zeros(h,w,zzc,'single');

for n=1:zzc
    FplaneData(:,:,n)=imgaussfilt(planeData(:,:,n),gausswidth);
end 

% planeData4ROIs=planeData;

planeDataM=mean(planeData,3);
FplaneDataM=mean(FplaneData,3);

% planeDataM=mean(planeData4ROIs,3);
% FplaneDataM=mean(FplaneData,3);

   myfilter=ones(deTrendScl,1)*(-1/deTrendScl);
   myfilter(ceil(deTrendScl/2))= myfilter(ceil(deTrendScl/2))+1;
 disp('here')

for n=1:size(planeData,3)

planeData(:,:,n)=planeData(:,:,n)-planeDataM;
FplaneData(:,:,n)=FplaneData(:,:,n)-FplaneDataM;

end
 disp('here 2')

for n=1:size(planeData,1)

planeData(n,:,:)=conv2(1,myfilter,squeeze(planeData(n,:,:)),'same');
FplaneData(n,:,:)=conv2(1,myfilter,squeeze(FplaneData(n,:,:)),'same');

end
     indnorm=vecnorm(planeData,2,3);
    meannorm=vecnorm(FplaneData,2,3);
    meannormsq=power(meannorm,2);
    findnorm=imgaussfilt(indnorm,gausswidth);
    findnormsq=power(findnorm,2);
    corrmap=meannormsq./findnormsq;

    %subtract the baseline correlation that is due to smearing of the PMT
    %signal or other noise
% blankarea=corrmap(bgrect(1):bgrect(1)+bgrect(3),bgrect(2):bgrect(2)+bgrect(4));
corrmap=corrmap-median(corrmap(:));

% myborder=10;
% imshow(corrmap*3)
corrmap(1:myborder,:)=0;
corrmap(:,1:myborder)=0;
corrmap(:,end-myborder+1:end)=0;
corrmap(end-myborder+1:end,:)=0;
    end
% figure
% imshow(corrmap*3)
% save(fullfile(savefolder,strcat('corrmap.mat')),'corrmap');



# Here is a first attempt to clean up the imaging analysis code, at least separating functions[4:36 PM]the main code to take a fish and run through and save the analysis - choosing 'savedata' will resave the data in a large form where it is easy to pull out the traces, and takes up space on the disk.

load('L:\For Mike\20241007_03_delay_2p-1_mitfaminusminuselavl3h2bgcamp6f_5dpf\savefolder\rois00000.mat')

allnums=roidata.allnums;
corrmapval=roidata.corrmapval;
avcorrval=roidata.avcorrval;

mytraces=roidata.mytraces;
mytracesraw=roidata.mytracesraw;
allrois=roidata.allrois;

% % clear corrmapval avcorrval allnums


mytracestruegood=mytraces(:,allnums>50);
D = pdist(mytracestruegood');
tree = linkage(D,'average');
leafOrder = optimalleaforder(tree,D);


figure
for n=1:length(leafOrder)/10
hold on
plot(mean(zscore(mytracestruegood(:,leafOrder((n-1)*10+1:n*10)))')+n*3)
end

% tracesforkmeans=zscore(mytracestruegood);
% for n=1:size(tracesforkmeans,2);
%     tracesforkmeans(:,n)=conv(tracesforkmeans(:,n),myfilter,'same');
% end
% nclusters=80;
% IDX = kmeans(tracesforkmeans', nclusters);
% 
% for n=1:nclusters
% plot(mean(aa(:,IDX==n),2)+n*3)
% hold on
% end
% [colz]=distinguishable_colors(nclusters);
% 
% goodindices=find(allnums>50);
% colimager=zeros(292,500);
% colimageg=zeros(292,500);
% colimageb=zeros(292,500);
% for n=1:nclusters
%    thiscoords=find(ismember(allrois,goodindices(IDX==n))); 
%    
% % colimager(thiscoords)=colz(n,1);
% % colimageg(thiscoords)=colz(n,2);
% % colimageb(thiscoords)=colz(n,3);
% colimager(thiscoords)=1;
% colimageg(thiscoords)=1;
% colimageb(thiscoords)=1;
% 
% end
% figure
% imshow(cat(3,colimager,colimageg,colimageb))
% 
% for n=1:nclusters
%     for nn=1:nclusters
%    thiscoords=find(ismember(allrois,goodindices(IDX==nn))); 
%    
%     % colimager(thiscoords)=colz(n,1);
%     %    colimageg(thiscoords)=colz(n,2);
%     % colimageb(thiscoords)=colz(n,3);
%     colimager(thiscoords)=1;
%     colimageg(thiscoords)=1;
%     colimageb(thiscoords)=1;
% 
%     end
%    thiscoords=find(ismember(allrois,goodindices(IDX==n))); 
%    
% % colimager(thiscoords)=colz(n,1);
% % colimageg(thiscoords)=colz(n,2);
% % colimageb(thiscoords)=colz(n,3);
% colimager(thiscoords)=1;
% colimageg(thiscoords)=0;
% colimageb(thiscoords)=0;
% subplot(2,1,1)
% imshow(cat(3,colimager,colimageg,colimageb))
% subplot(2,1,2)
% plot(mean(aa(:,IDX==n),2))
% drawnow
% pause
% end
% 
csRegressor=alldata.csRegressor(find(alldata.maskgoodframes));
% csRegressor=alldata.csRegressor;
csMask=alldata.csMask(find(alldata.maskgoodframes));
csRegressor=csRegressor(find(~csMask));
learnRegressor=alldata.learnRegressor(find(alldata.maskgoodframes));
learnRegressor=learnRegressor(find(~csMask));
for n=1:size(mytracestruegood,2)
    traceforcorrelation=mytracestruegood(:,n);
    maskedtrace=traceforcorrelation(find(~csMask));
   ccorr(n)=corr(maskedtrace,csRegressor);
   ucorr(n)=corr(traceforcorrelation,csMask);
   lcorr(n)=corr(maskedtrace,learnRegressor);
end

figure
subplot(1,3,1)
allcs=find(ccorr>0.15);
for n=1:length(allcs)
plot(mytracestruegood(:,allcs(n))+3*n)
hold on
end

subplot(1,3,2)
allls=find(lcorr>0.15);
for n=1:length(allls)
plot(mytracestruegood(:,allls(n))+3*n)
hold on
end

subplot(1,3,3)
allus=find(ucorr>0.3);
for n=1:length(allus)
plot(mytracestruegood(:,allus(n))+3*n)
hold on
end
 %plot some arbitrary stuff
figure
learningcells=find((lcorr>0.1)&(lcorr-ccorr>0.05));
plot(csRegressor*3)
hold on
plot(mean(mytracestruegood(find(~csMask),learningcells),2))


figure
uscells=find((ucorr>0.1));
plot(alldata.csRegressor(find(alldata.maskgoodframes))*3)
hold on
plot(mean(mytracestruegood(:,uscells),2))

figure
cscells=find((ccorr>0.1)&(lcorr-ccorr<0));
plot(csRegressor*3)
hold on
plot(mean(mytracestruegood(find(~csMask),cscells),2))




# This is just a placeholder of examples for looking at roi traces.


load('L:\For Mike\20241007_03_delay_2p-1_mitfaminusminuselavl3h2bgcamp6f_5dpf\savefolder\rois00000.mat')

allnums=roidata.allnums;
corrmapval=roidata.corrmapval;
avcorrval=roidata.avcorrval;

mytraces=roidata.mytraces;
mytracesraw=roidata.mytracesraw;
allrois=roidata.allrois;

% % clear corrmapval avcorrval allnums


mytracestruegood=mytraces(:,allnums>50);
D = pdist(mytracestruegood');
tree = linkage(D,'average');
leafOrder = optimalleaforder(tree,D);


figure
for n=1:length(leafOrder)/10
hold on
plot(mean(zscore(mytracestruegood(:,leafOrder((n-1)*10+1:n*10)))')+n*3)
end

% tracesforkmeans=zscore(mytracestruegood);
% for n=1:size(tracesforkmeans,2);
%     tracesforkmeans(:,n)=conv(tracesforkmeans(:,n),myfilter,'same');
% end
% nclusters=80;
% IDX = kmeans(tracesforkmeans', nclusters);
% 
% for n=1:nclusters
% plot(mean(aa(:,IDX==n),2)+n*3)
% hold on
% end
% [colz]=distinguishable_colors(nclusters);
% 
% goodindices=find(allnums>50);
% colimager=zeros(292,500);
% colimageg=zeros(292,500);
% colimageb=zeros(292,500);
% for n=1:nclusters
%    thiscoords=find(ismember(allrois,goodindices(IDX==n))); 
%    
% % colimager(thiscoords)=colz(n,1);
% % colimageg(thiscoords)=colz(n,2);
% % colimageb(thiscoords)=colz(n,3);
% colimager(thiscoords)=1;
% colimageg(thiscoords)=1;
% colimageb(thiscoords)=1;
% 
% end
% figure
% imshow(cat(3,colimager,colimageg,colimageb))
% 
% for n=1:nclusters
%     for nn=1:nclusters
%    thiscoords=find(ismember(allrois,goodindices(IDX==nn))); 
%    
%     % colimager(thiscoords)=colz(n,1);
%     %    colimageg(thiscoords)=colz(n,2);
%     % colimageb(thiscoords)=colz(n,3);
%     colimager(thiscoords)=1;
%     colimageg(thiscoords)=1;
%     colimageb(thiscoords)=1;
% 
%     end
%    thiscoords=find(ismember(allrois,goodindices(IDX==n))); 
%    
% % colimager(thiscoords)=colz(n,1);
% % colimageg(thiscoords)=colz(n,2);
% % colimageb(thiscoords)=colz(n,3);
% colimager(thiscoords)=1;
% colimageg(thiscoords)=0;
% colimageb(thiscoords)=0;
% subplot(2,1,1)
% imshow(cat(3,colimager,colimageg,colimageb))
% subplot(2,1,2)
% plot(mean(aa(:,IDX==n),2))
% drawnow
% pause
% end
% 
csRegressor=alldata.csRegressor(find(alldata.maskgoodframes));
% csRegressor=alldata.csRegressor;
csMask=alldata.csMask(find(alldata.maskgoodframes));
csRegressor=csRegressor(find(~csMask));
learnRegressor=alldata.learnRegressor(find(alldata.maskgoodframes));
learnRegressor=learnRegressor(find(~csMask));
for n=1:size(mytracestruegood,2)
    traceforcorrelation=mytracestruegood(:,n);
    maskedtrace=traceforcorrelation(find(~csMask));
   ccorr(n)=corr(maskedtrace,csRegressor);
   ucorr(n)=corr(traceforcorrelation,csMask);
   lcorr(n)=corr(maskedtrace,learnRegressor);
end

figure
subplot(1,3,1)
allcs=find(ccorr>0.15);
for n=1:length(allcs)
plot(mytracestruegood(:,allcs(n))+3*n)
hold on
end

subplot(1,3,2)
allls=find(lcorr>0.15);
for n=1:length(allls)
plot(mytracestruegood(:,allls(n))+3*n)
hold on
end

subplot(1,3,3)
allus=find(ucorr>0.3);
for n=1:length(allus)
plot(mytracestruegood(:,allus(n))+3*n)
hold on
end
 %plot some arbitrary stuff
figure
learningcells=find((lcorr>0.1)&(lcorr-ccorr>0.05));
plot(csRegressor*3)
hold on
plot(mean(mytracestruegood(find(~csMask),learningcells),2))


figure
uscells=find((ucorr>0.1));
plot(alldata.csRegressor(find(alldata.maskgoodframes))*3)
hold on
plot(mean(mytracestruegood(:,uscells),2))

figure
cscells=find((ccorr>0.1)&(lcorr-ccorr<0));
plot(csRegressor*3)
hold on
plot(mean(mytracestruegood(find(~csMask),cscells),2))