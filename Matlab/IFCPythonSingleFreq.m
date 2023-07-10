clc
clear
close all
%Simulations for the practical conversion from Square to Sinewave
%spectroscopy. By Karim Bouzid, MSc student of Benoit Gosselin and Sandro
%Carrara. 

% IMPORTANT: 2.9V and 2A max for the Source. 
%% Load data
%rawData = readmatrix('lastData.csv');
%rawData = readmatrix('SomeRecognitionTest.csv');
%rawData = readmatrix('T1_47k_saltedwater_beads_TOPROCESS_single.csv'); %57.3s, 70.78, 70.51, 70.4, 71.07
%rawData = readmatrix('T2_47k_saltedwater_beads_TOPROCESS_Best_single.csv'); %T2_47k_best -> 16.638s
%rawData = readmatrix('Correlation_180um_1MHz_7.csv'); %7210 great double peaks shape; 1min20 à 1min29 ->5 beads and a change of flow
%rawData = readmatrix('67um_DATASET7.csv');
rawData = readmatrix('buccal2.csv');

%T9 is pretty good. Classify bubbles and beads with it. Z1 ->middle
%electrode; Z2 ->side electrode. The last ones (6-7-8) are better since I used a
%smaller S&H time. 


counted_raw = rawData(:,1);
real1 = rawData(:,2);
real2 = rawData(:,3);
imaginary1 = rawData(:,4);
imaginary2 = rawData(:,5);
voltage5V = rawData(:,6);

%% Take out outliers and sort
idx = (real1>700 & real1<4100)&(real2>700 & real2<4100)&...
    (imaginary1>1700 & imaginary1<4000)&...
    (imaginary2>1700 & imaginary2<4000)& (voltage5V<2700 & voltage5V>1800);
%idx = 1:length(voltage5V);
counted = unwrap(counted_raw(idx), 10000); real1 = real1(idx); real2 = real2(idx);
imaginary1 = imaginary1(idx); imaginary2 = imaginary2(idx); voltage5V = voltage5V(idx);

[~,ind] = sort(counted); counted = counted(ind);
real1 = real1(ind); real2 = real2(ind);
imaginary1 = imaginary1(ind); imaginary2 = imaginary2(ind); voltage5V = voltage5V(ind);
%% Constants
Vdd = 1.943;
SamplingRate = 5461;
t = counted/SamplingRate;
t = t-t(1);

%% Plot Raw Data
figure   
subplot(3,2,1); plot(t,real1,'--r.')
xlabel('Time (s)'); ylabel('Digital [0,4095]')
title('Sampled Real voltage 1th electrode pair'); grid on; grid minor;
subplot(3,2,2); plot(t,real2,'--r.')
xlabel('Time (s)'); ylabel('Digital [0,4095]')
title('Sampled Real voltage 2nd electrode pair'); grid on; grid minor;
subplot(3,2,3); plot(t,imaginary1,'--b.')
xlabel('Time (s)'); ylabel('Digital [0,4095]')
title('Sampled Imaginary voltage 1th pair of electrodes'); grid on; grid minor;
subplot(3,2,4); plot(t,imaginary2,'--b.')
xlabel('Time (s)'); ylabel('Digital [0,4095]')
title('Sampled Imaginary voltage 2nd pair of electrodes'); grid on; grid minor;
subplot(3,2,[5,6]); plot(t,voltage5V,'--k*')
xlabel('Time (s)'); ylabel('Digital [0,4095]')
title('Sampled 5V'); grid on; grid minor;

%%
measGain1 = 1000/(1000+1000);
measGain2 = 1000/(1000+1000);
V5 = voltage5V*Vdd/4096*(4.01);
%V5 = 4.68*ones(size(voltage5V));
r1 = real1*Vdd/4096/measGain1*(0.99) - V5/2; 
r2 = real2*Vdd/4096/measGain2*(0.991) - V5/2;
ima1 = imaginary1*Vdd/4096/measGain1*(0.9875) - V5/2;
ima2 = imaginary2*Vdd/4096/measGain2*(0.9884) - V5/2;

%% Plot relative Input Data
figure   
subplot(3,2,1); plot(t,r1,'--r*')
xlabel('Time (s)'); ylabel('Voltage (V)')
title('Real voltage Z1'); grid on; grid minor;
subplot(3,2,2); plot(t,r2,'--r*')
xlabel('Time (s)'); ylabel('Voltage (V)')
title('Real voltage Z2'); grid on; grid minor;
subplot(3,2,3); plot(t,ima1,'--b*')
xlabel('Time (s)'); ylabel('Voltage (V)')
title('Imaginary voltage Z1'); grid on; grid minor;
subplot(3,2,4); plot(t,ima2,'--b*')
xlabel('Time (s)'); ylabel('Voltage (V)')
title('Imaginary voltage Z2'); grid on; grid minor;
subplot(3,2,[5,6]); plot(t,V5,'--k*')
xlabel('Time (s)'); ylabel('Voltage (V)')
title('5V'); grid on; grid minor;

%% Calculate Impedance
%Rg now is 20k//1pF, 47k//0.5pF, 100k//0.2pF and 227k//0.1pF
%Rg is now 4.7k//4.7pF, 47k//0.5pF, 150k?//0.2pF and INFINITY
Rg = 47e3; Cg = 0.3e-12;
freq = 200e3.*ones(size(r1));
ZRgRatio = (Rg./(1+1j*Rg*Cg*2*pi*freq))./Rg; %Rg in parallel with Cg
Gbuf = 1/8;
Gain = 1;
GainRatio = ones(size(freq));
Vin = V5/1.28;  %Too simple
%Vin = 3.83*ones(size(freq)); Vin(freq>=257000) = 3.87; Vin(freq>=955000) = 3.98; 
%Vin(freq>=1240000) = 4.08; Vin(freq>=4610000) = 4.20; Vin(freq>=7750000) = 4.10;
%Vin(freq>=22000000) = 4; Vin(freq>=33000000) = 3; Vin = Vin/1.07;
%Vin = smooth(Vin,length(r1)/5)';
Vin2 = 1.0*Vin;

newI1 = sqrt((r1.^2)+(ima1.^2));
yImp1 = Rg.*Vin*Gbuf*Gain/2./newI1;
impP1 = 180/pi*atan2(ima1,r1);
newI2 = sqrt((r2.^2)+(ima2.^2));
yImp2 = Rg.*Vin2*Gbuf*Gain/2./newI2;
impP2 = 180/pi*atan2(ima2,r2);

%% Rescale to the minimum of Z1 or Z2
if mean(yImp1) > mean(yImp2)
    yImp1 = yImp1.*(mean(yImp2)/mean(yImp1));
else
    yImp2 = yImp2.*(mean(yImp1)/mean(yImp2));
end

%% Impedance Amplitude and Phase
figure   
subplot(2,2,1);
plot(t,yImp1.*abs(ZRgRatio),'--b.', 'Color', [0.7,0,0])
xlabel('Time (s)'); ylabel('Magnitude (Ohm)')
title('Impedance magnitude of the 1th electrode pair in time'); grid on; grid minor;
subplot(2,2,2);
plot(t,yImp2.*abs(ZRgRatio),'--b.', 'Color', [0.7,0,0])
xlabel('Time (s)'); ylabel('Magnitude (Ohm)')
title('Impedance magnitude of the 2nd electrode pair in time'); grid on; grid minor;
subplot(2,2,3);
plot(t,impP1-180/pi*angle(ZRgRatio),'--b.', 'Color', [0,0.4,0.9])
xlabel('Time (s)'); ylabel('Phase (Degree)')
title('Impedance phase of the 1th electrode pair in time'); grid on; grid minor;
subplot(2,2,4);
plot(t,impP2-180/pi*angle(ZRgRatio),'--b.', 'Color', [0,0.4,0.9])
xlabel('Time (s)'); ylabel('Phase (Degree)')
title('Impedance phase of the 2nd electrode pair in time'); grid on; grid minor;

%% Preprocessing
yImp1preproc = preprocess(yImp1,SamplingRate, 4.294226749492056);
yImp2preproc = preprocess(yImp2,SamplingRate, 4.294226749492056);
impP1preproc = preprocess(impP1,SamplingRate, 0.2);
impP2preproc = preprocess(impP2,SamplingRate, 0.2);

 %% Record data to Excel
T = table(t, yImp1preproc, yImp2preproc, impP1preproc, impP2preproc);
T.Properties.VariableNames = {'Time' 'Impedance magnitude of 1th pair of electrodes' 'Impedance magnitude of 2nd pair of electrodes'...
    'Phase1' 'Phase 2'};
T.Properties.VariableUnits = {'s' 'Ohm' 'Ohm' 'Deg' 'Deg'};
writetable(T,'47-PreprocessedNonSegmented4.csv')

%% Impedance Amplitude and Phase difference
figure   
subplot(2,1,1);
plot(yImp1preproc-yImp2preproc,'--b*', 'Color', [0.6,0,0])
hold on
%plot(yImp1preproc-yImp2preproc,'--b*', 'Color', [0.9,0,0])
hold off
xlabel('Sample'); ylabel('Magnitude difference (Ohm)')
title('Impedance magnitude difference between the two electrode pairs'); grid on; grid minor;
subplot(2,1,2);
plot(impP1preproc-impP2preproc,'--b*', 'Color', [0.3,0,0.7])
hold on
%plot(impP1preproc-impP2preproc,'--b*', 'Color', [0.6,0,0.9])
hold off
xlabel('Time (s)'); ylabel('Phase (Degrees)')
title('Impedance phase difference in time with Vin at 150kHz'); grid on; grid minor;

figure   
subplot(2,1,1);
plot(t,yImp1preproc-yImp2preproc,'--b*', 'Color', [0.6,0,0])
hold on
%plot(t,yImp1preproc-yImp2preproc,'--b*', 'Color', [0.9,0,0])
hold off
xlabel('Time (s)'); ylabel('Magnitude difference (Ohm)')
title('Impedance magnitude difference between the two electrode pairs'); grid on; grid minor;
subplot(2,1,2);
plot(t,impP1preproc-impP2preproc,'--b*', 'Color', [0.3,0,0.7])
hold on
%plot(t,impP1preproc-impP2preproc,'--b*', 'Color', [0.6,0,0.9])
hold off
xlabel('Time (s)'); ylabel('Phase difference (Degrees)')
title('Impedance phase difference difference between the two electrode pairs'); grid on; grid minor;

%% 5V and Vin
figure
plot(V5)
hold on
plot(Vin)
hold off
title("Vin and 5V estimation")
xlabel('Sample'); ylabel('Voltage (V)')
legend(['V5'],['Vin'])
grid on; grid minor

%% Real and imaginary value of Z1 and Z2
figure   
subplot(2,2,1);
plot(r1,'--b*')
xlabel('Sample')
ylabel('Voltage (V)')
title('Real voltage Z1')
grid on; grid minor;
subplot(2,2,2);
plot(r2,'--b*')
xlabel('Sample')
ylabel('Voltage (V)')
title('Real voltage Z2')
grid on; grid minor;
subplot(2,2,3);
plot(ima1,'--b*')
xlabel('Sample')
ylabel('Voltage (V)')
title('Imaginary voltage Z1')
grid on; grid minor;
subplot(2,2,4);
plot(ima2,'--b*')
xlabel('Sample')
ylabel('Voltage (V)')
title('Imaginary voltage Z2')
grid on; grid minor;

%% Peak finders
[pks1,locs1] = findpeaks(yImp1preproc-yImp2preproc, 'MinPeakProminence',250,'MinPeakHeight',80, 'MinPeakDistance',11);
[pks2,locs2] = findpeaks(yImp2preproc-yImp1preproc, 'MinPeakProminence',250,'MinPeakHeight',80, 'MinPeakDistance',11);

figure
findpeaks(yImp1preproc-yImp2preproc,'MinPeakProminence',250,'MinPeakHeight',80, 'MinPeakDistance',11);
hold on
findpeaks(yImp2preproc-yImp1preproc,'MinPeakProminence',250,'MinPeakHeight',80, 'MinPeakDistance',11);
hold off

%% Segmentation
if length(locs1)<length(locs2)
    smallLocs = locs1; bigLocs = locs2;
    smallPks = pks1;   bigPks = -pks2;
else
    smallLocs = locs2; bigLocs = locs1;
    smallPks = pks2;   bigPks = -pks1;

end

smallLocsNew = zeros(size(smallLocs));
smallPksNew = zeros(size(smallPks));
for k=1:numel(smallLocs)
  [val,idx]=min(abs(bigLocs-smallLocs(k)));
  smallLocsNew(k)=bigLocs(idx);
  smallPksNew(k)=bigPks(idx);
end

figure
plot(smallLocs,smallPks, '.k', 'MarkerSize',15)
hold on
plot(smallLocsNew,smallPksNew, '.k', 'MarkerSize',15)
plot(yImp1preproc-yImp2preproc,'r.-');
hold off
title('Impedance magnitude difference Peak detection')
xlabel('Count'); ylabel('Impedance magnitude difference (Ohm)')

meanLocs = zeros(size(smallLocs));
for k=1:numel(smallLocs)
    meanLocs(k) = mean(smallLocs(k),smallLocsNew(k));
end

counted_segm = cell(size(smallLocs));
yImp1preproc_segm = cell(size(smallLocs));
yImp2preproc_segm = cell(size(smallLocs));
impP1preproc_segm = cell(size(smallLocs));
impP2preproc_segm = cell(size(smallLocs));
for k=2:numel(smallLocs)-1
    try
        counted_segm(k-1) = {(meanLocs(k)-floor((meanLocs(k)-meanLocs(k-1))/2):meanLocs(k)+floor((meanLocs(k+1)-meanLocs(k))/2))'};
        yImp1preproc_segm(k-1) = {yImp1preproc(counted_segm{k-1})};
        yImp2preproc_segm(k-1) = {yImp2preproc(counted_segm{k-1})};
        impP1preproc_segm(k-1) = {impP1preproc(counted_segm{k-1})};
        impP2preproc_segm(k-1) = {impP2preproc(counted_segm{k-1})};
    catch
    end
end

%%
point = 181;
figure
plot(counted_segm{point},yImp1preproc_segm{point},'b.-')
hold on
plot(counted_segm{point},yImp2preproc_segm{point},'r.-')
hold off

%%
point = 174;
figure
plot(counted_segm{point},yImp1preproc_segm{point}-yImp2preproc_segm{point},'k.-')

%% Record data to Excel
G = [counted_segm,yImp1preproc_segm,yImp2preproc_segm,impP1preproc_segm,impP2preproc_segm];
T = cell2table(G);
T.Properties.VariableNames = {'Time' 'Impedance magnitude of 1th pair of electrodes' 'Impedance magnitude of 2nd pair of electrodes'...
    'Phase1' 'Phase 2'};
T.Properties.VariableUnits = {'s' 'Ohm' 'Ohm' 'Deg' 'Deg'};
writetable(T,'SegmentedCounted.csv')
%% FF
% mov_rms = sqrt(movmean((impP1-impP2) .^ 2, 50));
% mov_mean = abs(movmean(impP1-impP2,50));
% FF = mov_rms./mov_mean;
% figure
% plot(FF)
% 
% %% Rate of change
% der = diff(yImp1-yImp2);
% figure; plot(der)
% 
% der = diff(impP1-impP2);
% figure; plot(der)
% 
%% P entropy
% [se,te] = pentropy(yImp1-yImp2,SamplingRate);
% figure; subplot(2,1,1); plot(yImp1-yImp2)
% subplot(2,1,2); plot(te, se)

%% Calibration
% %Specify the folder where the files live.
% %myFolder = 'C:\Users\karim\Dropbox\Maitrise\Python\Serial_Sampling_IFC\Calibration';  %PC
% myFolder = 'C:\Users\Client\Dropbox\Maitrise\Python\Serial_Sampling_IFC\Calibration'; %Laptop
% min_outlier = 3;
% 
% % Get a list of all files in the folder with the desired file name pattern.
% pattern = "*"+num2str(Rg/1000)+"k*.csv"; % Change to whatever pattern you need.
% filePattern = fullfile(myFolder, pattern); 
% theFiles = dir(filePattern);
% CalibrationData = zeros(300,length(theFiles));
% y1Calibrated = 0; y2Calibrated = 0; P1Calibrated = 0; P2Calibrated = 0;
% y1C = 0; y2C = 0; P1C = 0; P2C = 0;
% for k = 1 : length(theFiles)
%     fprintf(1, 'Now reading %s\n', theFiles(k).name);
%     CalibrationData = readmatrix(theFiles(k).name);    
%     Rcalib = str2double(theFiles(k).name(13:17));
% 
%     V5_C = CalibrationData(:,6)'*4*Vdd/4096;     
%     r1_C = CalibrationData(:,2)'*Vdd/4096/measGain1 - V5_C/2;
%     r2_C = CalibrationData(:,3)'*Vdd/4096/measGain2 - V5_C/2;
%     ima1_C = abs(CalibrationData(:,4)'*Vdd/4096/measGain1 - V5_C/2);
%     ima2_C = abs(CalibrationData(:,5)'*Vdd/4096/measGain2 - V5_C/2);
%     ZRgRatio_C = (Rg./(1+1j*Rg*Cg*2*pi*CalibrationData(:,1)'))./Rg; %Rg in parallel with Cg
%     GainRatio_C = ones(size(CalibrationData(:,1)'));
%     Vin_C = V5_C/1.25;  %Too simple
%     Vin2_C = 1*Vin_C;
%     
%     yImp1_C = Rg.*Vin_C*Gbuf.*Gain/2./(sqrt((r1_C.^2)+(ima1_C.^2)));
%     impP1_C = 180/pi*atan2(ima1_C,-r1_C);
%     yImp2_C = Rg.*Vin2_C*Gbuf.*Gain/2./(sqrt((r2_C.^2)+(ima2_C.^2)));
%     impP2_C = 180/pi*atan2(ima2_C,-r2_C);
%     
%     [FREQ_C, yImp1_C, yImp2_C, impP1_C, impP2_C, yImp1_STD_C, yImp2_STD_C, impP1_STD_C, impP2_STD_C, ~, ~, ~, ~, ZRgRatio_C, GainRatio_C, ~, ~] = ...
%     Convert2Spectroscopy(CalibrationData(:,1)', yImp1_C, yImp2_C, impP1_C, impP2_C, CalibrationData(:,2)', CalibrationData(:,3)', CalibrationData(:,4)',...
%     CalibrationData(:,5)', ZRgRatio_C, GainRatio_C, Vin_C, min_outlier);
%     
%     y1Calibrated = y1Calibrated + interp1(FREQ_C,yImp1_C.*abs(ZRgRatio_C).*abs(GainRatio_C),FREQ_U)/length(theFiles)/Rcalib;
%     y2Calibrated = y2Calibrated + interp1(FREQ_C,yImp2_C.*abs(ZRgRatio_C).*abs(GainRatio_C),FREQ_U)/length(theFiles)/Rcalib;
%     P1Calibrated = P1Calibrated + interp1(FREQ_C,impP1_C,FREQ_U)/length(theFiles);
%     P2Calibrated = P2Calibrated + interp1(FREQ_C,impP2_C,FREQ_U)/length(theFiles);
%     
%     y1C = y1C + interp1(FREQ_C,yImp1_C.*abs(ZRgRatio_C).*abs(GainRatio_C),freq)/length(theFiles)/Rcalib;
%     y2C = y2C + interp1(FREQ_C,yImp2_C.*abs(ZRgRatio_C).*abs(GainRatio_C),freq)/length(theFiles)/Rcalib;
%     P1C = P1C + interp1(FREQ_C,impP1_C,freq)/length(theFiles);
%     P2C = P2C + interp1(FREQ_C,impP2_C,freq)/length(theFiles);
% end
% 
% %% Spectrum
% idx = (counted>=0.8e5)&(counted<=1.3e5);
% yImp1_spec = yImp1(idx);
% yImp2_spec = yImp2(idx);
% 
% N = length(yImp1_spec);
% YIMP1 = fftshift(2*fft(yImp1_spec,N)/length(yImp1_spec));
% YIMP2 = fftshift(2*fft(yImp2_spec,N)/length(yImp2_spec));
% YIMP = fftshift(2*fft(yImp2_spec-yImp1_spec,N)/length(yImp1_spec));
% 
% nw = SamplingRate/2*(-1:(2/N):1-(2/N));  %base de fréquence
% figure
% plot(nw,20*log10(abs(YIMP1)),'b')
% hold on
% plot(nw,20*log10(abs(YIMP2)),'r')
% hold off
% 
% figure
% plot(nw,20*log10(abs(YIMP)),'b')
% 
%  %% Record data to Excel
% % T = table(t, yImp1, yImp2, impP1, impP2);
% % T.Properties.VariableNames = {'Time' 'Impedance magnitude of 1th pair of electrodes' 'Impedance magnitude of 2nd pair of electrodes'...
% %     'Phase1' 'Phase 2'};
% % T.Properties.VariableUnits = {'s' 'Ohm' 'Ohm' 'Deg' 'Deg'};
% % writetable(T,'SomeRecognitionEXCEL.xlsx','Sheet',1,'Range','A1')
% 
% %% Statistical operations
% freqstr = strings(1,length(u));
% figure
% for i = 1:length(u)
%     freqstr(i) = sprintf("%.1f kHz",u(i)/1000);
%     histogram(yImp2(freq==u(i)));
%     hold on
%     disp(sum(freq==u(i)))
% end
% hold off
% title('Distribution of the impedance Z2')
% xlabel('Impedance magnitude (ohm)'); ylabel('Sample count');
% legend(freqstr)
% 
% figure
% for i = 1:length(u)
%     freqstr(i) = sprintf("%.1f kHz",u(i)/1000);
%     histogram(impP2(freq==u(i)));
%     hold on
% end
% hold off
% title('Distribution of the impedance Z2')
% xlabel('Impedance phase (degree)'); ylabel('Sample count');
% legend(freqstr)
% 
% u = sort(unique(freq));
% freqstr = strings(1,length(u));
% figure
% for i = 1:length(u)
%     freqstr(i) = sprintf("%.1f kHz",u(i)/1000);
%     histogram(yImp1(freq==u(i)));
%     hold on
%     disp(sum(freq==u(i)))
% end
% hold off
% title('Distribution of the impedance Z1')
% xlabel('Impedance magnitude (ohm)'); ylabel('Sample count');
% legend(freqstr)
% 
% figure
% for i = 1:length(u)
%     freqstr(i) = sprintf("%.1f kHz",u(i)/1000);
%     histogram(impP1(freq==u(i)));
%     hold on
% end
% hold off
% title('Distribution of the impedance Z1')
% xlabel('Impedance phase (degree)'); ylabel('Sample count');
% legend(freqstr)

% %% Unsupervised Classification
% %% k-means
% dataset = [yImp2' impP2'];
% numK = 1;
% 
% opts = statset('Display','final');
% [idx,C] = kmeans(dataset,numK,...
%     'Start','cluster','Replicates',5,'Options',opts);
% 
% figure;
% for i = 1:max(idx)
%     plot(dataset(idx==i,1),dataset(idx==i,2),'.','MarkerSize',12)
%     freqstr(i) = sprintf('Centroid %.0f kHz', mode(freq(idx==i))/1000);
%     hold on
% end
% plot(C(:,1),C(:,2),'kx',...
%      'MarkerSize',15,'LineWidth',3) 
% legend(freqstr, 'Location', 'best')
% xlabel('Impedance magnitude (ohms)')
% ylabel('Impedance phase (degree)')
% title 'Cluster Assignments and Centroids Z2'
% hold off
% 
% %% Supervised learning based on Unsupervised learning
% %% Quadratic discriminant
% x1 = min(dataset(:,1)):(max(dataset(:,1))-min(dataset(:,1)))/40:max(dataset(:,1));
% x2 = min(dataset(:,2)):(max(dataset(:,2))-min(dataset(:,2)))/40:max(dataset(:,2));
% [x1G,x2G] = meshgrid(x1,x2);
% XGrid = [x1G(:),x2G(:)]; % Defines a fine grid on the plot
% 
% mu = zeros(max(idx),2);
% delta = zeros(2,2,max(idx));
% for i=1:max(idx)
%     mu(i,:) = mean(dataset(idx==i,:));
%     delta(:,:,i) = cov(dataset(idx==i,:));
%     for j=1:length(XGrid)
%         pp(i,j) = -1/2*(XGrid(j,:)-mu(i,:))*inv(delta(:,:,i))*(XGrid(j,:)-mu(i,:))';
%     end
% end
% [~,idx2Region] = max(pp);
% 
% % figure;
% % gscatter(XGrid(:,1),XGrid(:,2),idx2Region);
% % hold on;
% % plot(dataset(:,1),dataset(:,2),'k*','MarkerSize',5);
% % title 'Grid Quadratic discriminant Z2';
% % xlabel 'Impedance magnitude (ohm)';
% % ylabel 'Impedance phase (degrees)'; 
% % hold off;
% 
% %% K-Means
% dataset = [yImp1' impP1'];
% 
% opts = statset('Display','final');
% [idx,C] = kmedoids(dataset,numK,...
%     'Replicates',5,'Options',opts);
% 
% figure
% for i = 1:max(idx)
%     plot(dataset(idx==i,1),dataset(idx==i,2),'.','MarkerSize',12)
%     freqstr(i) = sprintf('Centroid %.0f kHz', mode(freq(idx==i))/1000);
%     hold on
% end
% plot(C(:,1),C(:,2),'kx',...
%      'MarkerSize',15,'LineWidth',3) 
% legend(freqstr, 'Location', 'best')
% xlabel('Impedance magnitude (ohms)')
% ylabel('Impedance phase (degree)')
% title 'Cluster Assignments and Centroids Z1'
% hold off
% 
% %% Supervised learning based on Unsupervised learning
% %% Quadratic discriminant
% x1 = min(dataset(:,1)):(max(dataset(:,1))-min(dataset(:,1)))/60:max(dataset(:,1));
% x2 = min(dataset(:,2)):(max(dataset(:,2))-min(dataset(:,2)))/60:max(dataset(:,2));
% [x1G,x2G] = meshgrid(x1,x2);
% XGrid = [x1G(:),x2G(:)]; % Defines a fine grid on the plot
% 
% mu = zeros(max(idx),2);
% delta = zeros(2,2,max(idx));
% for i=1:max(idx)
%     mu(i,:) = mean(dataset(idx==i,:));
%     delta(:,:,i) = cov(dataset(idx==i,:));
%     for j=1:length(XGrid)
%         pp(i,j) = -1/2*(XGrid(j,:)-mu(i,:))*inv(delta(:,:,i))*(XGrid(j,:)-mu(i,:))';
%     end
% end
% [~,idx2Region] = max(pp);
% 
% figure;
% gscatter(XGrid(:,1),XGrid(:,2),idx2Region);
% hold on;
% plot(dataset(:,1),dataset(:,2),'k*','MarkerSize',5);
% title 'Grid Quadratic discriminant Z1';
% xlabel 'Impedance magnitude (ohm)';
% ylabel 'Impedance phase (degrees)'; 
% hold off;
% 
% %% K-Means
% dataset = [(yImp1-yImp2)' (impP1-impP2)'];
% 
% opts = statset('Display','final');
% [idx,C] = kmeans(dataset,numK,...
%     'Replicates',5,'Options',opts);
% 
% figure;
% for i = 1:max(idx)
%     plot(dataset(idx==i,1),dataset(idx==i,2),'.','MarkerSize',12)
%     freqstr(i) = sprintf('Centroid %.0f kHz', mode(freq(idx==i))/1000);
%     hold on
% end
% plot(C(:,1),C(:,2),'kx',...
%      'MarkerSize',15,'LineWidth',3) 
% legend(freqstr, 'Location', 'best')
% xlabel('Impedance magnitude (ohms)')
% ylabel('Impedance phase (degree)')
% title 'Cluster Assignments and Centroids Z difference'
% hold off
% 
% x1 = min(dataset(:,1)):(max(dataset(:,1))-min(dataset(:,1)))/40:max(dataset(:,1));
% x2 = min(dataset(:,2)):(max(dataset(:,2))-min(dataset(:,2)))/40:max(dataset(:,2));
% [x1G,x2G] = meshgrid(x1,x2);
% XGrid = [x1G(:),x2G(:)]; % Defines a fine grid on the plot
% 
% mu = zeros(max(idx),2);
% delta = zeros(2,2,max(idx));
% for i=1:max(idx)
%     mu(i,:) = mean(dataset(idx==i,:));
%     delta(:,:,i) = cov(dataset(idx==i,:));
%     for j=1:length(XGrid)
%         pp(i,j) = -1/2*(XGrid(j,:)-mu(i,:))*inv(delta(:,:,i))*(XGrid(j,:)-mu(i,:))';
%     end
% end
% [~,idx2Region] = max(pp);
% 
% % figure;
% % gscatter(XGrid(:,1),XGrid(:,2),idx2Region);
% % hold on;
% % plot(dataset(:,1),dataset(:,2),'k*','MarkerSize',5);
% % title 'Grid Quadratic discriminant Z1';
% % xlabel 'Impedance magnitude (ohm)';
% % ylabel 'Impedance phase (degrees)'; 
% % hold off;
% 
% %% Opacity
% %figure
% %plot();