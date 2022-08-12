clc
clear
close all
% Convertion and data modeling from raw input voltages to calibrated impedance magnitude and phase. 
% 
% By Karim Bouzid, MSc student of Benoit Gosselin and Sandro Carrara. 

% IMPORTANT: 2.7V and 2A max for the Source. 
% DONT TOUCH THE GODDAMN CONSTANTS
% Saturation for the initial spike of the current response of the saline solutions at high salt concentrations. 
%% Load data
rawData = readmatrix('lastData.csv');
%rawData = readmatrix('T3_G_Nothing.csv'); LATEX_Multi-47k_nothing, 
%rawData = readmatrix('LATEX_Multi-47k_SalineWater_1_7247%_180um.csv');  
%rawData = readmatrix('LATEX_Multi-47k_SalineWater_15_3383%_180um.csv');  

freq = rawData(1:end,1)';
real1 = rawData(1:end,2)';
real2 = rawData(1:end,3)';
imaginary1 = rawData(1:end,4)';
imaginary2 = rawData(1:end,5)';
voltage5V = rawData(1:end,6)';

idx = freq<4e7 & freq>1e4 & voltage5V<3000 & real1<5000 & real2<5000 & imaginary1<5000 & imaginary2<5000;% & imaginary2>2100; 
idx([1939,6416,10480,14950,19300])=0;
freq = freq(idx); real1 = real1(idx); real2 = real2(idx);
imaginary1 = imaginary1(idx); imaginary2 = imaginary2(idx); voltage5V = voltage5V(idx);

%% Constant
Vdd = 1.983;
SamplingRate = 650;
t = (0:length(freq)-1)./SamplingRate;

%% Plot Raw Data
figure   
subplot(3,2,1); plot(t,real1,'--r*')
xlabel('Time (s)'); ylabel('Digital [0,4095]')
title('Sampled Real voltage Z1'); grid on; grid minor;
subplot(3,2,2); plot(t,real2,'--r*')
xlabel('Time (s)'); ylabel('Digital [0,4095]')
title('Sampled Real signal'); grid on; grid minor;
subplot(3,2,3); plot(t,imaginary1,'--b*')
xlabel('Time (s)'); ylabel('Digital [0,4095]')
title('Sampled Imaginary signal'); grid on; grid minor;
subplot(3,2,4); plot(t,imaginary2,'--b*')
xlabel('Time (s)'); ylabel('Digital [0,4095]')
title('Sampled Imaginary signal'); grid on; grid minor;
subplot(3,2,[5,6]); plot(t,voltage5V,'--k*')
xlabel('Time (s)'); ylabel('Digital [0,4095]')
title('Sampled 5V'); grid on; grid minor;

%%
measGain1 = 1000/(1000+1000);
measGain2 = 1000/(1000+1000);
V5 = voltage5V*Vdd/4096*4.0;% *(4.01)
%V5 = 4.68*ones(size(voltage5V));
r1 = real1*Vdd/4096/measGain1 - V5/2; %*(0.99*)
r2 = real2*Vdd/4096/measGain2 - V5/2; %*(0.991)
ima1 = imaginary1*Vdd/4096/measGain1 - V5/2; %*(0.9875)
ima2 = imaginary2*Vdd/4096/measGain2 - V5/2; %*(0.9884)

%% Real and imaginary value of Z1 and Z2
figure   
subplot(3,2,1); semilogx(freq,r1,'--r*')
xlabel('Sample'); ylabel('Voltage (V)')
title('Real voltage Z1'); grid on; grid minor;
subplot(3,2,2); semilogx(freq,r2,'--r*')
xlabel('Sample'); ylabel('Voltage (V)')
title('Real voltage Z2'); grid on; grid minor;
subplot(3,2,3); semilogx(freq,ima1,'--b*')
xlabel('Sample'); ylabel('Voltage (V)')
title('Imaginary voltage Z1'); grid on; grid minor;
subplot(3,2,4); semilogx(freq,ima2,'--b*')
xlabel('Sample'); ylabel('Voltage (V)')
title('Imaginary voltage Z2'); grid on; grid minor;
subplot(3,2,[5,6]); semilogx(freq,V5,'--k*')
xlabel('Sample'); ylabel('Voltage (V)')
title('5V'); grid on; grid minor;

%% Calculate Impedance (with Rg=10k, Gbuf=2/300 and Gain=30/3.805, VsampleMin = 0.645 = 3500ohm)
%Rg now is 20k//1pF, 47k//0.5pF, 100k//0.2pF and 227k//0.1pF
Rg = 47e3; Cg = 0.3e-12;
ZRgRatio = (Rg./(1+1j*Rg*Cg*2*pi*freq))./Rg; %Rg in parallel with Cg
Gbuf = 1/8;
Gain = 1;
GainRatio = ones(size(freq));
Vin = V5/1.3;  %Too simple
%Vin = 3.83*ones(size(freq)); Vin(freq>=257000) = 3.87; Vin(freq>=955000) = 3.98; 
%Vin(freq>=1240000) = 4.08; Vin(freq>=4610000) = 4.20; Vin(freq>=7750000) = 4.10;
%Vin(freq>=22000000) = 4; Vin(freq>=33000000) = 3; Vin = Vin/1.07;
%Vin = smooth(Vin,length(r1)/5)';
Vin2 = 1*Vin;

newI1 = sqrt((r1.^2)+(ima1.^2));
yImp1 = Rg.*Vin*Gbuf.*Gain/2./newI1;
impP1 = 180/pi*atan2(-ima1,r1);
newI2 = sqrt((r2.^2)+(ima2.^2));
yImp2 = Rg.*Vin2*Gbuf.*Gain/2./newI2;
impP2 = 180/pi*atan2(-ima2,r2);

%% Means and statistical operations
min_outlier = 3;        %Minimum number of points allowed at a frequency to consider it valid to include in the spectroscopy.
[FREQ_U, yImp1_U, yImp2_U, impP1_U, impP2_U, yImp1_STD, yImp2_STD, impP1_STD, impP2_STD, r1_U, r2_U, ima1_U, ima2_U, ZRgRatio_U, GainRatio_U, Vin_U, Vin2_U] = ...
    Convert2Spectroscopy(freq, yImp1, yImp2, impP1, impP2, r1, r2, ima1, ima2, ZRgRatio, GainRatio, Vin, min_outlier);

%% Calibration
%Specify the folder where the files live.
myFolder = 'C:\Users\karim\Dropbox\Maitrise\Python\Serial_Sampling_IFC\Calibration';  %PC
%myFolder = 'C:\Users\Client\Dropbox\Maitrise\Python\Serial_Sampling_IFC\Calibration'; %Laptop

% Get a list of all files in the folder with the desired file name pattern.
pattern = "CalibratioL*"+num2str(Rg/1000)+"k*.csv"; % Change to whatever pattern you need.
filePattern = fullfile(myFolder, pattern); 
theFiles = dir(filePattern);
CalibrationData = zeros(300,length(theFiles));
y1Calibrated = 0; y2Calibrated = 0; P1Calibrated = 0; P2Calibrated = 0;
y1C = 0; y2C = 0; P1C = 0; P2C = 0;
for k = 1 : length(theFiles)
    fprintf(1, 'Now reading %s\n', theFiles(k).name);
    CalibrationData = readmatrix(theFiles(k).name);    
    Rcalib = str2double(theFiles(k).name(13:17));

    V5_C = CalibrationData(:,6)'*4.00*Vdd/4096;
    r1_C = CalibrationData(:,2)'*Vdd/4096/measGain1 - V5_C/2;
    r2_C = CalibrationData(:,3)'*Vdd/4096/measGain2 - V5_C/2;
    ima1_C = CalibrationData(:,4)'*Vdd/4096/measGain1 - V5_C/2;
    ima2_C = CalibrationData(:,5)'*Vdd/4096/measGain2 - V5_C/2;
    ZRgRatio_C = (Rg./(1+1j*Rg*Cg*2*pi*CalibrationData(:,1)'))./Rg; %Rg in parallel with Cg
    GainRatio_C = ones(size(CalibrationData(:,1)'));
    Vin_C = V5_C/1.3;  %Too simple
    Vin2_C = 1*Vin_C;
    
    yImp1_C = Rg.*Vin_C*Gbuf.*Gain/2./(sqrt((r1_C.^2)+(ima1_C.^2)));
    impP1_C = 180/pi*atan2(-ima1_C,r1_C);
    yImp2_C = Rg.*Vin2_C*Gbuf.*Gain/2./(sqrt((r2_C.^2)+(ima2_C.^2)));
    impP2_C = 180/pi*atan2(-ima2_C,r2_C);
    
    [FREQ_C, yImp1_C, yImp2_C, impP1_C, impP2_C, yImp1_STD_C, yImp2_STD_C, impP1_STD_C, impP2_STD_C, ~, ~, ~, ~, ZRgRatio_C, GainRatio_C, ~, ~] = ...
    Convert2Spectroscopy(CalibrationData(:,1)', yImp1_C, yImp2_C, impP1_C, impP2_C, CalibrationData(:,2)', CalibrationData(:,3)', CalibrationData(:,4)',...
    CalibrationData(:,5)', ZRgRatio_C, GainRatio_C, Vin_C, min_outlier);
    
    y1Calibrated = y1Calibrated + interp1(FREQ_C,yImp1_C.*abs(ZRgRatio_C).*abs(GainRatio_C),FREQ_U)/length(theFiles)/Rcalib;
    y2Calibrated = y2Calibrated + interp1(FREQ_C,yImp2_C.*abs(ZRgRatio_C).*abs(GainRatio_C),FREQ_U)/length(theFiles)/Rcalib;
    P1Calibrated = P1Calibrated + interp1(FREQ_C,impP1_C,FREQ_U)/length(theFiles);
    P2Calibrated = P2Calibrated + interp1(FREQ_C,impP2_C,FREQ_U)/length(theFiles);
    
    y1C = y1C + interp1(FREQ_C,yImp1_C.*abs(ZRgRatio_C).*abs(GainRatio_C),freq)/length(theFiles)/Rcalib;
    y2C = y2C + interp1(FREQ_C,yImp2_C.*abs(ZRgRatio_C).*abs(GainRatio_C),freq)/length(theFiles)/Rcalib;
    P1C = P1C + interp1(FREQ_C,impP1_C,freq)/length(theFiles);
    P2C = P2C + interp1(FREQ_C,impP2_C,freq)/length(theFiles);
end

% y1Calibrated = ones(size(y1Calibrated));
% y2Calibrated = ones(size(y2Calibrated));
% P1Calibrated = zeros(size(P1Calibrated));
% P2Calibrated = zeros(size(P1Calibrated));

%% Plot Z1 and Z2 Magnitude and Phase
y1Theo = Rcalib*ones(size(freq));
y2Theo = abs(10e3 + 1./((1/4.7e3)+(1j*2*pi*freq*100e-12)));
y2Theo = 10e3*ones(size(freq));
P1Theo = zeros(size(freq));
P2Theo = 180/pi*phase(10e3 + 1./((1/4.7e3)+(1j*2*pi*freq*100e-12)));
P2Theo = zeros(size(freq));

figure
subplot(2,2,1); semilogx(freq,yImp1,'r.');
hold on; semilogx(freq,yImp1.*abs(ZRgRatio).*abs(GainRatio)./y1C,'r.','Color', [0.5, 0, 0]);
 hold off
xlabel('Frequency (Hz)'); ylabel('Magnitude (ohm)');
title('Impedance spectroscopy of Electrode 1');
legend(['Original'], ['Fully Calibrated']); grid on; grid minor;
subplot(2,2,3); semilogx(freq,impP1,'b.','Color',[0,0.4,0.7]);
hold on; semilogx(freq,impP1-P1C,'b.','Color', [0, 0, 0.3]);
hold off; xlabel('Frequency (Hz)'); ylabel('Phase (degree)');
title('Impedance spectroscopy');
legend(['Original'], ['Fully Calibrated']); grid on; grid minor;
subplot(2,2,2); semilogx(freq,yImp2,'r.');
hold on; semilogx(freq,yImp2.*abs(ZRgRatio).*abs(GainRatio)./y2C,'r.','Color', [0.5, 0, 0]);
hold off
xlabel('Frequency (Hz)'); ylabel('Magnitude (ohm)');
title('Impedance spectroscopy');
legend(['Original'], ['Fully Calibrated']); grid on; grid minor;
subplot(2,2,4); semilogx(freq,impP2,'b.','Color',[0,0.4,0.9]);
hold on; semilogx(freq,impP2 - P2C,'b.','Color', [0, 0, 0.3]);
hold off
xlabel('Frequency (Hz)'); ylabel('Phase (degree)');
title('Impedance spectroscopy');
legend(['Original'], ['Fully Calibrated']); grid on; grid minor;

%% Erreur relative de mesure
y1Err = ((yImp1.*abs(ZRgRatio).*abs(GainRatio)./y1C)-y1Theo)./y1Theo*100;
y2Err = ((yImp2.*abs(ZRgRatio).*abs(GainRatio)./y2C)-y2Theo)./y2Theo*100;

[FREQ_ERR, yImp1_ERR, yImp2_ERR, impP1_ERR, impP2_ERR, yImp1_ERR_STD, yImp2_ERR_STD, impP1_ERR_STD, impP2_ERR_STD, ~, ~, ~, ~, ZRgRatio_ERR, GainRatio_ERR, ~, ~] = ...
Convert2Spectroscopy(freq, y1Err, y2Err, impP1, impP2, r1, r2, ima1, ima2, ZRgRatio, GainRatio, Vin, min_outlier);

%% Current, 5V and Vin
figure
subplot(2,1,1); semilogx(freq, 2*newI1./Rg./Gain*1e6./abs(GainRatio)./abs(ZRgRatio), '--m*')
hold on; semilogx(freq,2*newI2./Rg./Gain*1e6./abs(GainRatio)./abs(ZRgRatio), '--c*')
hold off; title("Currents in the electrodes"); grid on; grid minor
xlabel('Sample'); ylabel('Current (uA)'); legend(['Z1'],['Z2'])
subplot(2,1,2); semilogx(freq,V5,'--b*')
hold on; semilogx(freq,Vin,'--r*')
hold off; title("Vin and 5V estimation"); grid on; grid minor
xlabel('Sample'); ylabel('Voltage (V)'); legend(['V5'],['Vin'])

%% Calibration data
figure
subplot(2,2,1)
semilogx(FREQ_C, yImp1_C,'r.'); hold on;
errorbar(FREQ_C, yImp1_C, yImp1_STD_C,'Color',[0,0,0]);
title("Calibration data Magnitude Electrode 1"); grid on; grid minor
xlabel('frequency (Hz)'); ylabel('Impedance (Ohm)');
subplot(2,2,2)
semilogx(FREQ_C, yImp2_C,'r.'); hold on;
errorbar(FREQ_C, yImp2_C, yImp2_STD_C,'Color',[0,0,0]);
title("Calibration data Magnitude Electrode 2"); grid on; grid minor
xlabel('frequency (Hz)'); ylabel('Impedance (Ohm)');
subplot(2,2,3)
semilogx(FREQ_C, impP1_C,'r.'); hold on;
errorbar(FREQ_C, impP1_C, impP1_STD_C,'Color',[0,0,0]);
title("Calibration data Phase Electrode 1"); grid on; grid minor
xlabel('frequency (Hz)'); ylabel('Impedance (Ohm)');
subplot(2,2,4)
semilogx(FREQ_C, impP2_C,'r.'); hold on;
errorbar(FREQ_C, impP2_C, impP2_STD_C,'Color',[0,0,0]);
title("Calibration data Phase Electrode 2"); grid on; grid minor
xlabel('frequency (Hz)'); ylabel('Impedance (Ohm)');

%% Square to Sin Spectroscopy Technique 11-12 (Best!)
%Then take only the unique values and median the groups. 
r1_U_calib = yImp1_U.*abs(ZRgRatio_U).*abs(GainRatio_U)./(y1Calibrated').*cos(pi/180*(impP1_U-P1Calibrated'));
r2_U_calib = yImp2_U.*abs(ZRgRatio_U).*abs(GainRatio_U)./(y2Calibrated').*cos(pi/180*(impP2_U-P2Calibrated'));
ima1_U_calib = yImp1_U.*abs(ZRgRatio_U).*abs(GainRatio_U)./(y1Calibrated').*sin(pi/180*(impP1_U-P1Calibrated'));
ima2_U_calib = yImp2_U.*abs(ZRgRatio_U).*abs(GainRatio_U)./(y2Calibrated').*sin(pi/180*(impP2_U-P2Calibrated'));

figure   
subplot(2,2,1);
semilogx(FREQ_U, yImp1_U.*cos(pi/180*impP1_U),'--r*'); hold on;
semilogx(FREQ_U, r1_U_calib,'--r*','Color', [0.6,0,0]); hold off;
xlabel('Frequence (kHz)'); ylabel('Impedance (Ohm)')
title('Resistance of Z1'); grid on; grid minor;
legend(['Original'],['Fully Calibrated']);
subplot(2,2,2);
semilogx(FREQ_U, yImp2_U.*cos(pi/180*impP2_U),'--r*'); hold on;
semilogx(FREQ_U, r2_U_calib,'--r*','Color', [0.6,0,0]); hold off;
xlabel('Frequence (kHz)'); ylabel('Impedance (Ohm)')
title('Resistance of Z2'); grid on; grid minor;
legend(['Original'],['Fully Calibrated']);
subplot(2,2,3);
semilogx(FREQ_U, yImp1_U.*sin(pi/180*impP1_U),'--b*','Color',[0,0.4,0.9]); hold on;
semilogx(FREQ_U, ima1_U_calib,'--b*','Color',[0,0,0.9]); hold off;
xlabel('Frequence (kHz)'); ylabel('Impedance (Ohm)')
title('Reactance of Z1'); grid on; grid minor;
legend(['Original'],['Fully Calibrated']);
subplot(2,2,4);
semilogx(FREQ_U, yImp2_U.*sin(pi/180*impP2_U),'--b*','Color',[0,0.4,0.9]); hold on;
semilogx(FREQ_U, ima2_U_calib,'--b*','Color',[0,0,0.9]); hold off;
xlabel('Frequence (kHz)'); ylabel('Impedance (Ohm)')
title('Reactance of Z2');grid on; grid minor;
legend(['Original'],['Fully Calibrated']);

A = zeros(length(FREQ_U),length(FREQ_U)); B = zeros(length(FREQ_U),length(FREQ_U));
r1_11 = zeros(size(r1_U)); ima1_11 = zeros(size(ima1_U));
r2_11 = zeros(size(r2_U)); ima2_11 = zeros(size(ima2_U));
iiR = [1 -3 -5 -7 -11 -13 15 -17 -19 21 -23 -29];
iiI = [1 3 -5 7 11 -13 -15 -17 19 -21 23 -29];
for f = 1:length(FREQ_U)
    for ii = 1:length(iiR)
        [~,closestIndex] = min(abs(FREQ_U-abs(iiR(ii))*FREQ_U(f)));
        if closestIndex~=length(FREQ_U)
            r1_11(f) = r1_11(f) + r1_U_calib(closestIndex)*sign(iiR(ii))/(iiR(ii)^2);
            ima1_11(f) = ima1_11(f) + ima1_U_calib(closestIndex)*sign(iiI(ii))/(iiI(ii)^2);
            r2_11(f) = r2_11(f) + r2_U_calib(closestIndex)*sign(iiR(ii))/(iiR(ii)^2);
            ima2_11(f) = ima2_11(f) + ima2_U_calib(closestIndex)*sign(iiI(ii))/(iiI(ii)^2);
            A(f,closestIndex) = sign(iiR(ii))/(iiR(ii)^2);
            B(f,closestIndex) = sign(iiI(ii))/(iiI(ii)^2);
        elseif ii == 1
            r1_11(f) = r1_U_calib(f);
            ima1_11(f) = ima1_U_calib(f);
            r2_11(f) = r2_U_calib(f);
            ima2_11(f) = ima2_U_calib(f);
        end
    end
    A(end,end) = 1;
    B(end,end) = 1;
end
yImp1Square = sum(abs(A),2).*sqrt((r1_11.^2)+(ima1_11.^2));
impP1_11 = 180/pi*atan2(ima1_11,r1_11);
yImp2Square = 1.02*sum(abs(A),2).*sqrt((r2_11.^2)+(ima2_11.^2));
impP2_11 = 180/pi*atan2(ima2_11,r2_11);

%% Plot erreur de mesure
figure
subplot(2,1,1); semilogx(FREQ_ERR,yImp1_ERR,'r--*','Color', [0.5, 0, 0]); hold on;
semilogx(FREQ_U,(yImp1Square'-abs(10e3 + 1./((1/4.7e3)+(1j*2*pi*FREQ_U*100e-12))))./abs(10e3 + 1./((1/4.7e3)+(1j*2*pi*FREQ_U*100e-12)))*100,'--m*'); hold off;
xlabel('Frequency (Hz)'); ylabel('Erreur relative (%)');
title('Relative error with Electrode 1');
legend(['Fully Calibrated', 'Square2Sine']); grid on; grid minor;
subplot(2,1,2); semilogx(FREQ_ERR,yImp2_ERR,'r--*','Color', [0.5, 0, 0]); hold on;
semilogx(FREQ_U,(yImp2Square'-abs(10e3 + 1./((1/4.7e3)+(1j*2*pi*FREQ_U*100e-12))))./abs(10e3 + 1./((1/4.7e3)+(1j*2*pi*FREQ_U*100e-12)))*100,'--m*'); hold off;
xlabel('Frequency (Hz)'); ylabel('Erreur relative (%)');
title('Relative error with Electrode 2');
legend(['Fully Calibrated', 'Square2Sine']); grid on; grid minor;

%% Plot Z1 and Z2
figure
subplot(2,2,1); semilogx(FREQ_U,yImp1_U,'--r*'); hold on;
semilogx(FREQ_U,(yImp1_U.*abs(ZRgRatio_U).*abs(GainRatio_U)./y1Calibrated'),'--r*','Color', [0.3, 0, 0]);
errorbar(FREQ_U, yImp1_U.*abs(ZRgRatio_U).*abs(GainRatio_U)./y1Calibrated', yImp1_STD,'Color',[0,0,0]);
hold off; xlabel('Frequency (Hz)'); ylabel('Impedance (ohm)');
title('Impedance spectroscopy of R=1k37 - Electrode 1');
legend(['Original'],['Fully Calibrated']); grid on; grid minor;
subplot(2,2,3); semilogx(FREQ_U,impP1_U,'--b*','Color',[0,0.4,0.9]); hold on;
semilogx(FREQ_U,impP1_U-P1Calibrated','--b*','Color', [0, 0, 0.3]);
errorbar(FREQ_U, impP1_U-P1Calibrated', impP1_STD,'Color',[0,0,0]);
hold off; xlabel('Frequency (Hz)'); ylabel('Phase (degrees)');
title('Impedance spectroscopy of R=1k37 - Electrode 1');
legend(['Original'],['Fully Calibrated']); grid on; grid minor;
subplot(2,2,2); semilogx(FREQ_U,yImp2_U,'--r*'); hold on
semilogx(FREQ_U,yImp2_U.*abs(ZRgRatio_U).*abs(GainRatio_U)./y2Calibrated','--r*','Color', [0.3, 0, 0]);
errorbar(FREQ_U, yImp2_U.*abs(ZRgRatio_U).*abs(GainRatio_U)./y2Calibrated', yImp2_STD,'Color',[0,0,0]);
hold off; xlabel('Frequency (Hz)'); ylabel('Impedance (ohm)');
title('Impedance spectroscopy of R=1k37 - Electrode 2');
legend(['Original'],['Fully Calibrated']); grid on; grid minor;
subplot(2,2,4); semilogx(FREQ_U,impP2_U,'--b*','Color',[0,0.4,0.9]); hold on;
semilogx(FREQ_U,impP2_U-P2Calibrated','--b*','Color', [0, 0, 0.3]);
errorbar(FREQ_U, impP2_U-P2Calibrated', impP2_STD,'Color',[0,0,0]);
hold off; xlabel('Frequency (Hz)'); ylabel('Phase (degrees)');
title('Impedance spectroscopy of R=1k37 - Electrode 2');
legend(['Original'],['Fully Calibrated']); grid on; grid minor;

%% Send data to Excel
% T = table(FREQ_U', yImp1_U.*abs(ZRgRatio_U).*abs(GainRatio_U)./y1Calibrated', yImp1_STD/2, yImp2_U.*abs(ZRgRatio_U).*abs(GainRatio_U)./y2Calibrated', yImp2_STD/2, impP1_U-P1Calibrated', impP1_STD/2, impP2_U-P2Calibrated', impP2_STD/2);
% T.Properties.VariableNames = {'Frequency' 'Impedance magnitude of 1th pair of electrodes' 'STD of impedance of 1th pair of electrodes' ...
%     'Impedance magnitude of 2nd pair of electrodes' 'STD of impedance of 2nd pair of electrodes' 'Phase1' 'Phase STD 1' 'Phase 2' 'Phase STD2'};
% T.Properties.VariableUnits = {'Hz' 'Ohm' 'Ohm' 'Ohm' 'Ohm' 'Deg' 'Deg' 'Deg' 'Deg'};
% writetable(T,'ImpedanceWater_PDMS_47k_good.xlsx','Sheet',1,'Range','A1')

%% Square to sine conversion
figure
subplot(2,2,1); semilogx(FREQ_U,yImp1_U,'--r*'); hold on; semilogx(FREQ_U,yImp1Square,'--b*','Color', [0, 0, 0.5]);
semilogx(FREQ_U,(yImp1_U.*abs(ZRgRatio_U).*abs(GainRatio_U)./y1Calibrated'),'--r*','Color', [0.5, 0, 0]);
hold off; xlabel('Frequency (Hz)'); ylabel('Magnitude (ohm)');
title('Impedance spectroscopy');
legend(['Original'],['Square Convertion'],['Fully Calibrated']); grid on; grid minor;
subplot(2,2,3); semilogx(FREQ_U,impP1_U,'--b*','Color',[0,0.4,0.9]); hold on;
semilogx(FREQ_U,impP1_11,'--m*'); semilogx(FREQ_U,impP1_U-P1Calibrated','b*','Color', [0, 0, 0.5]);
hold off; xlabel('Frequency (Hz)'); ylabel('Phase (degrees)');
title('Impedance spectroscopy');
legend(['Original'],['Square Convertion'],['Fully Calibrated']); grid on; grid minor;
subplot(2,2,2); semilogx(FREQ_U,yImp2Square,'--m*'); hold on; 
semilogx(FREQ_U,(yImp2_U.*abs(ZRgRatio_U).*abs(GainRatio_U)./y2Calibrated'),'--r*','Color', [0.5, 0, 0]);
semilogx(FREQ_U,10e3*ones(size(FREQ_U)),'k*')
hold off; xlabel('Frequency (Hz)'); ylabel('Magnitude (ohm)');
title('Impedance spectroscopy');
legend(['Square Convertion'],['Fully Calibrated'], ['Theoretical']); grid on; grid minor;
grid on; grid minor;
subplot(2,2,4); semilogx(FREQ_U,impP2_11,'--b*','Color', [0, 0, 0.5]); hold on;
semilogx(FREQ_U,impP2_U-P2Calibrated','--b*','Color', [0, 0, 0.5]);
semilogx(FREQ_U,zeros(size(FREQ_U)),'k*')
hold off; xlabel('Frequency (Hz)'); ylabel('Phase (degrees)');
title('Impedance spectroscopy');
legend(['Square Convertion'],['Fully Calibrated'], ['Theoretical']); grid on; grid minor;

%% Siemen plot
Conductance1 = 1000*(1./(yImp1_U.*abs(ZRgRatio_U).*abs(GainRatio_U)./y1Calibrated')).*...
    cos(pi/180*impP1_U-pi/180*P1Calibrated');
Conductance2 = 1000*(1./(yImp2_U.*abs(ZRgRatio_U).*abs(GainRatio_U)./y2Calibrated')).*...
    cos(pi/180*impP2_U-pi/180*P2Calibrated');
Susceptance1 = 1000*(1./(yImp1_U.*abs(ZRgRatio_U).*abs(GainRatio_U)./y1Calibrated')).*...
    sin(pi/180*impP1_U-pi/180*P1Calibrated');
Susceptance2 = 1000*(1./(yImp2_U.*abs(ZRgRatio_U).*abs(GainRatio_U)./y2Calibrated')).*...
    sin(pi/180*impP2_U-pi/180*P2Calibrated');

Conductance1_OG = 1000*(1./(yImp1_U)).*cos(pi/180*impP1_U);
Conductance2_OG = 1000*(1./(yImp2_U)).*cos(pi/180*impP2_U);
Susceptance1_OG = 1000*(1./(yImp1_U)).*sin(pi/180*impP1_U);
Susceptance2_OG = 1000*(1./(yImp2_U)).*sin(pi/180*impP2_U);

figure
subplot(3,2,1);
semilogx(FREQ_U,Conductance1,'--r*','Color', [0.3,0,0]); hold on;
semilogx(FREQ_U,Conductance1_OG,'--r*'); hold off;
title('Conductance of Electrode 1'); %legend(['Fully Calibrated'],['Original']);
xlabel('Frequency (Hz)'); ylabel('Conductance (mS)'); grid on; grid minor;
subplot(3,2,2);
semilogx(FREQ_U,Conductance2,'--r*','Color', [0.3,0,0]); hold on;
semilogx(FREQ_U,Conductance2_OG,'--r*'); hold off;
title('Conductance of Electrode 2'); %legend(['Fully Calibrated'],['Original']);
xlabel('Frequency (Hz)'); ylabel('Conductance (mS)'); grid on; grid minor;
subplot(3,2,3);
semilogx(FREQ_U,Susceptance1,'--b*','Color', [0,0,0.5]); hold on;
semilogx(FREQ_U,Susceptance1_OG,'--b*','Color',[0,0.4,0.9]); hold off;
title('Susceptance of Electrode 1'); %legend(['Fully Calibrated'],['Original']);
xlabel('Frequency (Hz)'); ylabel('Susceptance (mS)'); grid on; grid minor;
subplot(3,2,4);
semilogx(FREQ_U,Susceptance2,'--b*','Color', [0,0,0.5]); hold on;
semilogx(FREQ_U,Susceptance2_OG,'--b*','Color',[0,0.4,0.9]); hold off;
title('Susceptance of Electrode 2'); %legend(['Fully Calibrated'],['Original']);
xlabel('Frequency (Hz)'); ylabel('Susceptance (mS)'); grid on; grid minor;
subplot(3,2,5);
plot(Conductance1,Susceptance1,'--g*','Color', [0,0.3,0]); hold on;
plot(Conductance1_OG,Susceptance1_OG,'--g*'); hold off;
title('Conductance VS Susceptance - Electrode 1'); %legend(['Fully Calibrated'],['Original']);
xlabel('Conductance (mS)'); ylabel('Susceptance (mS)'); grid on; grid minor;
subplot(3,2,6);
plot(Conductance2,Susceptance2,'--g*','Color', [0,0.3,0]); hold on;
plot(Conductance2_OG,Susceptance2_OG,'--g*'); hold off;
title('Conductance VS Susceptance - Electrode 2'); %legend(['Fully Calibrated'],['Original']);
xlabel('Conductance (mS)'); ylabel('Susceptance (mS)'); grid on; grid minor;


%% Siemen plot RAW
Resistance1 = ((yImp1_U.*abs(ZRgRatio_U).*abs(GainRatio_U)./y1Calibrated')).*...
    cos(pi/180*impP1_U-pi/180*P1Calibrated');
Resistance2 = ((yImp2_U.*abs(ZRgRatio_U).*abs(GainRatio_U)./y2Calibrated')).*...
    cos(pi/180*impP2_U-pi/180*P2Calibrated');
Reactance1 = ((yImp1_U.*abs(ZRgRatio_U).*abs(GainRatio_U)./y1Calibrated')).*...
    sin(pi/180*impP1_U-pi/180*P1Calibrated');
Reactance2 = ((yImp2_U.*abs(ZRgRatio_U).*abs(GainRatio_U)./y2Calibrated')).*...
    sin(pi/180*impP2_U-pi/180*P2Calibrated');

Resistance1_OG = ((yImp1_U)).*cos(pi/180*impP1_U);
Resistance2_OG = ((yImp2_U)).*cos(pi/180*impP2_U);
Reactance1_OG = ((yImp1_U)).*sin(pi/180*impP1_U);
Reactance2_OG = ((yImp2_U)).*sin(pi/180*impP2_U);

figure
subplot(3,2,1);
semilogx(FREQ_U,Resistance1,'--r*')
title('Resistance Z1');
xlabel('Frequency (Hz)'); ylabel('Resistance (ohm)'); grid on; grid minor;
subplot(3,2,2);
semilogx(FREQ_U,Resistance2,'--r*')
title('Resistance Z2');
xlabel('Frequency (Hz)'); ylabel('Resistance (ohm)'); grid on; grid minor;
subplot(3,2,3);
semilogx(FREQ_U,Reactance1,'--b*')
title('Reactance Z1');
xlabel('Frequency (Hz)'); ylabel('Reactance (ohm)'); grid on; grid minor;
subplot(3,2,4);
semilogx(FREQ_U,Reactance2,'--b*')
title('Reactance Z2');
xlabel('Frequency (Hz)'); ylabel('Reactance (ohm)'); grid on; grid minor;
subplot(3,2,5);
plot(Resistance1,Reactance1,'--k*');
title('Resistance VS Reactance Z1');
xlabel('Resistance (ohm)'); ylabel('Reactance (ohm)'); grid on; grid minor;
subplot(3,2,6);
plot(Resistance2,Reactance2,'--k*');
title('Resistance VS Reactance Z2');
xlabel('Resistance (ohm)'); ylabel('Reactance (ohm)'); grid on; grid minor;