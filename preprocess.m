function [preproc] = preprocess(signal,Fs, threshold)
%PREPROCESS Preprocess a function
%   Remove DC, LPF, then smooth

 %  plot(ECG.signal);
    % Base Line Correction
        [C,L]=wavedec(signal,9,'db8');

        cA=appcoef(C,L,'db8',9);
        l=length(cA);
        C(1:l)=0;
        baseLineCorrectedSignal=waverec(C,L,'db8');
        sorh = 's';    % Specified soft or hard thresholding
        thrSettings = threshold;

    % wavelet filter
        waveletBaseLineCorrectedSignal = cmddenoise(baseLineCorrectedSignal,'db8',9,sorh,NaN,thrSettings);
        
    % low pass filter
        hFs=Fs/2;
        Wp=1500/hFs;
        Ws=2000/hFs;
        [Lp_n,Lp_Wn] = buttord(Wp,Ws,0.1,30);
        [b,a] = butter(Lp_n,Lp_Wn);
        % freqz(b,a);
        lowPassBandStopWaveletBaseLineCorrectedSignal=filter(b,a,waveletBaseLineCorrectedSignal);

    % smooth filter
        smoothedLowPassBandStopWaveletBaseLineCorrectedSignal = smooth(lowPassBandStopWaveletBaseLineCorrectedSignal,3);

preproc = smoothedLowPassBandStopWaveletBaseLineCorrectedSignal;
end