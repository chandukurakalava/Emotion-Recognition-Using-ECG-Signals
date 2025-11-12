clear all;
close all;
clc;
warning off;



[filename,pathname,ext]=uigetfile('*.txt','*.please select text file');
save filename
%%%Eliminate Baseline Drift

s11=fopen(filename);
s1=fread(s11);
save s1

%s2=smooth(s1,150);
%ECGsmooth=s1-s2;


s2=smooth(s1,150);
ECGsmooth=s1-s2;

comet (ECGsmooth); figure(gcf),title('ECG signal visualisation using comet plot');

% %%%apply Wavelet Transform
% [C,L]=wavedec(ECGsmooth,8,'db4');
% ldaClass = classify(s1,ECGsmooth,s1);
% bad = ~strcmp(ldaClass,ECGsmooth);
% ldaResubErr = sum(bad) / 10;
% [d1,d2,d3,d4,d5,d6,d7,d8]=detcoef(C,L,[1,2,3,4,5,6,7,8]);
% Train= [5.6,5.7,5.5,5.7 5.6;
%     5.7,5.3,5.1,5.0 5.2;
%     10.6,9.9,10.4,10.7 10.2;
%     10.7,9.8,9.9,10 10];
%  Class=[1;1;2;2];
% [thr,sorh,keepapp]=ddencmp('den','wv',ECGsmooth);
% cleanECG=wdencmp('gbl',C,L,'db4',8,thr,sorh,keepapp);
% %%%thresholding1
% max_value=max(cleanECG);
% mean_value=mean(cleanECG);
% threshold=(max_value-mean_value)/2;
% %%%R detection algorithm
% a5=appcoef(C,L,'db4',5);
% C1=[a5;d5;d4;d3];
% L1=[length(a5);length(d5);length(d4);length(d3);length(cleanECG)];
% R% end
% mean_R_detect=5*mean(R_detect_new);
% for q=1:length( R_detect_new)-1
%     if  R_detect_new(q)< mean_R_detect
%         R_detect_new(q)=0;
%     end
% 
% end
% %%%%%%%%%%%%%%%%%%
% d=0;
% for b=1:length( R_detect_detect_signal=waverec(C1,L1,'db4');
% R_detect_squared=R_detect_signal.^2;
% %%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%Beat_Rate_Extraction_Algorithm
% for a=1:length(R_detect_squared)
%     if R_detect_squared(a)>threshold
%         R_detect_new(a)=R_detect_squared(a); 
%     else
%         R_detect_new(a)=0;
%     end
%(new)-1
%         if ( R_detect_new(b)==0) & ( R_detect_new(b+1)~=0)
%         d=d+1;
%         indext(d)= b+1;
%         end
% end
% fs_R_deetect=length(R_detect_new)/20;
% time=indext.*1/fs_R_deetect;
% ind=0;
% for z=1:length(time)-1
%     ind=ind+1;
%     time_diff(ind)=time(z+1)-time(z);
% end
% av_time=mean(time_diff);
% Square_Number=av_time/.2;
% beat_Rate=300/Square_Number;
% high=max(R_detect_new);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%Plot the Orginal Signal and Eliminating Baseline Drift signal
subplot(411);plot(s1);title('Orginal ECG Signal');
subplot(412);plot(s1-s2);title('artifact removed signal in pre-processing phase');
%subplot(413);plot(cleanECG);title('baseline eliminated Signal');
load s1
%%%Eliminate Baseline Drift
signal = sin(2*pi*0.055*(0:1000-1)');
nvar  = 1.0;                  % Noise variance
noise = (signal)*nvar;   % White noise
grid; axis([0 1000 -4 4]);
nfilt  = fir1(31,0.5);             % 31st order Low pass FIR filter
fnoise = filter(nfilt,1,noise);    % Filtering the noise
d  = signal+fnoise;
figure,plot(0:199,d(1:200));
grid; axis([0 200 -4 4]);
title('denoised ECG after applying low pass FIR filter');
[cA,cH,cV,cD] = dwt2(ECGsmooth,'haar');

%%%apply Wavelet Transform
[C,L]=wavedec(ECGsmooth,8,'db4');
ldaClass = classify(cD,ECGsmooth,s1);
bad = ~strcmp(ldaClass,ECGsmooth);

d1=min(L);



%%%%%%%%%%%%%%%%%%55555



[c1,l1]=wavedec(s1,4,'db4');
x1_d1=wrcoef('d',c1,l1,'db5',1);
x1_d2=wrcoef('d',c1,l1,'db5',2);
x1_d3=wrcoef('d',c1,l1,'db5',3);
x1_d4=wrcoef('d',c1,l1,'db5',4);
x1_a4=wrcoef('a',c1,l1,'db5',4);


% d1=max(max(x1_a4));


figure,
subplot(611)
plot(s1)
subplot(612)
plot(x1_a4)
subplot(613)
plot(x1_d4)
subplot(614)
plot(x1_d3)
subplot(615)
plot(x1_d2)
subplot(616)
plot(x1_d1),title('ECG signal after wavelet decomposition');
% M     = 32;                 % Filter order
% lam   = 1;                  % Exponential weighting factor
% delta = 0.1;                % Initial input covariance estimate
% w0    = zeros(M,1);         % Initial tap weight vector
% P0    = (1/delta)*eye(M,M); % Initial setting for the P matrix
% Zi    = zeros(M-1,1);       % FIR filter initial states
% Hadapt = dsp.RLSFilter(M,lam,P0,w0,Zi);
% Hadapt.PersistentMemory = true;
% [y,e]  = filter(Hadapt,noise,d);
% H  = abs(freqz(Hadapt,64));
% H1 = abs(freqz(nfilt,1,64));
% wf = linspace(0,1,64);
% figure,plot(wf,H,wf,H1);
% xlabel('Normalized Frequency  (\times\pi rad/sample)');
% ylabel('Magnitude');
% legend('Adaptive Filter Response','Required Filter Response');
% grid;
% axis([0 1 0 2]);
% figure,plot(0:499,signal(1:500),0:499,e(1:500)); grid;
% axis([0 500 -4 4]);
% title('Original information bearing signal and the error signal');
% legend('Original Signal','Error Signal');
%fs=1;
%IS=1;
%[esTSNR,esHRNR]=WienerNoiseReduction(d,fs,IS);
load s1
signal = s1;
Hs = dsp.SignalSource(signal,'SamplesPerFrame',100,...
    'SignalEndAction','Cyclic repetition');
% figure,
% plot(0:199,signal(1:200));
% grid; axis([0 200 -2 2]);
% 
% title('ECG signal');
nvar  = 1.0;                  % Noise variance
noise = randn(1000,1)*nvar;   % White noise
Hn = dsp.SignalSource(noise,'SamplesPerFrame',100,...
    'SignalEndAction','Cyclic repetition');
% figure,
% plot(0:999,noise);
% title('Noise signal');
% grid; axis([0 1000 -4 4]);
Hd = dsp.FIRFilter('Numerator',fir1(31,0.5));% Low pass FIR filter
M      = 32;                 % Filter order
delta  = 0.1;                % Initial input covariance estimate
P0     = (1/delta)*eye(M,M); % Initial setting for the P matrix
Hadapt = dsp.RLSFilter(M,'InitialInverseCovariance',P0);
Hts = dsp.TimeScope('TimeSpan',1000,'YLimits',[-2,2]);
for k = 1:10
    n = step(Hn); % Noise
    s = step(Hs);
    d = step(Hd,n) + s;
    [y,e]  = step(Hadapt,n,d);
    step(Hts,[s,e]);
end
%rng default;
Fs = 1000;
t = 0:1/Fs:1-1/Fs;
x = cos(2*pi*100*t)+randn(size(t));
N = length(x);
xdft = fft(x);
xdft = xdft(1:N/2+1);
psdx = (1/(Fs*N)).*abs(xdft).^2;
psdx(2:end-1) = 2*psdx(2:end-1);
freq = 0:Fs/length(x):Fs/2;
plot(freq,20*log10(psdx)); grid on;
title('spectrum estimation Using FFT');
xlabel('Frequency (Hz)'); ylabel('Power/Frequency (dB/Hz)');
%figure;
[psdestx,Fxx] = periodogram(x,rectwin(length(x)),length(x),Fs);
%plot(Fxx,10*log10(psdestx)); grid on;
%xlabel('Hz'); ylabel('Power/Frequency (dB/Hz)');
%title(' Power Spectral Density Estimate');
max(psdx'-psdestx)
rng default;
n = 0:999;
x = cos(pi/4*n)+randn(size(t));
N = length(x);
xdft = fft(x);
xdft = xdft(1:N/2+1);
psdx = (1/(2*pi*N)).*abs(xdft).^2;
psdx(2:end-1) = 2*psdx(2:end-1);
freq = 0:(2*pi)/N:pi;
%plot(freq./pi,10*log10(psdx)); grid on;
%title('spectrum estimation Using FFT');
%xlabel('Normalized Frequency');
%ylabel('Power/Frequency (dB/rad/sample)');
%figure;
[psdestx,Fxx] = periodogram(x,rectwin(length(x)),length(x));
%plot(Fxx./pi,10*log10(psdestx));
%xlabel('Hz'); ylabel('Power/Frequency (dB/Hz)');
%title(' Power Spectral Density Estimate');
%xlabel('Normalized Frequency'); grid on;
%ylabel('Power/Frequency (dB/rad/sample)');
max(psdx'-psdestx)

rng default;
n = 0:999;
x = exp(1j*pi/4*n)+1/sqrt(2)*randn(size(n))+1j*1/sqrt(2)*randn(size(n));

N = length(x);
xdft = fft(x);
psdx = (1/(2*pi*N)).*abs(xdft).^2;
freq = 0:(2*pi)/N:2*pi-(2*pi)/N;
%plot(freq./pi,10*log10(psdx)); grid on;
%title('power spectrum calculation  Using FFT');
%xlabel('Normalized Frequency');
%ylabel('Power/Frequency (dB/rad/sample)');
[psdestx,Fxx] = periodogram(x,rectwin(length(x)),length(x),'twosided');
%figure;
%plot(Fxx./pi,10*log10(psdestx));
%grid on; xlabel('Normalized Frequency');
%ylabel('Power/Frequency (dB/rad/sample)');
%title(' Power Spectral Density Estimate');
max(psdx'-psdestx)
inputs = psdestx+signal(1,1);
targets = Fxx;
xnew=inputs;
% Create a Pattern Recognition Network
hiddenLayerSize = 10;
net = patternnet(hiddenLayerSize);
% Set up Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
% Train the Network
[net,tr] = train(net,inputs,targets);
% Test the Network
outputs = net(inputs);
errors = gsubtract(targets,outputs);
performance = perform(net,targets,outputs);
% View the Network
view(net)
y_train=tr;
[inputs,targets] = cancer_dataset;
hiddenLayerSize = 10;
net = patternnet(hiddenLayerSize);
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio   = 15/100;
net.divideParam.testRatio  = 15/100;
[net,tr] = train(net,inputs,targets);
outputs = net(inputs);
errors = gsubtract(targets,outputs);
performance = perform(net,targets,outputs);
tInd = tr.testInd;
tstOutputs = net(inputs(:,tInd));
tstPerform = perform(net,targets(:,tInd),tstOutputs);
view(net)

% if d1==54.3455;
% end
if d1==71;
helpdlg('happy');
end
if d1==79;
helpdlg('disgust');

end

if d1==82;
helpdlg('sad');

end
if d1==60;
    helpdlg('anger');
end
%figure, plotperform(tr)
%pfigure, plotconfusion(targets,outputs)
for n=1:size(net,3)
xnewer = [xnew(1:n-1,:); xnew(n+1:end, :)];
%y_train_newer= [y_train(:,1:n-1) y_train(:,n+1:end)];
%solve 
%c = pinv(xnewer'*xnewer)*(xnewer'*y_train_newer');
%test
testit = xnew(n);

end
% load signal
% pscore=((psdestx(7,1)/n)*100);
% disp('accuracy percentage--');
% k=pscore*signal(1,1);
% disp(k);