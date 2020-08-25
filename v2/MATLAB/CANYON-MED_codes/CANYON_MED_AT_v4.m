function out=CANYON_MED_AT_v4(gtime,lat,lon,pres,temp,psal,doxy)
% function out=CANYON_AT(gtime,lat,lon,pres,temp,psal,doxy)
% 
% Multi-layer perceptron to predict alkalinity / umol kg-1  
%
% Neural network training by Marine Fourrier from work by Raphaëlle Sauzède, LOV; 
% as Matlab function by Marine Fourrier, LOV
%
%
% input:
% gtime - date (UTC) as matlab time (days since 01-Jan-0000)
% lat   - latitude / °N  [-90 90]
% lon   - longitude / °E [-180 180] or [0 360]
% pres  - pressure / dbar
% temp  - in-situ temperature / °C
% psal  - salinity
% doxy  - dissolved oxygen / umol kg-1 (!)
%
% output:
% out   - alkalinity / umol kg-1
%
% check value: 2599.3535 umol kg-1
% for 09-Apr-2014, 35Â° N, 18Â° E, 500 dbar, 13.5 Â°C, 38.6 psu, 160 umol O2 kg-1
%
%
% Marine Fourrier, LOV
% 25.08.2020


% No input checks! Assumes informed use, e.g., same dimensions for all
% inputs, ...

basedir='D:\Documents\Thèse\Docs\science\PAPIER_CANYON_MED\CODES\CANYON-MED\v2\MATLAB\'; % relative or absolute path to CANYON training files

% input preparation
dec_year=decyear(gtime);
lon(lon>180)=lon(lon>180)-360;
% input sequence
%     lat,   lon,    dec_year,    temp,   sal,    oxygen, P 
data=[lat(:) lon(:) dec_year(:) temp(:) psal(:) doxy(:) pres(:)./2e4+1./((1+exp(-pres(:)./300)).^3)];


moy_F=load(strcat(basedir,'NN_AT/weights_biases/moy_AT_F.txt')); 
std_F=load(strcat(basedir,'NN_AT/weights_biases/std_AT_F.txt')); 

ne=7;  % Number of inputs
% NORMALISATION OF THE PARAMETERS
[rx,~]=size(data);
data_norm=(2./3)*(data-(ones(rx,1)*moy_F(1:ne)))./(ones(rx,1)*std_F(1:ne));

%
n_list=5;
AT_outputs_s=zeros(size(data_norm,1),n_list);
% AT_out_test_s=zeros(size(data_norm,1),n_list);
for i=1:n_list
    b1=load(strcat(basedir,'NN_AT\weights_biases\poids_AT_b1_F_',string(i),'.txt'));
    b2=load(strcat(basedir,'NN_AT\weights_biases\poids_AT_b2_F_',string(i),'.txt'));
    b3=load(strcat(basedir,'NN_AT\weights_biases\poids_AT_b3_F_',string(i),'.txt'));
    IW=load(strcat(basedir,'NN_AT\weights_biases\poids_AT_IW_F_',string(i),'.txt'));
    LW1=load(strcat(basedir,'NN_AT\weights_biases\poids_AT_LW1_F_',string(i),'.txt'));
    LW2=load(strcat(basedir,'NN_AT\weights_biases\poids_AT_LW2_F_',string(i),'.txt'));
    
    AT_outputs=(LW2*custom_MF(LW1*custom_MF(IW*data_norm(:,1:end)'+b1)+b2)+b3)';
    AT_outputs=1.5*AT_outputs*std_F(ne+1)+moy_F(ne+1);
    AT_outputs_s(:,i)=AT_outputs;
    
end
clear b1 b2 b3 IW LWI LW2

moy_G=load(strcat(basedir,'NN_AT/weights_biases/moy_AT_G.txt')); 
std_G=load(strcat(basedir,'NN_AT/weights_biases/std_AT_G.txt')); 

ne=7;  % Number of inputs
% NORMALISATION OF THE PARAMETERS
[rx,~]=size(data);
data_norm=(2./3)*(data-(ones(rx,1)*moy_G(1:ne)))./(ones(rx,1)*std_G(1:ne));


for i=1:n_list
    b1=load(strcat(basedir,'NN_AT\weights_biases\poids_AT_b1_G_',string(i),'.txt'));
    b2=load(strcat(basedir,'NN_AT\weights_biases\poids_AT_b2_G_',string(i),'.txt'));
    b3=load(strcat(basedir,'NN_AT\weights_biases\poids_AT_b3_G_',string(i),'.txt'));
    IW=load(strcat(basedir,'NN_AT\weights_biases\poids_AT_IW_G_',string(i),'.txt'));
    LW1=load(strcat(basedir,'NN_AT\weights_biases\poids_AT_LW1_G_',string(i),'.txt'));
    LW2=load(strcat(basedir,'NN_AT\weights_biases\poids_AT_LW2_G_',string(i),'.txt'));
    
    AT_outputs=(LW2*custom_MF(LW1*custom_MF(IW*data_norm(:,1:end)'+b1)+b2)+b3)';
    AT_outputs=1.5*AT_outputs*std_G(ne+1)+moy_G(ne+1);
    AT_outputs_s(:,i+5)=AT_outputs;
    
end

%
mean_nn=mean(AT_outputs_s,2);
std_nn=std(AT_outputs_s,0,2);

for i=1:size(AT_outputs_s,1)
    AT_out(i,1)=mean(AT_outputs_s(i,(AT_outputs_s(i,:)>(mean_nn(i)-std_nn(i))&AT_outputs_s(i,:)<(mean_nn(i)+std_nn(i)))));
end

out=AT_out;

