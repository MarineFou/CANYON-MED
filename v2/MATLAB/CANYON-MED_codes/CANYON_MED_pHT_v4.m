function out=CANYON_MED_pHT_v4(gtime,lat,lon,pres,temp,psal,doxy)
% function out=CANYON_pHT(gtime,lat,lon,pres,temp,psal,doxy)
% 
% Multi-layer perceptron to predict total pH (total scale at insitu PTS)
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
% out   - pHT (total scale at insitu PTS)
%
% check value: 8.0965
% for 09-Apr-2014, 35° N, 18° E, 500 dbar, 13.5 °C, 38.6 psu, 160 umol O2 kg-1
% for example run
% CANYON_MED_pHT_v4(datenum(2014,04,09),35,18,500,13.5,38.6,160)
%
% Marine Fourrier, LOV
% 25.08.2020

% No input checks! Assumes informed use, e.g., same dimensions for all
% inputs, ...

basedir=strcat(fileparts(fileparts(which('CANYON_MED_pHT_v4.m'))),'\'); % relative or absolute path to CANYON-MED training files

% input preparation
dec_year=decyear(gtime);

% if you don't have the aerospace toolbox, un comment lines 39-51 and
% comment line 35
% Get date components from datetime array
% if isa(gtime,"datetime")
% else
%     gtime= datetime( datenum(gtime), 'convertFrom', 'datenum' );
% end
% yearIn = year(gtime);
% ndays = 365*ones(size(yearIn));
% year_in = floor(yearIn);
% ly = ((mod(year_in,4) == 0 & mod(year_in,100) ~= 0) | mod(year_in,400) == 0);
% ndays(ly) = 366;
% firstDayofYear = datetime(yearIn, ones(size(yearIn)), ones(size(yearIn)));
% dayofyear = gtime - firstDayofYear;
% % Calculate the decimal year
% dec_year = yearIn + dayofyear./hours(ndays*24);

lon(lon>180)=lon(lon>180)-360;
% input sequence
%     lat,   lon,    dec_year,    temp,   sal,    oxygen, P 
data=[lat(:) lon(:) dec_year(:) temp(:) psal(:) doxy(:) pres(:)./2e4+1./((1+exp(-pres(:)./300)).^3)];


moy_F=load(strcat(basedir,'CANYON-MED_weights/moy_ph_F.txt')); 
std_F=load(strcat(basedir,'CANYON-MED_weights/std_ph_F.txt')); 

ne=7;  % Number of inputs
% NORMALISATION OF THE PARAMETERS
[rx,~]=size(data);
data_norm=(2./3)*(data-(ones(rx,1)*moy_F(1:ne)))./(ones(rx,1)*std_F(1:ne));

%
n_list=5;
ph_outputs_s=zeros(size(data_norm,1),n_list);
% ph_out_test_s=zeros(size(data_norm,1),n_list);
for i=1:n_list
    b1=load(strcat(basedir,'CANYON-MED_weights/poids_ph_b1_F_',string(i),'.txt'));
    b2=load(strcat(basedir,'CANYON-MED_weights/poids_ph_b2_F_',string(i),'.txt'));
    b3=load(strcat(basedir,'CANYON-MED_weights/poids_ph_b3_F_',string(i),'.txt'));
    IW=load(strcat(basedir,'CANYON-MED_weights/poids_ph_IW_F_',string(i),'.txt'));
    LW1=load(strcat(basedir,'CANYON-MED_weights/poids_ph_LW1_F_',string(i),'.txt'));
    LW2=load(strcat(basedir,'CANYON-MED_weights/poids_ph_LW2_F_',string(i),'.txt'));
    
    ph_outputs=(LW2*custom_MF(LW1*custom_MF(IW*data_norm(:,1:end)'+b1)+b2)+b3)';
    ph_outputs=1.5*ph_outputs*std_F(ne+1)+moy_F(ne+1);
    ph_outputs_s(:,i)=ph_outputs;
    
end
clear b1 b2 b3 IW LWI LW2

moy_G=load(strcat(basedir,'CANYON-MED_weights/moy_ph_G.txt')); 
std_G=load(strcat(basedir,'CANYON-MED_weights/std_ph_G.txt')); 

ne=7;  % Number of inputs
% NORMALISATION OF THE PARAMETERS
[rx,~]=size(data);
data_norm=(2./3)*(data-(ones(rx,1)*moy_G(1:ne)))./(ones(rx,1)*std_G(1:ne));


for i=1:n_list
    b1=load(strcat(basedir,'CANYON-MED_weights/poids_ph_b1_G_',string(i),'.txt'));
    b2=load(strcat(basedir,'CANYON-MED_weights/poids_ph_b2_G_',string(i),'.txt'));
    b3=load(strcat(basedir,'CANYON-MED_weights/poids_ph_b3_G_',string(i),'.txt'));
    IW=load(strcat(basedir,'CANYON-MED_weights/poids_ph_IW_G_',string(i),'.txt'));
    LW1=load(strcat(basedir,'CANYON-MED_weights/poids_ph_LW1_G_',string(i),'.txt'));
    LW2=load(strcat(basedir,'CANYON-MED_weights/poids_ph_LW2_G_',string(i),'.txt'));
    
    ph_outputs=(LW2*custom_MF(LW1*custom_MF(IW*data_norm(:,1:end)'+b1)+b2)+b3)';
    ph_outputs=1.5*ph_outputs*std_G(ne+1)+moy_G(ne+1);
    ph_outputs_s(:,i+5)=ph_outputs;
    
end

%
mean_nn=mean(ph_outputs_s,2);
std_nn=std(ph_outputs_s,0,2);

for i=1:size(ph_outputs_s,1)
    ph_out(i,1)=mean(ph_outputs_s(i,(ph_outputs_s(i,:)>(mean_nn(i)-std_nn(i))&ph_outputs_s(i,:)<(mean_nn(i)+std_nn(i)))));
end

out=ph_out;

