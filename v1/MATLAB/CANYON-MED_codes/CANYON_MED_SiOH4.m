function out=CANYON_MED_SiOH4(gtime,lat,lon,pres,temp,psal,doxy)
% function out=CANYON_SiOH4(gtime,lat,lon,pres,temp,psal,doxy)
% 
% Multi-layer perceptron to predict silicate concentration / umol kg-1 
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
% out   - silicate / umol kg-1
%
% check value: 6.3479 umol kg-1
% for 09-Apr-2014, 35° N, 18° E, 500 dbar, 13.5 °C, 38.6 psu, 160 umol O2 kg-1
%
%
% Marine Fourrier, LOV
% 10.06.2020

% No input checks! Assumes informed use, e.g., same dimensions for all
% inputs, ...

basedir='C:/Users/nouno/OneDrive/Documents/GitHub/CANYON-MED/MATLAB/'; % relative or absolute path to CANYON-MED folder

% input preparation
gvec=datevec(gtime);
year=gvec(:,1); % get year number
doy=floor(datenum(gtime)-datenum(gvec(1),1,0))*360/365; % only full yearday used; entire year (365 d) mapped to 360°
lon(lon>180)=lon(lon>180)-360;
% doy sigmoid scaling
presgrid=dlmread([basedir 'CY_doy_pres_limit.csv'],'\t');
[x,y]=meshgrid(presgrid(2:end,1),presgrid(1,2:end));
prespivot=interp2(x,y,presgrid(2:end,2:end)',lon(:),lat(:)); % Pressure pivot for sigmoid
fsigmoid=1./(1+exp((pres(:)-prespivot)./50));
% input sequence: independent of year
%     lat/90,   lon,    cos(day),    sin(day),    year,    temp,   sal,    oxygen, P 
data=[lat(:)/90 lon(:) cosd(doy(:)).*fsigmoid(:) sind(doy(:)).*fsigmoid(:) year(:) temp(:) psal(:) doxy(:) pres(:)./2e4+1./((1+exp(-pres(:)./300)).^3)];


Moy=load(strcat(basedir,'CANYON-MED_weights/moy_sil.txt'));
Ecart=load(strcat(basedir,'CANYON-MED_weights/std_sil.txt'));

ne=9;  % Number of inputs
% NORMALISATION OF THE PARAMETERS
[rx,~]=size(data);
data_norm=(2./3)*(data-(ones(rx,1)*Moy(1:ne)))./(ones(rx,1)*Ecart(1:ne));

%
n_list=10;
sil_outputs_s=zeros(size(data_norm,1),n_list);
for i=1:n_list
    b1=load(strcat(basedir,'CANYON-MED_weights/poids_sil_b1_',string(i),'.txt'));
    b2=load(strcat(basedir,'CANYON-MED_weights/poids_sil_b2_',string(i),'.txt'));
    b3=load(strcat(basedir,'CANYON-MED_weights/poids_sil_b3_',string(i),'.txt'));
    IW=load(strcat(basedir,'CANYON-MED_weights/poids_sil_IW_',string(i),'.txt'));
    LW1=load(strcat(basedir,'CANYON-MED_weights/poids_sil_LW1_',string(i),'.txt'));
    LW2=load(strcat(basedir,'CANYON-MED_weights/poids_sil_LW2_',string(i),'.txt'));
    
    sil_outputs=(LW2*custom_MF(LW1*custom_MF(IW*data_norm(:,1:end)'+b1)+b2)+b3)';
    sil_outputs=1.5*sil_outputs*Ecart(ne+1)+Moy(ne+1);
    sil_outputs_s(:,i)=sil_outputs;
end

sil_out=mean(sil_outputs_s,2);

out=sil_out;

 
