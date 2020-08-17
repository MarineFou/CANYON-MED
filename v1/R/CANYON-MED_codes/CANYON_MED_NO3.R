CANYON_MED_NO3<- function(date,lat,lon,pres,temp,psal,doxy) {
  # Multi-layer perceptron to predict nitrate concentration / umol kg-1 
  #
  # Neural network training by Marine Fourrier from work by Raphaelle Sauzede, LOV; 
  # as R function by Marine Fourrier, LOV
  #
  #
  # input:
  # gtime - date (UTC) as string ("yyyy-mm-dd HH:MM")
  # lat   - latitude / °N  [-90 90]
  # lon   - longitude / °E [-180 180] or [0 360]
  # pres  - pressure / dbar
  # temp  - in-situ temperature / °C
  # psal  - salinity
  # doxy  - dissolved oxygen / umol kg-1 
  #
  # output:
  # out   - nitrate / umol kg-1
  #
  # check value:  5.8614 umol kg-1
  # for 09-Apr-2014, 35° N, 18° E, 500 dbar, 13.5 °C, 38.6 psu, 160 umol O2 kg-1
  #
  #
  # Marine Fourrier, LOV
  # 10.06.2020
  
  # No input checks! Assumes informed use, e.g., same dimensions for all
  # inputs, ...
  require(fields)
  
  
  basedir <- "C:/Users/nouno/OneDrive/Documents/GitHub/CANYON-MED/R/" # relative or absolute path to CANYON-MED folder
  
  # input preparation
  date <- as.POSIXct(date)
  day <- as.numeric(format(date,"%j"))*360/365 # only full yearday used; entire year (365 d) mapped to 360°
  year <- as.numeric(format(date,"%Y"))
  lon[which(lon>180)]=lon[which(lon>180)]-360
  
  
  # doy sigmoid scaling
  pivot_doy <- read.table(paste(basedir,"CY_doy_pres_limit.csv",sep=""))
  pivot_mat <- as.matrix(pivot_doy[-1,-1])
  lat_pivot <- unlist(pivot_doy[1,][-1])
  lon_pivot <- pivot_doy[,1][-1]
  lon_rad <- lon*pi/180. 
  obj <- list(x=lon_pivot,y=lat_pivot,z=pivot_mat)
  loc <- matrix(cbind(lon,lat),ncol=2)
  Ppivot <- fields::interp.surface(obj,loc)
  fsigmoid <- 1 / (1 + exp((pres-Ppivot)/50 ))
  
  # input sequence: independent of year
  #     lat/90,   lon,    cos(day),    sin(day),    year,    temp,   sal,    oxygen, P 

  cos_dat <- cospi(day/180) * fsigmoid
  sin_dat <- sinpi(day/180) * fsigmoid
  
  data=data.frame(cbind(lat/90,lon,cos_dat,sin_dat,year,temp,psal,doxy,pres/2e4+1/((1+exp(-pres/300))^3)))

  
    
  Moy <- read.table(paste(basedir,"CANYON-MED_weights/moy_nit.txt",sep=""))
  Ecart <- read.table(paste(basedir,"CANYON-MED_weights/std_nit.txt",sep=""))
  
  ne = 9 # Number of inputs
  
  # NORMALISATION OF THE PARAMETERS
  data_N <- data[,1:ne]
  
  for(i in 1:ne){
    data_N[,i] <- (2/3)*((data[,i]-Moy[,i])/Ecart[,i])
  }
  
  data_N <- as.matrix(data_N)
  
  #
  n_list=10
  
  nit_outputs_s=rep(0,n_list)
  
  rx <-dim(data_N)[1]
  for(i in 1:n_list) {
    
    b1=read.table(paste(basedir,'CANYON-MED_weights/poids_nit_b1_',as.character(i),'.txt',sep=""))
    b2=read.table(paste(basedir,'CANYON-MED_weights/poids_nit_b2_',as.character(i),'.txt',sep=""))
    b3=read.table(paste(basedir,'CANYON-MED_weights/poids_nit_b3_',as.character(i),'.txt',sep=""))
    IW=read.table(paste(basedir,'CANYON-MED_weights/poids_nit_IW_',as.character(i),'.txt',sep=""))
    LW1=read.table(paste(basedir,'CANYON-MED_weights/poids_nit_LW1_',as.character(i),'.txt',sep=""))
    LW2=read.table(paste(basedir,'CANYON-MED_weights/poids_nit_LW2_',as.character(i),'.txt',sep=""))
    b1 <- as.matrix(b1)
    b2 <- as.matrix(b2)
    b3 <- as.matrix(b3)

    #hidden layers
    a <- 1.715905*tanh((2./3)*(data_N %*% t(IW)+t(b1 %*% t(rep(1,rx)))))
    b <- 1.715905*tanh((2./3)*(a %*% t(LW1)+t(b2 %*% t(rep(1,rx)))))
    y <- b %*% t(LW2)+t(b3 %*% rep(1,rx))
    nit_outputs=1.5*y*Ecart[1,ne+1]+Moy[1,ne+1]
    
    nit_outputs_s[i]=nit_outputs
  }
  
  nit_out=mean(nit_outputs_s);
  
  out=nit_out;
  return(out)
}
