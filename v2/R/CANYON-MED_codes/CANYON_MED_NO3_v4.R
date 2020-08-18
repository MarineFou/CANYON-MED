CANYON_MED_NO3_v4<- function(date,lat,lon,pres,temp,psal,doxy) {
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
  # check value:  5.9115 umol kg-1
  # for 09-Apr-2014, 35° N, 18° E, 500 dbar, 13.5 °C, 38.6 psu, 160 umol O2 kg-1
  #
  #
  # Marine Fourrier, LOV
  # 17.08.2020
  
  # No input checks! Assumes informed use, e.g., same dimensions for all
  # inputs, ...
  require(fields)
  require(matrixStats)
  
  
  basedir <- "D:/Documents/Thèse/Docs/science/PAPIER_CANYON_MED/CODES/CANYON-MED/v2/R/" # relative or absolute path to CANYON-MED folder
  
  # input preparation
  date=as.POSIXlt(date,tz="UTC") #decimal year from date
  year=format(date,"%Y") # year
  # add the two
  dec_year=as.numeric(year)+as.double(date-as.POSIXlt(paste0(year,"-01-01 00:00"),format="%Y-%m-%d %H:%M",tz="UTC"))/365
  lon[which(lon>180)]=lon[which(lon>180)]-360
  
  
  # input sequence: 
  #     lat,   lon,    dec_year,    temp,   sal,    oxygen, P 

  data=data.frame(cbind(lat,lon,dec_year,temp,psal,doxy,pres/2e4+1/((1+exp(-pres/300))^3)))

  
    
  moy_F<- read.table(paste(basedir,"CANYON-MED_weights/moy_nit_F.txt",sep=""))
  std_F <- read.table(paste(basedir,"CANYON-MED_weights/std_nit_F.txt",sep=""))
  
  ne = 7 # Number of inputs
  
  # NORMALISATION OF THE PARAMETERS
  data_N <- data[,1:ne]
  
  for(i in 1:ne){
    data_N[,i] <- (2/3)*((data[,i]-moy_F[,i])/std_F[,i])
  }
  
  data_N <- as.matrix(data_N)
  
  #
  n_list=5
  
  nit_outputs_s=data.frame(matrix(NA, nrow = nrow(data), ncol = 10))
  
  rx <-dim(data_N)[1]
  for(i in 1:n_list) {
    
    b1=read.table(paste(basedir,'CANYON-MED_weights/poids_nit_b1_F_',as.character(i),'.txt',sep=""))
    b2=read.table(paste(basedir,'CANYON-MED_weights/poids_nit_b2_F_',as.character(i),'.txt',sep=""))
    b3=read.table(paste(basedir,'CANYON-MED_weights/poids_nit_b3_F_',as.character(i),'.txt',sep=""))
    IW=read.table(paste(basedir,'CANYON-MED_weights/poids_nit_IW_F_',as.character(i),'.txt',sep=""))
    LW1=read.table(paste(basedir,'CANYON-MED_weights/poids_nit_LW1_F_',as.character(i),'.txt',sep=""))
    LW2=read.table(paste(basedir,'CANYON-MED_weights/poids_nit_LW2_F_',as.character(i),'.txt',sep=""))
    b1 <- as.matrix(b1)
    b2 <- as.matrix(b2)
    b3 <- as.matrix(b3)

    #hidden layers
    a <- 1.715905*tanh((2./3)*(data_N %*% t(IW)+t(b1 %*% t(rep(1,rx)))))
    b <- 1.715905*tanh((2./3)*(a %*% t(LW1)+t(b2 %*% t(rep(1,rx)))))
    y <- b %*% t(LW2)+t(b3 %*% rep(1,rx))
    nit_outputs=1.5*y*std_F[1,ne+1]+moy_F[1,ne+1]
    
    nit_outputs_s[i]=nit_outputs
  }
  
  #
  moy_G<- read.table(paste(basedir,"CANYON-MED_weights/moy_nit_G.txt",sep=""))
  std_G <- read.table(paste(basedir,"CANYON-MED_weights/std_nit_G.txt",sep=""))
  
  ne = 7 # Number of inputs
  
  # NORMALISATION OF THE PARAMETERS
  data_N <- data[,1:ne]
  
  for(i in 1:ne){
    data_N[,i] <- (2/3)*((data[,i]-moy_G[,i])/std_G[,i])
  }
  
  data_N <- as.matrix(data_N)
  
  #

  rx <-dim(data_N)[1]
  for(i in 1:n_list) {
    
    b1=read.table(paste(basedir,'CANYON-MED_weights/poids_nit_b1_G_',as.character(i),'.txt',sep=""))
    b2=read.table(paste(basedir,'CANYON-MED_weights/poids_nit_b2_G_',as.character(i),'.txt',sep=""))
    b3=read.table(paste(basedir,'CANYON-MED_weights/poids_nit_b3_G_',as.character(i),'.txt',sep=""))
    IW=read.table(paste(basedir,'CANYON-MED_weights/poids_nit_IW_G_',as.character(i),'.txt',sep=""))
    LW1=read.table(paste(basedir,'CANYON-MED_weights/poids_nit_LW1_G_',as.character(i),'.txt',sep=""))
    LW2=read.table(paste(basedir,'CANYON-MED_weights/poids_nit_LW2_G_',as.character(i),'.txt',sep=""))
    b1 <- as.matrix(b1)
    b2 <- as.matrix(b2)
    b3 <- as.matrix(b3)
    
    #hidden layers
    a <- 1.715905*tanh((2./3)*(data_N %*% t(IW)+t(b1 %*% t(rep(1,rx)))))
    b <- 1.715905*tanh((2./3)*(a %*% t(LW1)+t(b2 %*% t(rep(1,rx)))))
    y <- b %*% t(LW2)+t(b3 %*% rep(1,rx))
    nit_outputs=1.5*y*std_G[1,ne+1]+moy_G[1,ne+1]
    
    nit_outputs_s[i+5]=nit_outputs
  }
  
  #mean_nn=rowMeans(nit_outputs_s)
  #std_nn = apply(nit_outputs_s[,], 1, sd)
  #std_nn=sd(nit_outputs_s)
  
  mean_nn<-rowMeans(nit_outputs_s)
  std_nn<-rowSds(as.matrix(nit_outputs_s))
  
  lim_inf<-mean_nn-std_nn
  lim_sup<-mean_nn+std_nn
  
  nit_t<-nit_outputs_s
  nit_t[nit_t<lim_inf|nit_t>lim_sup]<-NA

  #nit_t$out<-mean(nit_t,na.rm = TRUE)
  nit_out<-rowMeans(nit_t,na.rm = TRUE)
  
  
  #nit_out<-data.frame()[nrow(nit_outputs_s),1 ]
  
  #nit_t<-nit_outputs_s
  #nit_t[nit_t<(mean_nn-std_nn)|nit_t>(mean_nn+std_nn)] <- NA

  #nit_t <- nit_t %>% tibble() %>% 
  #  mutate(mean_out = mean(.))
    
  

  out<-nit_out;
  return(out)
}
