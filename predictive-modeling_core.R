
suppressMessages(library(raster))
suppressMessages(library(rgdal))
suppressMessages(library(foreach))
suppressMessages(library(caret))

# deifne your working directory
workdir="E:/"

#load ancillary functions
source(paste(workdir, "predictive-modeling_functions.R", sep=""))

#KAR/THF
site="THF"

#sample point pixel spacing in meters
targetres=12

#number of cores for parallel processing
n.cores=6

########################################################
#SAMPLING OF RASTER LAYERS

#load all image files within an image folder matching the defined criteria
imagepath="D:/"
images=lapply(patternator(path,in_p=site,in_s=".tif"),function(x)raster::raster(x))

###########
#small sample

name.sample.in=paste(workdir,site,"_small.shp",sep="")
name.sample.out=paste(workdir,site,"_small.shp",sep="")
sample=shapereader(name.sample.in)

smallsample=multiextract(rasters,sample,cores=n.cores)

shapewriter(smallsample,name.sample.out)

###########
#full sample

#load bounding box shapefile
bbox=shapereader(paste(workdir,"bbox_",site,".shp",sep=""))

#perform the sampling
fullsample=multiextract(images,sampler(bbox,targetres),cores=n.cores)

outname=paste(workdir,site,"_full.RData",sep="")
save(fullsample,file=outname)

########################################################
#MODELING

model="rf"

tags.pred=c("^rs","^ta","^td","^idem")

n.straps=500

sample=smallsample@data

# species,biomass
response.type="biomass"

###########
#biomass

if(response.type=="biomass"){
  response.name=if (site=="THF")"Bio1_t_ha" else "bm_cl_tha"
  fitControl=caret::trainControl(method="repeatedcv",number=10,repeats=5,returnResamp="final",allowParallel=T)
  metric="RMSE"
}else if(response.type=="species"){
  response.name=if (site=="THF") "Spec_Sho" else "species"
  fitControl=caret::trainControl(method="repeatedcv",number=10,repeats=5,returnResamp="all",allowParallel=T,savePredictions=T,classProbs=T)
  metric="Accuracy"
}

straps=resample(sample[,response.name],n.straps)

cl=parallel::makeCluster(n.cores)
doParallel::registerDoParallel(cl,n.cores)

tune=list()
for(tag.pred in tags.pred){
  pred.ids=grep(tag.pred,names(sample))
  
  descriptor=substring(tag.pred,2,nchar(tag.pred))
  
  if(!length(pred.ids)==0){
    tune[[descriptor]]=lapply(straps,function(x)caret::train(sample[x,pred.ids],sample[x,response.name],
                                                             method=model,trControl=fitControl,tuneLength=12,importance=T))
  }
}
unregister(cl)

outname=paste(workdir,"predstrat_",response.type,"_",site,"_",model,".RData",sep="")
save(tune,file=outname)

########################################################
#VISUALIZATION OF GOODNESS OF FIT (BIOMASS ONLY)

#Rsquared, RMSE, rRMSE
measure="RMSE"

name.var.lookup=list("Rsquared"=substitute(expression(R^2)),
                     "RMSE"=expression(paste("RMSE [",t%.%ha^-1,"]",sep="")),
                     "rRMSE"="rRMSE [%]")

#extraction of modeling result measures
if(measure=="rRMSE"){
  measures.l=lapply(tune,function(x)sapply(x,function(y)(min(y$results[,"RMSE"])/mean(y$trainingData$.outcome))*100))
}else{
  measures.l=lapply(tune,function(x)sapply(x,function(y)min(y$results[,measure])))
}

measures.df=data.frame(matrix(unlist(measures.l),nrow=length(measures.l[[1]]),byrow=F))
names(measures.df)=names(tune)

name.out=paste(workdir,"/",site,"_predstrat_beanplot_",measure,".pdf",sep="")
beanplotter(measures.df,name.out,name.var.lookup[[measure]])
########################################################
#COMPUTATION OF ACCURACY MEASURES (SPECIES ONLY)

pred=lapply(tune,function(x)unlist(lapply(x[1:(length(x)-1)],function(y)y$pred$pred)))
obs=lapply(tune,function(x)unlist(lapply(x[1:(length(x)-1)],function(y)y$pred$obs)))

acc=lapply(seq(pred),function(x)accuracy(pred[[x]],obs[[x]]))
names(acc)=names(tune)
########################################################
#VISUALIZATION OF VARIABLE IMPORTANCE

for(name in names(tune)){
  name.out.varimp=paste(workdir,site,"_predstrat_varimp_",name,".pdf",sep="")
  importance2(tune[[name]],name.out.varimp)
}
########################################################
#FULL PREDICTION AND RASTERIZATION: BIOMASS

for(name.tune in names(tune)){
  prediction=multipredict2(tune[[name.tune]],fullsample@data,cores=n.cores)
  
  prediction.df=data.frame(matrix(unlist(prediction),nrow=length(prediction[[1]]),byrow=F))
  
  save(prediction.df,file=paste(tools::file_path_sans_ext(name.tune),"_prediction.RData",sep=""))
  
  fullsample@data$pred_av=apply(prediction.df,1,mean,na.rm=T)
  fullsample@data$pred_sd=apply(prediction.df,1,sd,na.rm=T)
  fullsample@data$pred_cv=fullsample$pred_sd/fullsample$pred_av*100
  
  for(var in c("pred_av","pred_sd","pred_cv")){
    name.out=paste(tools::file_path_sans_ext(name.tune),"_",var,sep="")
    point2ras(sample,var,outname=name.out)
  }
}
########################################################
#FULL PREDICTION AND RASTERIZATION: SPECIES

for(name.tune in names(tune)){
  prediction=multipredict2(tune[[name.tune]],fullsample@data,cores=n.cores)
  
  prediction.df=data.frame(matrix(unlist(prediction),nrow=length(prediction[[1]]),byrow=F))
  
  save(prediction.df,file=paste(tools::file_path_sans_ext(name.tune),"_prediction.RData",sep=""))
  
  fullsample@data$pred_med=apply(prediction,1,median,na.rm=T)
  #full$pred=unclass(factor(apply(prediction,1,median,na.rm=T)))
  
  pred_med.index=grep("pred_med",names(fullsample@data))
  
  fullsample@data$pred_acc=apply(prediction,1,function(x)length(x[x==x[pred_med.index]])/ncol(prediction))
  
  for(var in c("pred_med","pred_acc")){
    point2ras(fullsample,var,outname=paste(site,tune[[1]]$method,"_",var,sep=""))
  }
}
########################################################
