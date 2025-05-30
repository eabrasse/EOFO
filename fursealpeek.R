library(ggplot2)
library(cetcolor)
library(sf)
setwd('/Users/elizabethbrasseale/Projects/EOFO/')

fname<-'data/Fur seal 632 georeferenced dives.rds'
furseal632 = readRDS(fname)

NFS_trip1<-furseal632[furseal632$tripno == 1 & furseal632$maxDepth > 4,]
#NFS_dive1<-furseal632[furseal632$diveno == 1736 & furseal632$maxDepth > 4,]

ggplot(NFS_trip1[NFS_trip1$diveno==1736,], aes(x=gmt, y=-CorrectedDepth))+
  geom_point(aes(color=etemp))+
  scale_x_datetime(date_labels =  "%d %b %Y, %H:%M:%S")+
  theme(axis.text.x=element_text(angle=60, hjust=1))+
  scale_colour_gradientn(colours = cet_pal(5, name = "inferno"),name = "Temp")+
  xlab('Time') + ylab('Depth') + ggtitle('Northern Fur Seal 632\nDive no. 1736')

ggplot(NFS_trip1, aes(x=gmt, y=-CorrectedDepth))+
  geom_point(aes(color=etemp))+
  scale_x_datetime(date_labels =  "%d %b %Y, %H:%M:%S")+
  theme(axis.text.x=element_text(angle=60, hjust=1))+
  scale_colour_gradientn(colours = cet_pal(5, name = "inferno"),name = "Temp")+
  xlab('Time') + ylab('Depth') + ggtitle('Northern Fur Seal 632')

ggplot(NFS_trip1)+geom_point(aes(geometry=geometry),stat="sf_coordinates")
