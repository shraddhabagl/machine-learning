setwd("C:\\Users\\admin\\Desktop\\advance analytics\\Datasets")
housing<-read.csv("Housing.csv")
ctab<-table(housing$gashw,housing$prefarea)
chisq.test(ctab)