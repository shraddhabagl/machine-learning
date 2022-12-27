setwd("C:\\Users\\admin\\Desktop\\advance analytics\\Datasets")
agr<-read.csv("Yield.csv")
agr_ols<-aov(Yield ~ Treatments,data=agr_ols)
anova(agr_ols)
##turkey
TukeyHSD(agr_ols)