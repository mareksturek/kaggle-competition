
# TFI Restaurant Revenue Prediction. 
# The goal is to predict the annual revenue of a particular location given a set of datapoints associated with that location. 
# https://www.kaggle.com/c/restaurant-revenue-prediction/
# 10/27/2020


# Import dataset 
setwd("~/Documents/Github/kaggle-competition/data")
tfitrain <- read.csv("tfitrain.csv",header=T,stringsAsFactors=F )

# View(tfi_data)
str(tfitrain)
summary(tfitrain)
head(tfitrain)


# Exploratory Data Analysis "Correlations"
install.packages("psych")
library(psych)
pairs.panels(tfitrain[, c(1:10,43)])
pairs.panels(tfitrain[, c(11:21,43)])
pairs.panels(tfitrain[, c(22:33,43)])
pairs.panels(tfitrain[, c(34:42,43)])


# Beginning at P14 avg median reduces
boxplot(tfitrain[,6:25])

# from P24 median drops to near zero, very skewed
boxplot(tfitrain[,26:42])


# Convert Open.Date from chr to date
# Reformat Open.date and add to data frame
Open.date.new <- as.Date((tfitrain$Open.Date),format="%m/%d/%Y")
tfitrain2 <- data.frame(tfitrain,Open.date.new)

Open.date.month <- as.factor(months(Open.date.new))
Open.date.quarter <- as.factor(quarters(Open.date.new))
tfitrain2 <- data.frame(tfitrain2,Open.date.quarter,Open.date.month)

library(caret)
featurePlot(x=tfitrain2[,c("Open.date.new","Open.date.quarter","Open.date.month")],
            y = tfitrain2$revenue,
            plot="pairs")

# Show plots 
qplot(Open.date.new,revenue,data=tfitrain2)
qplot(Open.date.new,revenue,colour=City.Group,data=tfitrain2, size=I(6))
qplot(Open.date.new,revenue,colour=Type,data=tfitrain2, size=I(6))
qplot(Open.date.new,revenue,colour=City,data=tfitrain2, size=I(6))

qplot(Open.date.month,revenue,data=tfitrain2, 
      geom = c("jitter", "boxplot"))
qplot(Open.date.quarter,revenue,data=tfitrain2, 
      geom = c("jitter", "boxplot"))
qplot(Open.date.quarter,revenue,data=tfitrain2, 
      geom = c("boxplot", "jitter"))

#Preprocessing
#Center P1-P37, columns 6-42
library(caret)
set.seed(55232)
preObj <- preProcess(tfitrain2[,6:42],method = c("center","scale"))
predict(preObj, tfitrain2[,6:42]) 
predict(preObj, tfitrain2[,6:42])$P37
mean(predict(preObj,tfitrain2[,6:42])$P37)
P1_37_scld <- predict(preObj, tfitrain2[,6:42])


# Model fit on centered data
set.seed(25332)
tfitrain3 <- data.frame(tfitrain2$revenue,tfitrain2[,6:42])
modelFit <- train(tfitrain2.revenue ~ ., data = tfitrain3,
                  preProcess = c("center","scale"),method="glm")
modelFit

# Normalize using Box-Cox
library(e1071)
preObj2 <- preProcess(tfitrain3[,-1],method = c("BoxCox"))
P1_37_BoxCox <- predict(preObj2, tfitrain3[,-1]) 


# Remove zero covariates
nsv <- nearZeroVar(tfitrain3,saveMetrics=TRUE)
nsv

# Splines
library(splines)
bsBasis <- bs(tfitrain3$P37,df=3)
bsBasis
lm1 <- lm(tfitrain2.revenue ~ bsBasis, data = tfitrain3)
plot(tfitrain3$P37,tfitrain3$tfitrain2.revenue,pch=19,cex=0.5)
points(tfitrain3$P37,predict(lm1,newdata=tfitrain3),col="red",pch=19,cex=0.5)
lines(tfitrain3$P37,predict(lm1,newdata=tfitrain3),col="red",pch=19,cex=0.5) 


# Linear model fits, multiple regression
# Columns P6,P8,P20,P26,P28
summary(lm(tfitrain3)) 

library(caret)
modFit <- train(tfitrain2.revenue ~ P6 + P8 + P20 + P26 + P28,
                method = "lm", data = tfitrain3)
print(modFit)

modFit2 <- train(revenue ~ P6 + P8 + P20 + P26 + P28 + Open.date.new,
                 method = "lm", data = tfitrain2)
print(modFit2)

modFit3 <- train(revenue ~ P6 + P8 + P20 + P26 + P28 + Open.date.new
                 + as.factor(City),
                 method = "lm", data = tfitrain2) 
print(modFit3)


# Predictions
library(caret)
set.seed(12553)

# By default, simple bootstrap resampling, K=10-fold CV
fitControl <- trainControl(# 10-fold CV
        method = "repeatedcv",
        number = 10,
        ## repeated ten times
        repeats = 10)

# Boosted tree model via the gbm package. 
set.seed(39552)
tfitrain4 <- data.frame(tfitrain3,tfitrain2$Open.date.new)
gbmFit1 <- train(tfitrain2.revenue ~ ., data = tfitrain4,
                 method = "gbm",
                 trControl = fitControl,
                 verbose = FALSE)
gbmFit1

# Import test data
tfitest <- read.csv("tfitest.csv",header=T,stringsAsFactors=F)
Open.date.test.new <- as.Date((tfitest$Open.Date),format="%m/%d/%Y")
tfitest2 <- data.frame(tfitest[,6:42],Open.date.test.new)
names(tfitest2)[38] <- "tfitrain2.Open.date.new" #for prediction
predict(gbmFit1,newdata=head(tfitest2)) #first 6 predictions
rpred <- predict(gbmFit1,newdata=tfitest2) #vector of predictions

# Create columns "ID" and "Predictions"
rpred <- cbind(0:(length(rpred)-1),rpred)
rpred <- data.frame(rpred)
colnames(rpred)[1:2] <- c("Id","Prediction")

# Write results 
write.csv(rpred,file="tfiresults.csv",row.names=F)