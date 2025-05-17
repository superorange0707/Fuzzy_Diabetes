install.packages("mice")
install.packages("Rlof")
install.packages("corrplot")
install.packages("PerformanceAnalytics")
install.packages("CORElearn")
install.packages("caret")
install.packages("sampling")
install.packages("VIM")
install.packages("GGally")
install.packages("frbs")
install.packages("anfis")
install.packages("FSelector")
install.packages("MLeval")
library(mice)
library(VIM)
library(corrplot)
library(FSelector)
library(ggplot2)
library(GGally)
library(psych)
library(dplyr)
library(gridExtra)
library(PerformanceAnalytics)
library(Rlof)
library(tidyr)
library(CORElearn)
library(caret)
library(sampling)
library(scorecard)
library(scatterplot3d)
library(pROC)
library(MLeval)
library(frbs)


getwd()

# read data
df = as.data.frame(read.csv('diabetes.csv',stringsAsFactors = F))
# df$Outcome = as.factor(df$Outcome)

# missing value
md.pattern(df)

aggr_plot = aggr(df, col=c('navyblue','red'), 
                  numbers=TRUE, 
                  sortVars=TRUE, 
                  labels=names(df), 
                  cex.axis=.7, 
                  gap=3, 
                  ylab=c("hist of missing value","pattern"))
  

# Statics 
stat = describe(df)
summary(df)

df1 = as.data.frame(df$Insulin)

# boxplot
g1=ggplot(stack(df[,c(1,7)]), aes(x = ind, y = values),par(las='2')) + geom_boxplot() + xlab("Attributes") + ylab("Vaule")  + theme(axis.text.x=element_text(angle = -90, hjust = 0),panel.grid.major=element_line(colour=NA),panel.background = element_rect(fill = "transparent",colour = NA),plot.background = element_rect(fill = "transparent",colour = NA),panel.grid.minor = element_blank())
g2=ggplot(stack(df[,c(2,3,4,6,8)]), aes(x = ind, y = values)) + geom_boxplot() + xlab("Attributes") + ylab("Vaule") + theme(axis.text.x=element_text(angle = -90, hjust = 0),panel.grid.major=element_line(colour=NA),panel.background = element_rect(fill = "transparent",colour = NA),plot.background = element_rect(fill = "transparent",colour = NA),panel.grid.minor = element_blank())
g3=ggplot(df1, aes(x='Insulin', y=df1$`df$Insulin`)) +  geom_boxplot()+ xlab("Attributes") + ylab("Vaule") + theme(axis.text.x=element_text(angle = -90, hjust = 0),panel.grid.major=element_line(colour=NA),panel.background = element_rect(fill = "transparent",colour = NA),plot.background = element_rect(fill = "transparent",colour = NA),panel.grid.minor = element_blank())
grid.arrange(g1,g2,g3,top = "Boxplot of Features",nrow = 1, ncol = 3)

# remove outlier 
df = df[-c(15,176,444,581),]



# correlation
df_ma = as.matrix(df[,c(1:8)])
df_ma = apply(df_ma, 2, as.numeric)
chart.Correlation(df_ma, histogram=TRUE, pch=19)


# correlation with outcome

ggpairs(df, aes(color=df$Outcome, alpha=0.75), lower=list(continuous="smooth"),upper = list(continuous = wrap("cor", size = 2)))+ theme_bw()+
  
labs(title="Correlation Plot of Variance(diabetes)")+
  
theme(plot.title=element_text(face='bold',color='black',hjust=0.5,size=12))
# ggsave ( file =  "corr.eps",width=12, height=8,dpi = 1200)




# miss value
# fill missing value with mean
mean_glu = stat[stat$variable=='Glucose']$mean
mean_bp = stat[stat$variable=='BloodPressure',]$mean
mean_st = stat[stat$variable=='SkinThickness',]$mean
mean_in = stat[stat$variable=='Insulin',]$mean
mean_bmi = stat[stat$variable=='BMI',]$mean

df$Glucose[df$Glucose==0] = mean_glu
df$BloodPressure[df$BloodPressure==0] = mean_bp
df$SkinThickness[df$SkinThickness==0] = mean_st
df$Insulin[df$Insulin==0] = mean_in
df$BMI[df$BMI == 0] = mean_bmi



# feature extraction：stepwise
# step1
tlm = lm(df$Outcome~df$Pregnancies+df$Glucose+df$BloodPressure+df$SkinThickness+df$Insulin+df$BMI+df$DiabetesPedigreeFunction+df$Age)
summary(tlm)

# step2
tstep = step(tlm)
summary(tstep)

# step3
drop1(tstep)

# step3
tlm1 = lm(df$Outcome~df$Pregnancies+df$Glucose+df$BMI+df$DiabetesPedigreeFunction)
summary(tlm1)


# EXTRACTION
df_data = df[,c(1,2,6,7,9)]
# df_data$Outcome=as.factor(df_data$Outcome)

# train and test set:Stratified Sampling
data_split = split_df(df_data,ratios = c(0.7,0.3))
train_set = data_split$train
test_set = data_split$test
train_set$Outcome=as.factor(train_set$Outcome)
test_set$Outcome=as.factor(test_set$Outcome)

# train
# K-fold cross validation:10
tran_fit = trainControl(method = "cv", number = 10)


# KNN_model
# # data standardization and train model
# try different k value
knn_model = train(Outcome~.,train_set,method = "knn", preProcess = c('center','scale'), trControl = tran_fit, tuneLength = 10 )
# draw the plot of accuracy of different k
knn_k = as.data.frame(knn_model$results)
ggplot(data=knn_k, mapping = aes(x=factor(k), y=Accuracy, group = 1)) + geom_line(color="steelblue") + geom_point()+geom_text(mapping = aes(label = round(Accuracy,4)),vjust = -1)+ xlab("K Vaule") + ylab("Accuracy") +ggtitle("Accuracy of different K value")+theme(plot.title = element_text(hjust = 0.5))

# prediction
pred_knn = predict(knn_model,newdata = test_set)

true =test_set$Outcome

confusionMatrix(pred_knn, true,mode = "everything")

roc_knn = roc(true,factor(pred_knn,ordered = T), plot = TRUE, legacy.axes = TRUE,
    percent = FALSE, xlab = "1-Specificity",
    ylab = "Sensitivity ", main="ROC and AUC of KNN", col = "#377eb8", lwd = 2,
    print.auc = TRUE)

ev_knn = confusionMatrix(pred_knn, true,mode = "everything")
res_knn = as.data.frame(rbind(ev_knn$overall))
res_knn$Accuracy







# SVM_Radial
# with optimization
svm_Radial_model = train(Outcome~.,train_set,method = "svmRadial", preProcess = c('center','scale'), trControl = tran_fit, tuneGrid = expand.grid(C= c(0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.75, 1),sigma =c(0, 0.05:2)))
# draw the plot of different parameters of SVM
svm_pa = as.data.frame(svm_Radial_model$results)
scatterplot3d(svm_pa[,1:3], pch = 16, color="steelblue")

# prediction
pred_svm = predict(svm_Radial_model,newdata = test_set)
confusionMatrix(pred_svm, true,mode = "everything")

roc_svm = roc(true,factor(pred_svm,ordered = T), plot = TRUE, legacy.axes = TRUE,
              percent = FALSE, xlab = "1-Specificity",
              ylab = "Sensitivity ", main="ROC and AUC of SVM", col = "#377eb8", lwd = 2,
              print.auc = TRUE)

ev_svm = confusionMatrix(pred_svm, true,mode = "everything")
res_svm = as.data.frame(rbind(ev_svm$overall))
res_svm$Accuracy


# Random_Forest
rf_model = train(Outcome~.,train_set,method = "rf", preProcess = c('center','scale'), trControl = tran_fit, tuneGrid = expand.grid(.mtry= c(1:10)), ntree = 500, nodesize = 2, maxnodes = 10)
rf_model
# draw the plot of accuracy of different mtry
rf_mtry = as.data.frame(rf_model$results)
ggplot(data=rf_mtry, mapping = aes(x=factor(mtry), y=Accuracy, group = 1)) + geom_line(color="steelblue") +geom_point()+geom_text(mapping = aes(label = round(Accuracy,4)),vjust = -1)+ xlab("mtry Vaule") + ylab("Accuracy") +ggtitle("Accuracy of different mtry value")+theme(plot.title = element_text(hjust = 0.5))

# prediction
pred_rf = predict(rf_model,newdata = test_set)
confusionMatrix(pred_rf, true,mode = "everything")



roc_rf = roc(true,factor(pred_rf,ordered = T), plot = TRUE, legacy.axes = TRUE,
              percent = FALSE, xlab = "1-Specificity",
              ylab = "Sensitivity ", main="ROC and AUC of RF", col = "#377eb8", lwd = 2,
              print.auc = TRUE)



ev_rf = confusionMatrix(pred_rf, true)
res_rf = as.data.frame(rbind(ev_rf$overall))
res_rf$Accuracy





# ANFIS
train = data_split$train
process = preProcess(train[,c(1:4)], method=c("range"))
norm_scale <- predict(process, train[,c(1:4)])
train_norm = norm_scale
train_norm$Outcome = train$Outcome
train_norm$Outcome = as.numeric(train_norm$Outcome)
train$Outcome = as.numeric(train$Outcome)


test = test_set[,c(1:4)]
process_test = preProcess(test, method=c("range"))
norm_scale_test <- predict(process_test,test)
test = norm_scale_test
test_set$Outcome =  as.factor(test_set$Outcome )

class(train_norm$Outcome)
class(test_set$Outcome)
length(train_norm$Outcome[train_norm$Outcome == "1"])
length(test_set$Outcome[test_set$Outcome == 0])


# generate value for yes
yes = runif(179, min=0.1, max=0.5)
# generate value for no
no = runif(337, min= 0.6, max=1)


train_norm$Outcome[train_norm$Outcome == 1] = yes
train_norm$Outcome[train_norm$Outcome == 0] = no

control.anfis <- list(num.labels = 5, max.iter=10,  step.size = 0.01 , type.mf = "GAUSSIAN", type.tnorm = "MIN", type.snorm = "MAX", type.implication.func = "ZADEH") 

# data_range = matrix(c(min(test$Pregnancies), max(test$Pregnancies), min(test$Glucose), max(test$Glucose), min(test$BMI), max(test$BMI), min(test$DiabetesPedigreeFunction), max(test$DiabetesPedigreeFunction), 1, 2), ncol=5, byrow = FALSE)

object.anfis = frbs.learn(train_norm, range.data = NULL, method.type = c("ANFIS"),control = control.anfis)

#plotMF(object.anfis)
pred_anfis = as.data.frame(predict(object.anfis,test))
length(pred_anfis$V1[pred_anfis$V1>0.95])
length(pred_anfis$V1[pred_anfis$V1<0.95])
pred_anfis = as.data.frame(predict(object.anfis,test))
pred_anfis$V1[pred_anfis$V1<0.94] = 0
pred_anfis$V1[pred_anfis$V1>0.94] = 1

pred_anfis$V1 = as.factor(pred_anfis$V1)

confusionMatrix(pred_anfis$V1, test_set$Outcome)

roc_anfis = roc(true,factor(pred_anfis$V1,ordered = T), plot = TRUE, legacy.axes = TRUE,
             percent = FALSE, xlab = "1-Specificity",
             ylab = "Sensitivity ", main="ROC and AUC of ANFIS", col = "#377eb8", lwd = 2,
             print.auc = TRUE)



ev_anfis = confusionMatrix(pred_anfis$V1, test_set$Outcome)
res_anfis = as.data.frame(rbind(ev_anfis$overall))
res_anfis$Accuracy



# Evaluation
# Accuracy
model_name = c('KNN', 'SVM', 'RF', 'ANFIS')
model_accuracy=c(res_knn$Accuracy,res_svm$Accuracy,res_rf$Accuracy,res_anfis$Accuracy)
model_kappa =c(res_knn$Kappa,res_svm$Kappa,res_rf$Kappa,res_anfis$Kappa)
model_info = data.frame(model_name, model_accuracy, model_kappa)
ggplot(data=model_info, mapping = aes(x=reorder(model_name, model_accuracy), y=model_accuracy, group = 1)) + geom_bar(stat = 'identity',fill = 'steelblue', colour = 'black')+ geom_text(mapping = aes(label = round(model_accuracy,4)),vjust = 1.2)+ xlab("Model") + ylab("Accuracy") +ggtitle("Accuracy of different Model")+theme(plot.title = element_text(hjust = 0.5))


# Consistency
ggplot(data=model_info, mapping = aes(x=reorder(model_name, model_kappa), y=model_kappa, group = 1)) + geom_bar(stat = 'identity',fill = 'steelblue', colour = 'black')+ geom_text(mapping = aes(label = round(model_kappa,4)),vjust = 1.2)+ xlab("Model") + ylab("Consistency") +ggtitle("Consistency of different Model")+theme(plot.title = element_text(hjust = 0.5))


# draw multiple-models' roc curve
plot(roc_knn,col="red",main="ROC curve of different Models",legacy.axes = TRUE)
plot(roc_svm, add=TRUE, col="blue")
plot(roc_rf, add=TRUE, col="green")
plot(roc_anfis, add=TRUE, col="black")
legend("bottomright", legend=c("knn","svm","rf","anfis"),col=c("red","blue","green","black"),lty=1, lwd=3, cex =0.7,bty="n",x.intersp=0.5,y.intersp=0.5,text.font=8,seg.len=1,inset=c(0.1,0.3))


# AUC
model_AUC = c(roc_knn$auc, roc_svm$auc, roc_rf$auc,roc_anfis$auc)
model_info$model_AUC = model_AUC
ggplot(data=model_info, mapping = aes(x=reorder(model_name, model_AUC), y=model_AUC, group = 1)) + geom_bar(stat = 'identity',fill = 'steelblue', colour = 'black')+ geom_text(mapping = aes(label = round(model_AUC,4)),vjust = 1.2)+ xlab("Model") + ylab("AUC") +ggtitle("AUC of different Model")+theme(plot.title = element_text(hjust = 0.5))
