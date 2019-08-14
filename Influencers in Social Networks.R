################################################################################################
#### This code is for the Influencers in Social Networks Kaggle Competition
## Created: August 13, 2019
## Edited:
################################################################################################

rm(list = ls())

# video url: https://www.youtube.com/watch?v=ufHo8vbk6g4
# Higgs Boson Data from Kaggle: https://www.kaggle.com/c/higgs-boson/data

library(xgboost)
library(methods)
library(DiagrammeR)



################################################################################################
# Begin Code
################################################################################################
# influencers in social networks
test_sn <- read.csv('test_sn.csv', header=TRUE)
train_sn <- read.csv('train_sn.csv', header=TRUE)
sub <- read.csv('sample_predictions.csv', header=TRUE)

# set up our test, train, and variable to be predicted from the trainig set
y = train_sn[, 1]
train_sn = as.matrix(train_sn[, -1])
test = as.matrix(test_sn)

# take a look at the data
colnames(train_sn)
train_sn[1,]



#######################
# feature engineering #
#######################



# increase the information in the data
# double the data by flipping the order of the predictors
new.train = cbind(train_sn[,12:22], train_sn[,1:11])
train_sn = as.matrix(rbind(train_sn, new.train))
y = as.matrix(c(y, 1-y))

# combine training and testing sets
x = rbind(train_sn, test_sn)

# smooth the ratio by a constanct
calcRatio = function(dat, i, j, lambda=1) (dat[,i]+lambda)/(dat[,j]+lambda)

# calculate the ratios with the function
A.follow.ratio = calcRatio(x, 1, 2)
A.mention.ratio = calcRatio(x, 4, 6)
A.retweet.ratio = calcRatio(x, 5, 7)
A.follow.post = calcRatio(x, 1, 8)
A.mention.post = calcRatio(x, 4, 8)
A.retweet.post = calcRatio(x, 5, 8)
B.follow.ratio = calcRatio(x, 12, 13)
B.mention.ratio = calcRatio(x, 15, 17)
B.retweet.ratio = calcRatio(x, 16, 18)
B.follow.post = calcRatio(x, 12, 19)
B.mention.post = calcRatio(x, 15, 19)
B.retweet.post = calcRatio(x, 16, 19)

# combine all the features into a data set
x = cbind(x[,1:11],
          A.follow.ratio, A.mention.ratio, A.retweet.ratio, A.follow.post, A.mention.post, A.retweet.post,
          x[,12:22],
          B.follow.ratio, B.mention.ratio, B.retweet.ratio, B.follow.post, B.mention.post, B.retweet.post)

# compare the differences between A and B
# XGBoost scale is invariant -> subtraction and division are essentially the same (only care about relative ranking)
AB.diff = x[,1:17] - x[,18:34]

# set up new testing and trainign sets with the A-B difference calculation included
x = cbind(x, AB.diff)
train_sn = as.matrix(x[1:nrow(train_sn),])
test_sn = as.matrix(x[-(1:nrow(train_sn)),])



############
# modeling #
############


set.seed(1234)
cv.res = xgb.cv(data=train_sn, nfold=3, label=y, nrounds=100, verbose=FALSE,
                objective='binary:logistic', eval_metric='auc')

# after some trials
cv.res = xgb.cv(data=train_sn, nfold=3, label=y, nrounds=3000,
                objective='binary:logistic', eval_metric='auc',
                eta=0.005, gamma=1, lambda=3, nthread=8,
                max_depth=4, min_child_weight=1, verbose=F,
                subsample=0.8, colsample_bytree=0.8)

names(cv.res)
which.max(cv.res$evaluation_log$test_auc_mean)
best_mean_auc <- max(cv.res$evaluation_log$test_auc_mean)
best_mean_auc
best_mean_auc_std <- max(cv.res$evaluation_log$test_auc_std)
best_mean_auc_std
best <- best_mean_auc - best_mean_auc_std
best