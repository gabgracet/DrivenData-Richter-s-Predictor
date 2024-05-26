#[XGBoost Model]

#Loading packages
library(data.table) #used for reading and manipulation of data
#install.packages("recipes")
library(recipes)
#install.packages("xgboost")
library(xgboost)
#install.packages("caret")
library(caret)
#install.packages("SHAPforxgboost")
#library(SHAPforxgboost)
#library(ggplot2)

#(set working directorynya boleh disesuaikan)
setwd("C:/Users/User/Documents/UPH/Applied Math/Sem 8 (Genap)/Capstone/Richter's Predictor")

#PROCESS DATA ------------------------------------------------------------------
set.seed(2)

#1 import data
train_values = read.csv("train_values.csv")
train_labels = read.csv("train_labels.csv")
test_values = read.csv("test_values.csv")

#2 data frames (rows, coloumns)
cat("# Train Values: (", nrow(train_values), ",", ncol(train_values),")\n")
cat("# Train Labels: (", nrow(train_labels), ",", ncol(train_labels),")\n")
cat("# Test Values: (", nrow(test_values), ",", ncol(test_values),")\n")

head(train_values)
head(train_labels)
head(test_values)

#3 check missing values
## train_values
missing_values1 = sum(is.na(train_values))
cat("Total missing values:", missing_values1,")\n")
missing_values2 = sum(is.na(train_labels))
cat("Total missing values:", missing_values2,")\n")
missing_values3 = sum(is.na(test_values))
cat("Total missing values:", missing_values3,")\n")

#4 mengubah variabel character menjadi categorical
## train_values
train_values$land_surface_condition = factor(train_values$land_surface_condition)
train_values$foundation_type = factor(train_values$foundation_type)
train_values$roof_type = factor(train_values$roof_type)
train_values$ground_floor_type = factor(train_values$ground_floor_type)
train_values$other_floor_type = factor(train_values$other_floor_type)
train_values$position = factor(train_values$position)
train_values$plan_configuration = factor(train_values$plan_configuration)
train_values$legal_ownership_status = factor(train_values$legal_ownership_status)
## train_labels
train_labels$damage_grade = factor(train_labels$damage_grade)
##test_values
test_values$land_surface_condition = factor(test_values$land_surface_condition)
test_values$foundation_type = factor(test_values$foundation_type)
test_values$roof_type = factor(test_values$roof_type)
test_values$ground_floor_type = factor(test_values$ground_floor_type)
test_values$other_floor_type = factor(test_values$other_floor_type)
test_values$position = factor(test_values$position)
test_values$plan_configuration = factor(test_values$plan_configuration)
test_values$legal_ownership_status = factor(test_values$legal_ownership_status)

#5 split data categorical and numerical
##train_values
train_data = cbind(train_values[,1:8],train_values[,16:26])
train_data = cbind(train_data,train_values[,28:39])
##test_values
test_data = cbind(test_values[,1:8],test_values[,16:26])
test_data = cbind(test_data,test_values[,28:39])

#6 menghitung korelasi dengan metode Pearson Correlation
cor_matrix <- cor(train_data)

#7 check data numerical whether high correlated or not
heatmap(cor_matrix, 
        col = colorRampPalette(c("blue", "white", "green"))(1000))

## hasil yg terlihat high correlated: 
## height_percentage & count_floors_pre_eq
## has_secondary_use & has_secondary_use_agriculture

## check apakah high correlated or not (syarat: > 0.9)
cor(train_data$height_percentage, train_data$count_floors_pre_eq)
cor(train_data$has_secondary_use_agriculture, train_data$has_secondary_use)
## both tidak terlalu high correlated, maka dari itu datanya tidak dibuang.

#8 mengubah variabel categorical menjadi numerical 
## using one-hot encoding
### data_train
datatrain <- data.frame(
  train_values$land_surface_condition,
  train_values$foundation_type,
  train_values$roof_type,
  train_values$ground_floor_type,
  train_values$other_floor_type,
  train_values$position,
  train_values$plan_configuration,
  train_values$legal_ownership_status
)

recipe_obj = recipe(~., data = datatrain)%>%
  step_dummy(all_nominal())

data_encoded <- recipe_obj %>% prep() %>% bake(new_data = datatrain)

print(data_encoded)
### data_test
datatest <- data.frame(
  test_values$land_surface_condition,
  test_values$foundation_type,
  test_values$roof_type,
  test_values$ground_floor_type,
  test_values$other_floor_type,
  test_values$position,
  test_values$plan_configuration,
  test_values$legal_ownership_status
)

recipe_obj = recipe(~., data = datatest)%>%
  step_dummy(all_nominal())

data_encoded1 <- recipe_obj %>% prep() %>% bake(new_data = datatest)

print(data_encoded1)

#9 combine data_encoded with the train and test data
train_values = cbind(train_data, data_encoded)
test_values = cbind(test_data, data_encoded1)

#9 rename coloumn test_values data
colnames(train_values)[32:61] = c(1:30)
colnames(test_values)[32:61] = c(1:30)


#BUILDING MODEL ----------------------------------------------------------------

#using XGBoost method
xgb = xgboost(data = data.matrix(train_values[,-1]), 
              label = train_labels$damage_grade, 
              eta = 0.1,
              max_depth = 15, 
              nround=200, #jumlah trainnya
              subsample = 0.5,
              colsample_bytree = 0.5,
              eval_metric = "merror",
              objective = "multi:softmax",
              num_class = 12,
              nthread = 3)

#predict using train_values data
y_pred = predict(xgb, data.matrix(train_values[,-1]))
y_pred = factor(y_pred)

evaluation = confusionMatrix(data = y_pred, reference = train_labels$damage_grade, mode = "everything")
evaluation #F1-score test result di sini

# F1-score = 2*(precision*recall)/(precision + recall)
# F1 = TP/(TP + 0.5 (FP+FN))

#predict using test_values data
y_pred_1 = predict(xgb, data.matrix(test_values[,-1]))
y_pred_1 = factor(y_pred_1)
