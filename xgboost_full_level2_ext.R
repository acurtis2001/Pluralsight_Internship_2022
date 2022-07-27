library(dplyr)
library(lubridate)
library(fastDummies)
library(randomForest)
library(Metrics)
library(ggplot2)
library(xgboost)
library(data.table)

### read in external data WITH LEVEL 2 TAGS !!!!!
external <- read.csv("/Users/aubree-curtis@pluralsight.com/Downloads/external_level2.csv")

# remove row names (if applicable)
external$X <- NULL

# rename avg_demand_normalized
external <- external %>%
  dplyr::rename(demand = demand_score, l2 = level_2)

#### Add nominal factor, quarter, and age ####

# nominal month
external <- external %>% mutate(month = month(dt))
# save month and date to temp
ext_temp <- external %>%
  select(l2, dt, month) %>%
  mutate(month = as.factor(month))

# quarter
external <- external %>% mutate(q = lubridate::quarter(dt, with_year = FALSE))

# tag data age
external <- external %>%
  group_by(l2) %>%
  mutate(earliest_date = min(dt))

external <- external %>%
  group_by(dt) %>%
  mutate(age = as.integer(lubridate::interval(as.Date(earliest_date), as.Date(dt)) %/% months(1)) + 1) %>%
  ungroup() %>%
  dplyr::select(-earliest_date, -dt) %>%
  filter(demand != 0) %>%
  mutate(month = as.factor(month), q = as.factor(q), l2 = as.factor(l2), demand = demand)

external_pre_dummy <- external

#### Create Dummy Columns for Level 2 Tags ####

external <- fastDummies::dummy_cols(external, select_columns = c('l2'))
external[is.na(external)] <- 0

####.####

#### Split Train/Test ####

## External
#Set seed for reproducibility
set.seed(42069)
#Train-test split
train_size <- floor(0.7 * nrow(external))
train_ind <- sample(seq_len(nrow(external)), size = train_size)
#Splitting train and test sets by index
train_data <- external[train_ind, ]
test_data <- external[-train_ind, ]
#Gather all features for train and test sets
ext_train_data <- train_data %>%
  na.omit() %>%
  dplyr::select(-l2)
ext_test_data <- test_data %>%
  na.omit() %>%
  dplyr::select(-l2)

####.####

#### XGBoost Training ####

# target extraction (demand score)
ext_train_demand <- ext_train_data$demand
# clean train data
cleaned_ext_train <- ext_train_data[, -c(grep("demand", colnames(ext_train_data)))]
setDT(cleaned_ext_train)
# creation of model matrix with redundant columns removed
tr_matrix <- model.matrix(~.+0, data = cleaned_ext_train)
# data table creation for xgb.matrix formatting
dtrain <- xgb.DMatrix(data = tr_matrix, label = ext_train_demand)

# same thing for test data
ext_test_demand <- external$demand

cleaned_ext_test <- external[, -1]
cleaned_ext_test <- cleaned_ext_test[, -c(grep("demand", colnames(cleaned_ext_test)))]                                    
setDT(cleaned_ext_test)

test_matrix <- model.matrix(~.+0, data = cleaned_ext_test)
dtest <- xgb.DMatrix(data = test_matrix, label = ext_test_demand)

# parameters
xgb_model_ext_l2 <- xgb.train(
  eta = 0.1498432,
  booster = 'gbtree',
  objective = "reg:logistic",
  #Maximum number of available threads to train with.
  nthread = 0,
  eval_metric = 'rmse',
  early_stopping_rounds = 40,
  verbose = TRUE,
  watchlist = list(train = dtrain, eval = dtest),
  data = dtrain,
  nrounds = 5445,
  print_every_n = 100)

## run xgboost l2 model, assign to predictions column
external$predictions <- predict(object = xgb_model_ext_l2, dtest)
mae(external$predictions, external$demand)

# residuals
external$residuals <- external$predictions - external$demand

# for nominal month
external_preds <- external %>%
  select(l2, month, predictions)

# for dt variable
external_preds <- external_preds %>%
  left_join(ext_temp, by = c("l2" = "l2", "month" = "month")) 

external_preds <- external_preds %>%
  select(l2, dt, predictions)

####.####

# write to csv with dt variable
write.csv(external_preds, "/Users/aubree-curtis@pluralsight.com/Downloads/external_preds_date.csv", row.names = FALSE)

# write to csv with nominal month variable
write.csv(external, "/Users/aubree-curtis@pluralsight.com/Downloads/external_preds.csv", row.names = FALSE)





