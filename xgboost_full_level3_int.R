
library(dplyr)
library(lubridate)
library(fastDummies)
library(randomForest)
library(Metrics)
library(ggplot2)
library(xgboost)
library(data.table)

### read in internal data with the LEVEL 3 TAGS !!!!!!!
internal <- read.csv("/Users/aubree-curtis@pluralsight.com/Downloads/internal_level3 (1).csv")

# get rid of row numbers column (if needed)
internal$X <- NULL

internal$event_month <- as.Date(internal$event_month)

#### Add nominal factor, quarter, and age ####

# nominal month
internal <- internal %>% mutate(month = month(event_month))

# quarter
internal <- internal %>% mutate(q = lubridate::quarter(event_month, with_year = FALSE))

# tag data age
internal <- internal %>%
  group_by(level_3) %>%
  mutate(earliest_date = min(event_month))

internal <- internal %>%
  group_by(event_month) %>%
  mutate(age = as.integer(lubridate::interval(as.Date(earliest_date), as.Date(event_month)) %/% months(1)) + 1) %>%
  ungroup()
  
int_temp <- internal %>%
  select(level_3, event_month, month, age) %>%
  mutate(month = as.factor(month)) 
  
internal <- internal %>%
  dplyr::select(-earliest_date, -event_month) %>%
  filter(search_cnt != 0) %>%
  mutate(month = as.factor(month), q = as.factor(q), level_3 = as.factor(level_3), search_cnt = search_cnt) %>%
  group_by(level_3, month, q, age) %>%
  summarise(search_cnt = sum(search_cnt))

internal <- internal %>%
  filter(month != 4)

#### Create Dummy Columns for Tags ####

#Generate dummy columns for  level 3 tags
internal <- fastDummies::dummy_cols(internal, select_columns = c('level_3'))
internal[is.na(internal)] <- 0

####.####

#### Split Train/Test ####

#Set seed for reproducibility
set.seed(42069)
#Train-test split
train_size <- floor(0.7 * nrow(internal))
train_ind <- sample(seq_len(nrow(internal)), size = train_size)
#Splitting train and test sets by index
train_data <- internal[train_ind,]
test_data <- internal[-train_ind,]
#Gather all features for train and test sets
int_train_data <- train_data %>%
  na.omit() %>%
  dplyr::select(-level_3)
int_test_data <- test_data %>%
  na.omit() %>%
  dplyr::select(-level_3)

####.####

#### XGBoost Training ####

# target extraction (search count)
int_train_search <- int_train_data$search_cnt
# clean train data
cleaned_int_train <- int_train_data[, -c(grep("search_cnt", colnames(int_train_data)))]
setDT(cleaned_int_train)
# creation of model matrix with redundant columns removed
tr_matrix <- model.matrix(~.+0, data = cleaned_int_train)
# data table creation for xgb.matrix formatting
dtrain <- xgb.DMatrix(data = tr_matrix, label = int_train_search)

# same thing for test data (below)

int_test_search <- internal$search_cnt

cleaned_int_test <- internal[, -1]
cleaned_int_test <- cleaned_int_test[, -c(grep("search_cnt", colnames(cleaned_int_test)))]                                    
setDT(cleaned_int_test)

test_matrix <- model.matrix(~.+0, data = cleaned_int_test)
dtest <- xgb.DMatrix(data = test_matrix, label = int_test_search)

# parameters
xgb_model_int_l3 <- xgb.train(
  eta = 0.1498432,
  booster = 'gbtree',
  objective = 'reg:squarederror',
  nthread = 0,
  eval_metric = 'rmse',
  early_stopping_rounds = 10,
  verbose = TRUE,
  watchlist = list(train = dtrain, eval = dtest),
  data = dtrain,
  nrounds = 5445,
  print_every_n = 100)

# run xgb model, assign predictions
internal$predictions <- predict(object = xgb_model_int_l3, dtest)
mae(internal$predictions, internal$search_cnt)

# residuals
internal$residuals <- internal$predictions - internal$search_cnt

# for nominal month and age
internal_preds <- internal %>%
  select(level_3, month, age, predictions)

# for event_month variable
internal_preds <- internal_preds %>%
  left_join(int_temp, by = c("level_3" = "level_3", "month" = "month", "age" = "age")) 

internal_preds <- internal_preds %>%
  select(level_3, event_month, predictions)

####.####

# write to csv with date variable
write.csv(internal_preds, "/Users/aubree-curtis@pluralsight.com/Downloads/internal_preds_date.csv", row.names = FALSE)

# write to csv with nominal month variable
write.csv(internal, "/Users/aubree-curtis@pluralsight.com/Downloads/internal_preds.csv", row.names = FALSE)



