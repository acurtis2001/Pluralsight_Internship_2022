library(caret)
library(data.table)
library(xgboost)
library(mlr3)
library(dplyr)

#Reading dataset from file 'production_dataset_build.R'
model_dataset <- read.csv("PAHTNAME")
#Read in predictions for joining
course_tags <- read.csv("PATHNAME")
course_tags$X <- NULL
course_tags <- course_tags %>%
  mutate(course_title = tolower(course_title)) %>%
  group_by(course_name, course_title) %>%
  distinct()
#Remove 0's view time predictions, remove actual demand values
final_model_preds <- model_dataset %>%
  filter(view_time_perc != 0) %>%
  mutate(log_view_time_perc = log(view_time_perc),
         quant_vt = as.factor(quant_vt)) %>%
  select(-demand_score, -search_cnt) 

final_model_preds <- final_model_preds %>%
  left_join(course_tags, by = c('course_name', 'course_title')) %>%
  left_join(external_preds, by = c('level_2' = 'l2', 'usage_year_month' = 'dt'))

final_model_preds$usage_year_month <- as.Date(final_model_preds$usage_year_month)

final_model_preds <- final_model_preds %>%
  left_join(internal_preds, by = c('level_3' = 'level_3', 'usage_year_month' = 'event_month')) %>%
  select(-level_2, -level_3)

final_model_preds$predictions <- NULL

final_model_preds$demand_score <- final_model_preds$predictions.x 
final_model_preds$search_cnt <- final_model_preds$predictions.y

final_model_preds$predictions.x <- NULL
final_model_preds$predictions.y <- NULL

## group by everything, take average of demand score, sum of search count
final_model_preds <- final_model_preds %>%
  select(everything()) %>%
  group_by_at(vars(-demand_score, -search_cnt)) %>%
  summarise(demand_score = mean(demand_score), search_cnt = sum(search_cnt)) %>%
  ungroup()
  
final_model_preds <- final_model_preds %>%
  filter(!is.na(final_model_preds$demand_score))

final_model_preds <- final_model_preds %>%
  filter(!is.na(final_model_preds$search_cnt))

### Generate predictions using demand predictions
set.seed(42069)
#Clean train data, now scaled so that code isn't reliant on indices
cleaned_train <- final_model_preds[,-c(grep("course_name|course_title|usage_year_month|published_date|platform_size|log_view_time_perc|view_time_perc", colnames(final_model_preds)))]
setDT(cleaned_train)
#Creation of model matrix with redundant columns removed
full_matrix <- model.matrix(~.+0,data = cleaned_train) 
full_matrix <- full_matrix[,xgb_model$feature_names]
# calculate predictions
final_model_preds$ac_predictions <- exp(predict(object = xgb_model, newdata = full_matrix))

mae(final_model_preds$ac_predictions, final_model_preds$view_time_perc)

final_model_preds$demand_preds <- final_model_preds$demand_score
final_model_preds$search_preds <- final_model_preds$search_cnt

final_model_preds$demand_score <- NULL
final_model_preds$search_cnt <- NULL

final_model_preds$pred_view_time_perc <- final_model_preds$ac_predictions
final_model_preds$ac_predictions <- NULL

final_model_preds <- final_model_preds %>%
  select(course_name, course_title, usage_year_month, view_time_perc, pred_view_time_perc, demand_preds, search_preds)

write.csv(final_model_preds, "PATHNAME", row.names = FALSE)
