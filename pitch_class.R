
# load packages -----------------------------------------------------------

library("here")
library("caret")
library("keras")
library("tensorflow")
library("reticulate")
library("baseballr")
library('fastDummies')
library('randomForest')
library('e1071')
library('furrr')
library('doParallel')
library('tictoc')
library("tidyverse"); theme_set(theme_minimal())



# Read Data ---------------------------------------------------------------

# dates <- c(seq(as.Date("2022-04-07"), as.Date("2022-06-17"), by="days"),
#   seq(as.Date("2022-06-22"), as.Date("2022-10-07"), by="days"))
#

# plan(multisession(workers = availableCores()))
# tb <- dates %>%
#   future_map(~ statcast_search(.x, .x), .progress = TRUE)
#
# plan(sequential)

 tb <- read_rds('data_list.rds')

 id <- map(tb, ~ .x$pitch_type %>% is.character()) %>% unlist()
 pitch_data <- bind_rows(tb[id])

 pitch_data <- pitch_data %>%
   select(
     pitch_type,release_speed,release_pos_x,release_pos_z,zone,
     p_throws,balls,strikes,pfx_x,pfx_z,plate_x,plate_z,vx0,vy0,
     vz0,ax,ay,az,release_spin_rate,spin_axis
   ) %>%
   drop_na() %>%
   filter(pitch_type %in% c('CH','CU','FC','FF','FS','KC','SI','SL')) %>%
   mutate(pitch_type_group = case_when(
     pitch_type %in% c('FC','FF','FS','SI') ~ 1,
     pitch_type %in% c('CU','KC','SL') ~ 2,
     pitch_type == 'CH' ~ 3
   ))
 # write_rds(tb, file = "data_list.rds")
 write_rds(pitch_data, file = "pitch_data_simple.rds")



# Train Test Split --------------------------------------------------------
set.seed(123)
pitch_data <- read_rds('pitch_data_simple.rds')

pitch_data <- pitch_data %>%
  # mutate(pitch_type_fb = pitch_type_group == "FB") %>%
  select(-c('pitch_type', 'zone', 'p_throws', 'balls', 'strikes'))

#If you do not have an ID per row, use the following code to create an ID
tb <- pitch_data %>% mutate(id = row_number()) %>% sample_frac(.25)
#Create training set
train <- tb %>% sample_frac(.7)
#Create test set
test  <- anti_join(tb, train, by = 'id') %>% select(-id)
train <- select(train, -id)


# EDA ---------------------------------------------------------------------

# pitch_data %>%
#   ggplot(aes(pitch_type_group)) +
#     geom_bar()
#
# pitch_data %>%
#   sample_frac(.1) %>%
#   ggplot(aes(release_spin_rate, spin_axis, color = as.factor(pitch_type_group))) +
#     geom_point(alpha = .25)
#
# pitch_data %>%
#   sample_frac(.1) %>%
#   ggplot(aes(release_speed, pfx_x, color =  as.factor(pitch_type_group))) +
#     geom_point(alpha = .25)




# Neural Network ----------------------------------------------------------
tic()

train_matrix <- model.matrix(pitch_type_group ~ . , train)[, -1] %>%
  scale()

train_label <- train %>% pull(pitch_type_group) %>% to_categorical() %>% .[,-1]

test_matrix <- model.matrix(pitch_type_group ~ . , test)[, -1] %>%
  scale()

test_label <- test %>% pull(pitch_type_group)

nn_mod <- keras_model_sequential() %>%
  layer_dense(units = 32, input_shape = ncol(train_matrix), activation = "relu") %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 3, activation = "softmax")

summary(nn_mod)


nn_mod %>% compile(loss = "categorical_crossentropy",
                   optimizer = "adam",
                   metrics = c("accuracy"))

trained_nn <- nn_mod %>% fit(
  x = train_matrix, # using for prediction
  y = train_label, # predicting
  batch_size = 50, # how many samples to pass to our model at a time
  epochs = 45, # how many times we'll look @ the whole dataset
  validation_split = 0.2) # how much data to hold out for testing as we go along

write_rds(nn_mod, "models/nn_mod.rds")

test_preds <- predict(nn_mod, test_matrix, type = "response")
test_pred_labels <- k_argmax(test_preds) %>% as.numeric() + 1

nn_acc <- mean(test_pred_labels == test_label)

nn_pitch_acc <- tibble(predicted = test_pred_labels, actual = test_label) %>%
  group_by(actual) %>%
  summarise(accuracy = mean(predicted == actual))



write_rds(nn_pitch_acc, "Simple/nn_pitch_acc.rds")
write_rds(nn_acc, "Simple/nn_acc.rds")


















# K-Nearest Neighbor ------------------------------------------------------
knn_train <- train %>%
  select(-pitch_type_group)

knn_train_lab <- pull(train, pitch_type_group) %>% factor()

cl <- makePSOCKcluster(15)
registerDoParallel(cl)
knnFit <- caret::train(knn_train, knn_train_lab,
                 method = "knn",
                 preProcess = c("center", "scale"),
                 tuneLength = 10,
                 trControl = trainControl(method = "cv"))
stopCluster(cl)

env <- foreach:::.foreachGlobals
rm(list=ls(name=env), pos=env)


write_rds(knnFit, "models/knnFit.rds")


knnFit <- read_rds("models/knnFit.rds")

knn_test_pred <- predict(knnFit, newdata = test)

knn_acc <- mean(knn_test_pred == test_label)

knn_pitch_acc <- tibble(Predicted = knn_test_pred, Actual = test_label) %>%
  group_by(Actual) %>%
  summarise(Accuracy = mean(Predicted == Actual))


write_rds(knn_pitch_acc, "Simple/knn_pitch_acc.rds")
write_rds(knn_acc, "Simple/knn_acc.rds")











# Random Forest -----------------------------------------------------------


cl <- makePSOCKcluster(15)
registerDoParallel(cl)
ctrl <- trainControl(method="cv",
                     number = 10,
                     search = "random",
                     verboseIter = TRUE)
forestFit <- caret::train(factor(pitch_type_group) ~ ., data = train,
                   method = "rf",
                   trControl = ctrl,
                   tuneLength = 15,
                   metric = "Accuracy")
stopCluster(cl)

env <- foreach:::.foreachGlobals
rm(list=ls(name=env), pos=env)

write_rds(forestFit, "models/forestFit.rds")



forestFit <- read_rds("models/forestFit.rds")

forest_test_pred <- predict(forestFit, newdata = test)

forest_acc <- mean(forest_test_pred == test_label)

forest_pitch_acc <- tibble(Predicted = forest_test_pred, Actual = test_label) %>%
  group_by(Actual) %>%
  summarise(Accuracy = mean(Predicted == Actual))


write_rds(forest_pitch_acc, "Simple/forest_pitch_acc.rds")
write_rds(forest_acc, "Simple/forest_acc.rds")

toc()



# XGBoost ----------------------------------------------------------

set.seed(123)
cl <- makePSOCKcluster(12)
registerDoParallel(cl)
ctrl <- trainControl(method="cv",
                     number = 10,
                     search = "random",
                     verboseIter = TRUE)
xgbFit <- caret::train(factor(pitch_type_group) ~ ., data = train,
                       method = "xgbTree",
                       trControl = ctrl,
                       tuneLength = 15,
                       metric = "Accuracy")

stopCluster(cl)

env <- foreach:::.foreachGlobals
rm(list=ls(name=env), pos=env)


write_rds(xgbFit, "models/xgbFit.rds")

xgbFit <- read_rds("models/xgbFit.rds")

xgb_test_pred <- predict(xgbFit, newdata = test)

xgb_acc <- mean(xgb_test_pred == test_label)

xgb_pitch_acc <- tibble(Predicted = xgb_test_pred, Actual = test_label) %>%
  group_by(Actual) %>%
  summarise(Accuracy = mean(Predicted == Actual))


write_rds(xgb_pitch_acc, "Simple/xgb_pitch_acc.rds")
write_rds(xgb_acc, "Simple/xgb_acc.rds")








# Support Vector Machine --------------------------------------------------



cl <- makePSOCKcluster(15)
registerDoParallel(cl)
ctrl <- trainControl(method="cv",
                     number = 50,
                     search = "random",
                     verboseIter = TRUE)
svmFit <- caret::train(factor(pitch_type_group) ~ ., data = train,
                          method = "svmRadial",
                          trControl = ctrl,
                          tuneLength = 10,
                          metric = "Accuracy")
stopCluster(cl)

env <- foreach:::.foreachGlobals
rm(list=ls(name=env), pos=env)

write_rds(svmFit, "models/svmFit.rds")

svmFit <- read_rds("models/svmFit.rds")

svm_pred <- predict(svmFit, newdata = test)

svm_acc <- mean(svm_pred == test_label)

svm_pitch_acc <- tibble(Predicted = svm_pred, Actual = test_label) %>%
  group_by(Actual) %>%
  summarise(Accuracy = mean(Predicted == Actual))


write_rds(xgb_pitch_acc, "Simple/svm_pitch_acc.rds")
write_rds(xgb_acc, "Simple/svm_acc.rds")


toc()













