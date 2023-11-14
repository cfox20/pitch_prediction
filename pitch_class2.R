
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
library('doParallel')8
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
#
# tb <- read_rds('data_list.rds')
#
# id <- map(tb, ~ .x$pitch_type %>% is.character()) %>% unlist()
# pitch_data <- bind_rows(tb[id])
#
# pitch_data <- pitch_data %>%
#   select(
#     pitch_type,release_speed,release_pos_x,release_pos_z,zone,
#     p_throws,balls,strikes,pfx_x,pfx_z,plate_x,plate_z,vx0,vy0,
#     vz0,ax,ay,az,release_spin_rate,spin_axis
#   ) %>%
#   drop_na() %>%
#   filter(pitch_type %in% c('CH','CU','FC','FF','FS','KC','SI','SL')) %>%
#   mutate(pitch_type_group = case_when(
#     pitch_type %in% c('FC','FF','FS','SI') ~ 1,
#     pitch_type %in% c('CU','KC','SL') ~ 2,
#     pitch_type == 'CH' ~ 3
#   )) %>%
#   mutate(pitch_type = case_when(
#     pitch_type == 'FF' ~ 1,
#     pitch_type == 'FC' ~ 2,
#     pitch_type == 'SI' ~ 3,
#     pitch_type == 'FS' ~ 4,
#     pitch_type == 'CH' ~ 5,
#     pitch_type %in% c('CU','KC') ~ 6,
#     pitch_type == 'SL' ~ 7
#   ))
#
# write_rds(tb, file = "data_list.rds")
# write_rds(pitch_data, file = "pitch_data.rds")



# Train Test Split --------------------------------------------------------
tic()
set.seed(123)
pitch_data <- read_rds('pitch_data.rds')

pitch_data <- pitch_data %>%
  # mutate(pitch_type_fb = pitch_type_group == "FB") %>%
  select(-c('pitch_type_group', 'zone', 'p_throws', 'balls', 'strikes'))

#If you do not have an ID per row, use the following code to create an ID
tb <- pitch_data %>% mutate(id = row_number()) %>% sample_frac(.25)
#Create training set
train <- tb %>% sample_frac(.7)
#Create test set
test  <- anti_join(tb, train, by = 'id') %>% select(-id)
train <- select(train, -id)





# Neural Network ----------------------------------------------------------

train_matrix <- model.matrix(pitch_type ~ . , train)[, -1] %>%
  scale()

train_label <- train %>% pull(pitch_type) %>% to_categorical() %>% .[,-1]

test_matrix <- model.matrix(pitch_type ~ . , test)[, -1] %>%
  scale()

test_label <- test %>% pull(pitch_type)

nn_mod <- keras_model_sequential() %>%
  layer_dense(units = 32, input_shape = ncol(train_matrix), activation = "relu") %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 7, activation = "softmax")

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

# write_rds(nn_mod, "models/nn_mod2.rds")

# nn_mod2 <- read_rds("models/nn_mod2.rds")


test_preds <- predict(nn_mod, test_matrix, type = "response")
test_pred_labels <- k_argmax(test_preds) %>% as.numeric() + 1

nn_acc <- mean(test_pred_labels == test_label)

nn_pitch_acc <- tibble(predicted = test_pred_labels, actual = test_label) %>%
  group_by(actual) %>%
  summarise(accuracy = mean(predicted == actual))



# write_rds(nn_pitch_acc, "Complex/nn_pitch_acc.rds")
# write_rds(nn_acc, "Complex/nn_acc.rds")







# K-Nearest Neighbor ------------------------------------------------------

train_label <- train %>% pull(pitch_type)

test_label <- test %>% pull(pitch_type)


knn_train <- train %>%
  select(-pitch_type)

knn_train_lab <- pull(train, pitch_type) %>% factor()

cl <- makePSOCKcluster(12)
registerDoParallel(cl)
knnFit <- caret::train(knn_train, knn_train_lab,
                       method = "knn",
                       preProcess = c("center", "scale"),
                       tuneLength = 10,
                       trControl = trainControl(method = "cv"))
stopCluster(cl)

env <- foreach:::.foreachGlobals
rm(list=ls(name=env), pos=env)


write_rds(knnFit, "models/knnFit2.rds")


knnFit2 <- read_rds("models/knnFit2.rds")

knn2_test_pred <- predict(knnFit2, newdata = test)

knn_acc <- mean(knn2_test_pred == test_label)

knn_pitch_acc <- tibble(Predicted = knn2_test_pred, Actual = test_label) %>%
  group_by(Actual) %>%
  summarise(Accuracy = mean(Predicted == Actual))


write_rds(knn_pitch_acc, "Complex/knn_pitch_acc.rds")
write_rds(knn_acc, "Complex/knn_acc.rds")






# Random Forest -----------------------------------------------------------


cl <- makePSOCKcluster(14)
registerDoParallel(cl)
ctrl <- trainControl(method="cv",
                     number = 10,
                     search = "random",
                     verboseIter = TRUE)
forestFit <- caret::train(factor(pitch_type) ~ ., data = train,
                          method = "rf",
                          trControl = ctrl,
                          tuneLength = 15,
                          metric = "Accuracy")
stopCluster(cl)

env <- foreach:::.foreachGlobals
rm(list=ls(name=env), pos=env)

write_rds(forestFit, "models/forestFit2.rds")

forestFit2 <- read_rds("models/forestFit2.rds")

forest_test_pred <- predict(forestFit2, newdata = test)

forest_acc <- mean(forest_test_pred == test_label)

forest_pitch_acc <- tibble(Predicted = forest_test_pred, Actual = test_label) %>%
  group_by(Actual) %>%
  summarise(Accuracy = mean(Predicted == Actual))


write_rds(forest_pitch_acc, "Complex/forest_pitch_acc.rds")
write_rds(forest_acc, "Complex/forest_acc.rds")


# Gradient Boosting -------------------------------------------------------

set.seed(123)
cl <- makePSOCKcluster(14)
registerDoParallel(cl)
ctrl <- trainControl(method="cv",
                     number = 10,
                     search = "random",
                     verboseIter = TRUE)
xgbFit <- caret::train(factor(pitch_type) ~ ., data = train,
                method = "xgbTree",
                trControl = ctrl,
                tuneLength = 15,
                metric = "Accuracy")

stopCluster(cl)

env <- foreach:::.foreachGlobals
rm(list=ls(name=env), pos=env)


write_rds(xgbFit, "models/xgbFit2.rds")

xgbFit2 <- read_rds("models/xgbFit2.rds")

xgb_test_pred <- predict(xgbFit2, newdata = test)

xgb_acc <- mean(xgb_test_pred == test_label)

xgb_pitch_acc <- tibble(Predicted = xgb_test_pred, Actual = test_label) %>%
  group_by(Actual) %>%
  summarise(Accuracy = mean(Predicted == Actual))


write_rds(xgb_pitch_acc, "Complex/xgb_pitch_acc.rds")
write_rds(xgb_acc, "Complex/xgb_acc.rds")



# Support Vector Machine --------------------------------------------------


cl <- makePSOCKcluster(12)
registerDoParallel(cl)
ctrl <- trainControl(method="cv",
                     number = 5,
                     search = "random",
                     verboseIter = TRUE)
svmFit <- caret::train(factor(pitch_type) ~ ., data = train,
                       method = "svmRadial",
                       trControl = ctrl,
                       tuneLength = 10,
                       metric = "Accuracy")
stopCluster(cl)

env <- foreach:::.foreachGlobals
rm(list=ls(name=env), pos=env)

# write_rds(svmFit, "models/svmFit2.rds")

svmFit <- read_rds("models/svmFit2.rds")

svm_pred <- predict(svmFit, newdata = test)

svm_acc <- mean(svm_pred == test_label)

svm_pitch_acc <- tibble(Predicted = svm_pred, Actual = test_label) %>%
  group_by(Actual) %>%
  summarise(Accuracy = mean(Predicted == Actual))


write_rds(xgb_pitch_acc, "Complex/svm_pitch_acc.rds")
write_rds(xgb_acc, "Complex/svm_acc.rds")


toc()


# Pitch Reproitoire Prediction -----------------------------------------

tb <- read_rds('data_list.rds')

id <- map(tb, ~ .x$pitch_type %>% is.character()) %>% unlist()
tb <- bind_rows(tb[id])

predict_pitch_rep <- function(k, player_data, player_kmeans) {
  player_data %>%
    mutate(cluster = player_kmeans[[k]]$cluster, pred = predict(xgbFit2, newdata = player_data)) %>%
    mutate(pred = factor(case_when(
      pred == 1 ~ 'FF',
      pred == 2 ~ 'FC',
      pred == 3 ~ 'SI',
      pred == 4 ~ 'FS',
      pred == 5 ~ 'CH',
      pred == 6 ~ 'CU',
      pred == 7 ~ 'SL'), levels = c('FF', 'FC', 'SI', 'FS', 'CH', 'CU', 'SL'))) %>%
    group_by(cluster, pred) %>%
    summarise(count = n()) %>%
    group_by(cluster) %>%
    mutate(perc = count/sum(count)) %>%
    ungroup() %>%
    select(-count) %>%
    pivot_wider(names_from = pred, values_from = perc, values_fill = 0)
}

pitch_rep <- function(first, last) {

  player_id <- baseballr::playerid_lookup(last_name = last, first_name = first) %>%
    dplyr::pull(mlbam_id)

  player_data <- tb %>%
    filter(pitcher == player_id) %>%
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
    )) %>%
    mutate(pitch_type = case_when(
      pitch_type == 'FF' ~ 1,
      pitch_type == 'FC' ~ 2,
      pitch_type == 'SI' ~ 3,
      pitch_type == 'FS' ~ 4,
      pitch_type == 'CH' ~ 5,
      pitch_type %in% c('CU','KC') ~ 6,
      pitch_type == 'SL' ~ 7)) %>%
    select(-c('pitch_type','pitch_type_group', 'zone', 'p_throws', 'balls', 'strikes'))


  player_kmeans <- map(1:7, ~ kmeans(player_data, centers = .x, nstart = 25, iter.max = 50))


  wss <- map_dbl(1:7, ~ player_kmeans[[.x]]$tot.withinss)

  kmeans_plot <- tibble(k = 1:7, wss = wss) %>%
    ggplot(aes(k, wss)) +
    geom_line() +
    geom_point()

  list(plot = kmeans_plot, pitch_pred = map(1:7, predict_pitch_rep, player_data, player_kmeans))
}


dylan_cease <- pitch_rep("Dylan", "Cease")
sandy_alcantara <- pitch_rep("Sandy", "Alcantara")
lucas_giolito <- pitch_rep("Lucas", "Giolito")

# write_rds(lucas_giolito, "lucas_giolito.rds")













