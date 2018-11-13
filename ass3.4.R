## Install packages
#install.packages("keras")
#install_keras()
#install.packages('curl')
#install.packages("tm")
#install.packages("qdap")

library(keras)
install_keras(tensorflow = "gpu")
library(tm)
library(ggplot2)

set.seed(12345)

setwd("~/Uni/2018-2/FIT5149 Applied Data Analysis/Assignment")


## Load Data
# Load text
text <- readLines("training_docs.txt")
text <- as.data.frame(text)
text <- as.data.frame(text[!(text$text=="EOD"),], header = FALSE)
colnames(text) <- c("text")
#text <- text[!grepl("ID te_doc_",text$text),] # For test dataset
text <- text[!grepl("ID tr_doc_",text$text),] # For train dataset
text <- as.data.frame(text)
text <- as.data.frame(text[!(text$text==""),],header= FALSE)
colnames(text) <- c("text")
text <- lapply(text,function(x) gsub("TEXT ","",x))
text <- as.data.frame(text)

# Load labels
label <- as.data.frame(readLines("training_labels_final.txt"))
label$labels <- sub(".* ", "",label$`readLines("training_labels_final.txt")`)
label$`readLines("training_labels_final.txt")` <- sub(" .*", "",label$`readLines("training_labels_final.txt")`)
colnames(label) <- c("doc","label")


## Initialise
# Word vector
maxlen <- 200 # Input documents have 200 words each
max_words <- 50000  # Size of the featues in the text data. original = 50000
embedding_dim <- 200 # Dim size of embedding matrix


## Custom Functions
# Calculate F score
F_score <- function(cm_table){
  
  # Recall
  my_recall <- vector(mode="numeric", length=23)
  for (i in 1:23){
    my_recall[i] <- cm_table[i,i]/sum(cm_table[,i]) # Uses Confusion matrix
  }
  
  # Precision
  my_precision <- vector(mode="numeric", length=23)
  for (i in 1:23){
    my_precision[i] <- cm_table[i,i]/sum(cm_table[i,])
  }
  
  my_F1 <- vector(mode="numeric", length=23)
  for (i in 1:23){
    # Calculates based on formula in assignment details
    my_F1[i] <- 2*((my_recall[i]*my_precision[i]) / (my_recall[i]+my_precision[i]))
  }
  Final_F1 <- mean(my_F1)
  
  return (Final_F1)
  
} 

# Class Accuracy
class_accuracy <- function(cm_table){
  for (i in 1:ncol(cm_table)){
    # Takes confusion matrix as input, find accuracy of the model for each class
    class_acc[i,2] <- (sum(cm_table[i,i]) / sum(cm_table[,i]))
  }
  return (class_acc)
}

# Early stopping
# If validation accuracy does not improve, traning terminates 
es_callback <- callback_early_stopping(monitor='val_acc', min_delta=0, patience=3, verbose=0)


# Compile + Fitting model
# Compiles the model and fits, returns history(results of the training)
compile_fit <- function(model, epochs=10, batch=32, split_ratio=0.2, optimizer="rmsprop"){
  model %>% compile(
    optimizer = optimizer,
    loss = "categorical_crossentropy",
    metrics = c("acc")
  )
  history <- model %>% fit(
    x_train, y_train,
    epochs = epochs,
    batch_size = batch,
    callbacks = list(es_callback),
    validation_split = split_ratio,
    shuffle = FALSE # For reproducible result
  )
  return (history)
}



# Ensemble
# Emsemble function takes validation data, weights as input
# Apply input weights to each model, generates weighted average prediction
ensemble <- function(val_data = 'x_val', w = c(0.5, 0.30, 0.20)){
  
  # Initialise
  N <- nrow(val_data)
  w = w # weighting

  # Reset existing
  model1_pred <- 0
  model2_pred <- 0
  model3_pred <- 0
  
  # Probability storing
  model1_pred <- predict_proba(object = cbi_lstm, x = val_data)
  model2_pred <- predict_proba(object = conv_pool_cnn, x = val_data)
  model3_pred <- predict_proba(object = model_test, x=val_data)
  
  # Weighted average
  cs <-(w[1]*model1_pred[1:N,]+w[2]*model2_pred[1:N,]+w[3]*model3_pred[1:N,])
  
  # Return prediction
  # Maximum probability will be chosen for each class
  return(apply(cs, MARGIN = 1,FUN = which.max))
}




# Pre-processing text
corpus <- Corpus(VectorSource(text$text))
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeWords, stopwords('english'))
corpus <- tm_map(corpus, content_transformer(stripWhitespace))
cleaned.docs = data.frame(text=sapply(corpus,identity), stringsAsFactors = F)

head(cleaned.docs,1)

# Only used for overviewing the features
# DTM_train <- DocumentTermMatrix(corpus, control=list(wordLengths=c(4,Inf)))
# DTM_train # Number of features = terms: 159389
# DTM_train <- removeSparseTerms(DTM_train,0.95)
# DTM_train # Number of features = terms: 493
# tfm <- weightTfIdf(TermDocumentMatrix(corpus[training_indices]))

# Tokenise texts
texts = cleaned.docs$text
tokenizer <- text_tokenizer(num_words = max_words) %>% # max_word = most common words
    fit_text_tokenizer(texts)

# Gether the sequences
sequences <- texts_to_sequences(tokenizer, texts)

# Gether word index
word_index = tokenizer$word_index

# Pad sequences to datafarme
data <- pad_sequences(sequences, maxlen = maxlen)

# Clean labels
labels = as.array(label$label)
labels = as.numeric(gsub("C", "", labels))


## Data Split
# Random selection
set.seed(12345) # Just to make sure seed is the same for reproducible output
indices <- sample(1:nrow(data))
training_samples = 100000
training_indices <- indices[1:training_samples]
validation_indices <- indices[(training_samples + 1):nrow(data)]

# Split into train / validation
x_train <- data[training_indices,]
y_train <- labels[training_indices]
y_train <- y_train-1 # To make class category 1:23
y_train = to_categorical(y_train,num_classes = length(unique(y_train)))
x_val <- data[validation_indices,]
y_val <- labels[validation_indices]



########### Model 1 [ Convolutional Bi-directional LSTM ] ##############
# 0.7527 epoch 3, 100k data
# 0.7607 epoch 3, 100k data
# 0.7403 epoch 3, 70k data
cbi_lstm <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words, output_dim = embedding_dim,
                  input_length = maxlen) %>%
  layer_conv_1d(filters = 1000, kernel_size = 4, activation = 'relu') %>%
  layer_max_pooling_1d(pool_size=2) %>%
  layer_spatial_dropout_1d(0.5) %>%
  bidirectional(layer_cudnn_lstm(units = 500, return_sequences = TRUE)) %>%
  layer_dropout(0.25) %>%
  bidirectional(layer_cudnn_lstm(units = 250, return_sequences = TRUE)) %>%
  layer_dropout(0.25) %>%
  layer_flatten() %>%
  layer_dense(units = 23, activation = "softmax")
summary(cbi_lstm)

compile_fit(cbi_lstm, epochs = 2)

F_score(table(y_val, predict_classes(cbi_lstm, x_val)))


# # Saving Model
# Model Saved / Loaded in current working directory
save_model_hdf5(cbi_lstm, "cbi_lstm_model", overwrite = TRUE,
                include_optimizer = TRUE)
# # Loading Model
cbi_lstm <- load_model_hdf5('cbi_lstm_model')





########### Model 2 [ ConvPool-CNN-C ] 0.7188 ##############
# 0.7265 epoch 3, 100k data
# 0.7279 epoch 3, 100k data
# 0.6962 epoch 3, 70k data
conv_pool_cnn <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words, output_dim = embedding_dim,
                  input_length = maxlen) %>%
  layer_conv_1d(filters = 96, kernel_size=3, activation='relu') %>%
  layer_conv_1d(filters = 96, kernel_size = 3,  activation='relu') %>%
  layer_max_pooling_1d(pool_size=3) %>%
  layer_conv_1d(filters = 192, kernel_size = 3, activation='relu', padding = 'same') %>%
  layer_conv_1d(filters = 192, kernel_size = 3, activation='relu', padding = 'same') %>%
  layer_max_pooling_1d(pool_size=3) %>%
  layer_conv_1d(filters = 192, kernel_size = 3, activation='relu', padding = 'same') %>%
  layer_conv_1d(filters = 192, kernel_size = 1, activation='relu') %>%
  layer_conv_1d(filters = 23, kernel_size = 1) %>%
  layer_global_max_pooling_1d() %>%
  layer_activation(23, activation='softmax')
summary(conv_pool_cnn)

compile_fit(conv_pool_cnn, epochs = 3)

F_score(table(y_val, predict_classes(conv_pool_cnn, x_val)))

# # Saving Model
# Model Saved / Loaded in current working directory
save_model_hdf5(cbi_lstm, "conv_pool_cnn", overwrite = TRUE,
                include_optimizer = TRUE)
# # Loading Model
conv_pool_cnn <- load_model_hdf5('conv_pool_cnn')






########### Model 3 [ LSTM Conv Net ] 0.7361 ##############
# 0.7456 epoch 3, 100k data
# 0.7394 epoch 4, 100k data
# 0.7234 epoch 3, 70k data
model_test <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words, output_dim = embedding_dim,
                  input_length = maxlen) %>%
  layer_spatial_dropout_1d(0.2) %>%
  layer_cudnn_lstm(units = 128, return_sequences = TRUE) %>%
  layer_spatial_dropout_1d(0.5) %>%
  layer_conv_1d(filters = 64, kernel_size = 2, activation = 'relu') %>%
  layer_average_pooling_1d(pool_size = 2) %>%
  layer_max_pooling_1d(pool_size = 2) %>%
  layer_spatial_dropout_1d(0.2) %>%
  layer_flatten() %>%
  layer_dense(units = 23, activation = "softmax")
summary(model_test)

compile_fit(model_test, 4)

F_score(table(y_val, predict_classes(model_test, x_val)))



## Ensemble ##
en_prediction <- ensemble(val_data=x_val, w=c(0.5,0.3,0.2))

f_ensemble <- F_score(table(y_val, en_prediction))
f_ensemble

# 0.7586 - 80k data - seed 1234 'Actual Test run' was 0.757
# 0.7652 - 100k data, seed 12345 c(0.5,0.3,0.2) 
# 0.7652 - 100k data, seed 12345
# w=c(0.5,0.2,0.3) 0.7531831

# Current best model 
# seed 12345, w = c(0.5,0.3,0.2) F = 0.7655
table(y_val, en_prediction)





# # Saving Model
# save_model_hdf5(cbi_lstm, "./Model/blcnn_model", overwrite = TRUE,
#                 include_optimizer = TRUE)
# save_model_hdf5(conv_pool_cnn, "./Model/conv_pool_cnn_model", overwrite = TRUE,
#                 include_optimizer = TRUE)
# # Loading Model
# cbi_lstm <- load_model_hdf5('./Model/blcnn_model')
# conv_pool_cnn <- load_model_hdf5('./Model/conv_pool_cnn_model')








#prediction
##create model to compare accuracy of each model to each classes
final_models_compare = data.frame(1:23)
final_models_compare$model1_pred = model1_acc$V2
final_models_compare$model2_pred = model2_acc$V2
final_models_compare$model3_pred = model3_acc$V2
model_per_sample_size <- final_models_compare
model_per_sample_size[5] <- data.frame(sample_size=50000)
model_per_sample_size


## make prediction for each models
model1_pred <- predict_proba(object = cbi_lstm, x = x_val)
model2_pred <- predict_proba(object = conv_pool_cnn, x = x_val)
model3_pred <- predict_proba(object = model_test, x=x_val)

## select the model for each class that model is best on predicting that class
model_choice = apply(data.frame(final_models_compare$model1_pred,final_models_compare$model2_pred,final_models_compare$model3_pred), MARGIN = 1,FUN = which.max)
##make predictions
model1_choice = apply(model1_pred[,1:23], MARGIN = 1,FUN = which.max)
model2_choice = apply(model2_pred[,1:23], MARGIN = 1,FUN = which.max)
model3_choice = apply(model3_pred[,1:23], MARGIN = 1,FUN = which.max)
model_choice

final_models_compare

model_per_sample_size <- final_models_compare
model_per_sample_size[5] <- data.frame(sample_size=50000)
model_per_sample_size



final_model_choice =data.frame(1:length(model1_choice))
final_model_choice$m1c = 0
final_model_choice$m1c[which(model1_choice %in% which(model_choice==1))] = model1_choice[which(model1_choice %in% which(model_choice==1))]

final_model_choice

which(model1_choice %in% which(model_choice==1))


final_model_choice$m2c = 0
final_model_choice$m2c[which(model2_choice %in% which(model_choice==2))] = model2_choice[which(model2_choice %in% which(model_choice==2))]

final_model_choice$m3c = 0
final_model_choice$m3c[which(model3_choice %in% which(model_choice==3))] = model3_choice[which(model3_choice %in% which(model_choice==3))]

final_model_choice$mfc = 0
final_model_choice
max_compare = function(row){
  if(row[1]==0 && row[2] ==0 && row[3]==0){
    row[4] = model1_choice[row]
  }
  else if(row[1]==0 && row[2]==0){
    row[4] = row[3]
  }
  else if(row[2]==0 && row[3]==0){
    row[4] = row[1]
  }
  else if(row[1]==0 && row[3]==0){
    row[4] = row[2]
  }
}

model1_choice


# Best model calss method
N <- length(model1_choice)
N

for (row in 1:N){
  final_model_choice[row,5] <- ifelse(max(final_model_choice[row,2:4])==0, model1_choice[row], 
                                      ifelse(length(which(final_model_choice[200,2:4] != 0))>1, 0))
} 

final_model_choice[,5]
final_model_choice

length(which(final_model_choice[200,2:4] != 0))

final_models_compare[row[2],3]
final_models_compare[row[4],5]

which(max(final_models_compare[final_model_choice[200,2],3],
          final_models_compare[final_model_choice[200,3],4],
          final_models_compare[final_model_choice[200,4],5]))

a <- final_model_choice[200,2:4]
a

which.max(c(final_models_compare[a$m1c,3], final_models_compare[a$m2c,4], final_models_compare[a$m3c,5]))


final_models_compare[a$m1c,3]


which.max(c(final_models_compare[final_model_choice[200,2],3],
            final_models_compare[final_model_choice[200,3],4],
            final_models_compare[final_model_choice[200,4],5]))


final_models_compare[final_model_choice[200,2],3]
final_models_compare[final_model_choice[200,3],4]
final_models_compare[final_model_choice[200,4],5]


final_models_compare

final_model_choice[200,]


final_models_compare

y_val




final_choice_model <- final_model_choice[,5]
F_score(table(y_val, final_choice_model))
table(y_val, final_choice_model)



en_prediction <- ensemble(val_data=x_val, w=c(0.5,0.3,0.2))
F_score(table(y_val, en_prediction))
table(y_val, en_prediction)



apply(final_model_choice[,c("m1c","m2c","m3c","mfc")], MARGIN = 1, FUN=max_compare)


model1_pred_c <- predict_classes(object = cbi_lstm, x = x_val)
model2_pred_c <- predict_classes(object = conv_pool_cnn, x = x_val)
model3_pred_c <- predict_classes(object = model_test, x=x_val)

model_full <- data.frame(model1_pred_c,model2_pred_c,model3_pred_c)

model_full




















##############  Evaluation  #################
# cbi_lstm %>% evaluate(x_val, y_val)
# conv_pool_cnn %>% evaluate(x_val, y_val)


# Evaluation M1
#history
class_acc <-data.frame('Class'=(1:23), 'Accuracy'=rep(0,23))

# With Validation data
Y_test_predicted1 <- predict_classes(cbi_lstm, x_val)
cm_table_cbi_lstm <- table(y_val, Y_test_predicted1)
cm_table_cbi_lstm
class_accuracy(cm_table_cbi_lstm)
F_score(cm_table_cbi_lstm)
cbi_lstm_model_result <- class_accuracy(cm_table_cbi_lstm)


# Evaluation M2
class_acc <-data.frame('Class'=(1:23), 'Accuracy'=rep(0,23))

# With Validation data
Y_test_predicted2 <- predict_classes(conv_pool_cnn, x_val)
cm_table_conv_cnn <- table(y_val, Y_test_predicted2)
cm_table_conv_cnn
class_accuracy(cm_table_conv_cnn)
F_score(cm_table_conv_cnn)
conv_model_result <- class_accuracy(cm_table_conv_cnn)


# Evaluation M3
class_acc <-data.frame('Class'=(1:23), 'Accuracy'=rep(0,23))

# With Validation data
Y_test_predicted3 <- predict_classes(model_test, x_val)
cm_table_model_test <- table(y_val, Y_test_predicted3)
cm_table_model_test
class_accuracy(cm_table_model_test)
F_score(cm_table_model_test)



# Combined
run5 <- cbind(cbi_lstm_model_result, conv_model_result)
class_comb <- cbind(class_accuracy(cm_table_cbi_lstm), class_accuracy(cm_table_conv_cnn), class_accuracy(cm_table_model_test))
class_comb


cb_m <- as.numeric(gsub('%','',run1[,2])) + as.numeric(gsub('%','',run2[,2])) + as.numeric(gsub('%','',run3[,2]))+ as.numeric(gsub('%','',run4[,2]))+ as.numeric(gsub('%','',run5[,2]))
conv_m <- as.numeric(gsub('%','',run1[,4])) + as.numeric(gsub('%','',run2[,4])) + as.numeric(gsub('%','',run3[,4])) + as.numeric(gsub('%','',run4[,4]))+ as.numeric(gsub('%','',run5[,2]))
cbind(bl_cnn=cb_m, conv=conv_m, outperform=(conv_m-bc_m)/5)

#####



## Finalising Output
text <- readLines("testing_docs.txt")
text <- as.data.frame(text)
text <- as.data.frame(text[!(text$text=="EOD"),], header = FALSE)
colnames(text) <- c("text")
text <- text[!grepl("ID te_doc_",text$text),] # For test dataset
text <- as.data.frame(text)
text <- as.data.frame(text[!(text$text==""),],header= FALSE)
colnames(text) <- c("text")
text <- lapply(text,function(x) gsub("TEXT ","",x))
text <- as.data.frame(text)

# Pre-processing text
corpus <- Corpus(VectorSource(text$text))
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeWords, stopwords('english'))
#corpus <- tm_map(corpus, content_transformer(stemDocument)) # Stemming didn't help
corpus <- tm_map(corpus, content_transformer(stripWhitespace))
cleaned.docs = data.frame(text=sapply(corpus,identity), stringsAsFactors = F)

head(cleaned.docs,1)



### Build Input Data
texts = cleaned.docs$text

# Gether the sequences
sequences <- texts_to_sequences(tokenizer, texts)

# Gether word index
word_index = tokenizer$word_index

# Pad sequences to datafarme
test_data <- pad_sequences(sequences, maxlen = maxlen)

labels = as.array(label$label)
labels = as.numeric(gsub("C", "", labels))


# Prediction to output
final_prediction <- ensemble(val_data=test_data, w=c(0.6,0.2,0.2))

# Read Labels
testing_label <- readLines("testing_docs.txt")
Y_test_predicted<-data.frame(final_prediction)

doc <- data.frame(testing_label)
doc <- data.frame(label=testing_label[seq(from=1,to=nrow(doc),by=4)])
doc <- lapply(doc, function(x) gsub("ID ","",x))
doc <- data.frame(doc, stringsAsFactors = F)

for (line in 1:nrow(doc)){
  doc[line,] <- paste0(doc[line,], " C", Y_test_predicted[line,])
}
head(doc,2)


testing_labels_final1 <- write.table(doc, "C:/Users/abcd0/Documents/Uni/2018-2/FIT5149 Applied Data Analysis/Assignment/testing_labels_final1.txt", sep="\t", row.names = FALSE, quote = FALSE, col.names = FALSE)







######################################### BELOW IS FOR TESTING #################################################################


# Training samples vs accuracy
#model_per_sample_size_f <- data.frame(size=c("10K","30K","50K","70K","100K"),f_ensemble=c(0,0,0,0,f_ensemble))
model_per_sample_size_f[1,2] <- f_ensemble
model_per_sample_size_f

final_models_compare = data.frame(1:23)
final_models_compare$model1_pred = model1_acc$V2
final_models_compare$model2_pred = model2_acc$V2
final_models_compare$model3_pred = model3_acc$V2
model_per_sample_size <- final_models_compare
model_per_sample_size[6] <- f_ensemble
model_per_sample_size[6] <- data.frame(sample_size=50000)
model_per_sample_size
model_per_sample_size_f <- data.frame(size=c("10K","30K","50K","70K","100K"),f_ensemble=c(0,0,0,0,f_ensemble))
model_per_sample_size_f[4,2] <- f_ensemble
model_per_sample_size_f

model_vs_sample <- model_per_sample_size_f[2:5,]
model_vs_sample

model_vs_sample[1,1:2] <- model_per_sample_size_f[2,1:2]
model_vs_sample[2,1:2] <- model_per_sample_size_f[3,1:2]
model_vs_sample[3,1:2] <- model_per_sample_size_f[4,1:2]
model_vs_sample[4,1:2] <- model_per_sample_size_f[1,1:2]
model_vs_sample[5,1:2] <- model_per_sample_size_f[5,1:2]


model_vs_sample[1,1:2] <- model_vs_sample[1:4]
rownames(model_vs_sample) <- c(1,2,3,4)
model_vs_sample$size<-c(30,50,70,90,100)
model_vs_sample

ggplot(data=model_vs_sample)+aes(x=size, y=f_ensemble, group=1)+geom_line()+theme_minimal()+ylab("F Score")+xlab("Size of Training set '000'")


########





testing_labels_final <- readLines("training_labels_final.txt")
testing_labels_final1 <- readLines("testing_labels_final1.txt")

sum(cbind(testing_labels_final==testing_labels_final1))/length(testing_labels_final)

head(testing_labels_final)
head(testing_labels_final1)




head(doc)

Y_test_predicted <- data.frame(Y_test_predicted)
head(Y_test_predicted)
tail(Y_test_predicted)

cbind(doc,Y_test_predicted)


head(doc)




head(Y_test_predicted)
head(training_labels_final)


doc <- lapply(training_labels_final,function(x) gsub("C1","",x))

head(doc,1)



head(doc)



# F1
cm_table
history
F_score(cm_table)

class_accuracy(cm_table)







### Below needs to be completed

#data.correct <- data.val[which(data.val$y==data.val$predicted),]


# With train data for training model 2 use
Y_test_model1 <- predict_classes(model, x_train)
y_train <- labels[training_indices]
cm_table_m1 <- table(y_train, Y_test_model1)
cm_table_m1


data.val <- data.frame(y=y_train,x=x_train, predicted=Y_test_model1)
#data.correct <- data.val[which(data.val$y==data.val$predicted),]

head(data.val,2)




########### Model 2 ##############
# Create missclassified set
data.miss <- data.val[which(data.val$y!=data.val$predicted),]
data.miss <- data.miss[,1:101]

x_train_miss <- as.matrix(data.miss[,2:101])
y_train_miss <- as.matrix(data.miss[,1])

# Revert back to text
text_2 <- data.frame(y=y_train_miss, tokenizer$sequences_to_texts(x_train_miss))

# For each category ...
c2 <- text_2[text_2$y==2,]




#define model2
model2 <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words, output_dim = embedding_dim,
                  input_length = maxlen) %>%
  layer_flatten() %>%
  layer_dense(units = 1000, activation = "relu") %>%
  layer_dense(units = 500, activation = "relu") %>%
  layer_dense(units = 50, activation = "relu") %>%
  layer_dense(units = 50, activation = "relu") %>%
  layer_dense(units = 24, activation = "sigmoid")
summary(model)

# Compile model
model2 %>% compile(
  optimizer = "adam",
  loss = "categorical_crossentropy",
  metrics = c("acc")
)

# Early stopping
callback <- callback_early_stopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='min')

# Fitting model
history2 <- model2 %>% fit(
  x_train_miss, y_train_miss,
  epochs = 20,
  batch_size = 32,
  validation_split = (0.1)
  # callbacks = callback
)


model %>% evaluate(x_train_miss, y_train_miss)

Y_test_hat2 <- predict_classes(model2, x_train_miss)
cm_table2 <- table(data.miss[,1], Y_test_hat2)
cm_table2
mean(data.miss[,1] == Y_test_hat2)

cm_table3 <- cm_table + cm_table2
cm_table3


########## Evaluation #############


# Confusion matrix
# Combined table
cm_table_n <- cm_table3




## Save

#keras_save(mod, "full_model.h5")
#keras_save_weights(mod, "weights_model.h5")
#keras_model_to_json(mod, "model_architecture.json")


# install_keras(method = c("auto", "virtualenv", "conda"),
#               conda = "auto", version = "default", tensorflow = "default",
#               extra_packages = c("tensorflow-hub"))
#install_keras(tensorflow = "gpu")
# 
# https://keras.rstudio.com/reference/install_keras.html
