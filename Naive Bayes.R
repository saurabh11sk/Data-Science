#This chapter covers the Naive 
#Bayes algorithm, which uses probabilities in much the 
#same way as a weather forecast. 
# Step 1  download data 

# step 2  .  exploring and preparing the data
sms_raw<-read.csv(file.choose(),stringsAsFactors = FALSE)

View(sms_raw)

str(sms_raw)
# 'data.frame':	5574 obs. of  2 variables:
#$ type: chr  "ham" "ham" "spam" "ham" ...
#$ text: chr  "Go until jurong point, crazy.. 
#Available only in bugis n great world la e buffet...
#Cine there got amore wat..." "Ok lar... Joking wif u oni...
#" "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005.
#Text FA to 87121 to receive entry question("| __truncated__ "U dun say so early hor... 
#U c already then say..." ...

#The type element is currently a character vector. Since this is a categorical variable, it 
#would be better to convert it into a factor, as shown in the following code

sms_raw$type<- factor(sms_raw$type)


#Examining this with the str() and table() functions, we see that type has now 
#been appropriately recoded as a factor. Additionally, we see that 747 (about 13                                                                       percent) of SMS messages in our data were labeled as spam, while the others were 
#labeled as ham:

str(sms_raw$type)


table(sms_raw$type)

#Data preparation – cleaning and standardizing text 
#data


#SMS messages are strings of text composed of words, spaces, numbers, and 
#punctuation. Handling this type of complex data takes a lot of thought and 
#effort. One needs to consider how to remove numbers and punctuation; handle 
#uninteresting words such as and, but, and or; and how to break apart sentences into 
#individual words. Thankfully, this functionality has been provided by the members 
#of the R community in a text mining package titled tm.


#install.packages("tm")  Text mining
library(tm)

#The first step in processing text data involves creating a corpus, which is a collection 
#of text documents. 

#In our case, the corpus  
#will be a collection of SMS messages.


#In order to create a corpus, we'll use the VCorpus() function in the tm package, 
#which refers to a volatile corpus—volatile as it is stored in memory as opposed to 
#being stored on disk (the PCorpus() function can be used to access a permanent 
#corpus stored in a database). This function requires us to specify the source of 
#documents for the corpus, which could be from a computer's filesystem, a database, 
#the Web, or elsewhere. Since we already loaded the SMS message text into R, we'll 
#use the VectorSource() reader function to create a source object from the existing 
#sms_raw$text vector, which can then be supplied to VCorpus() as follows:

sms_corpus<-VCorpus(VectorSource(sms_raw$text))

print(sms_corpus)


#Because the tm corpus is essentially a complex list, we can use list operations to select 
#documents in the corpus. To receive a summary of specific messages, we can use the 
#inspect() function with list operators. For example, the following command will 
#view a summary of the first and second SMS messages in the corpus:


inspect(sms_corpus[1:2])


#To view the actual message text, the as.character() function must be applied to 
#the desired messages. To view one message, use the as.character() function on  
#a single list element, noting that the double-bracket notation is required

as.character(sms_corpus[[1]])


#for all

lapply(sms_corpus[1:2],as.character)


#As noted earlier, the corpus contains the raw text of 5,559 text messages. In order 
#to perform our analysis, we need to divide these messages into individual words. 
#But first, we need to clean the text, in order to standardize the words, by removing 
#punctuation and other characters that clutter the result. For example, we would like 
#the strings Hello!, HELLO, and hello to be counted as instances of the same word.


#The tm_map() function provides a method to apply a transformation (also known 
#as mapping) to a tm corpus. We will use this function to clean up our corpus using a 
#series of transformations and save the result in a new object called corpus_clean.

sms_corpus_clean<-tm_map(sms_corpus,content_transformer(tolower))

?tm_map
# check with original

as.character(sms_corpus[[1]])

as.character(sms_corpus_clean[[1]])

# clean further 

sms_corpus_clean<-tm_map(sms_corpus_clean,removeNumbers)

#Our next task is to remove filler words such as to, and, but, and or from our SMS 
#messages. These terms are known as stop words and are typically removed prior to 
#text mining. This is due to the fact that although they appear very frequently, they do 
#not provide much useful information for machine learning



sms_corpus_clean<-tm_map(sms_corpus_clean,removeWords,stopwords())


#Since stopwords() simply returns a vector of stop words, had we chosen so, we 
#could have replaced it with our own vector of words to be removed. In this way, we 
#could expand or reduce the list of stop words to our liking or remove a completely 
#different set of words entirely

sms_corpus_clean<-tm_map(sms_corpus_clean,removePunctuation)

#The removePunctuation() transformation strips punctuation characters from the 
#text blindly, which can lead to unintended consequences. For example, consider 
#what happens when it is applied as follows:
# 
#As shown, the lack of blank space after the ellipses has caused the words hello and 
#world to be joined as a single word. While this is not a substantial problem for our 
#analysis, it is worth noting for the future




#Another common standardization for text data involves reducing words to their root 
#form in a process called stemming. The stemming process takes words like learned, 
#learning, and learns, and strips the suffix in order to transform them into the base 
#form, learn. This allows machine learning algorithms to treat the related terms as a 
#single concept rather than attempting to learn a pattern for each variant.
#The tm package provides stemming functionality via integration with the SnowballC 
#package. At the time of this writing, SnowballC was not installed by default with tm. 
#Do so with install.packages("SnowballC") if it is not installed already

library(SnowballC)

sms_corpus_clean<- tm_map(sms_corpus_clean,stemDocument)

sms_corpus_clean<-tm_map(sms_corpus_clean,stripWhitespace)


# Data preparation – splitting text documents into 
#words

#Now that the data are processed to our liking, the final step is to split the messages 
#into individual components through a process called tokenization. A token is a 
#single element of a text string; in this case, the tokens are words.


#The fact that each cell in the table is zero implies that none of the words listed on 
#the top of the columns appear in any of the first five messages in the corpus. This 
#highlights the reason why this data structure is called a sparse matrix; the vast 
#majority of the cells in the matrix are filled with zeros. Stated in real-world terms, 
#although each message must contain at least one word, the probability of any one 
#word appearing in a given message is small.


sms_dtm <-DocumentTermMatrix(sms_corpus_clean)

sms_dtm2 <- DocumentTermMatrix(sms_corpus, control = list(
  tolower = TRUE,
  removeNumbers = TRUE,
  stopwords = TRUE,
  removePunctuation = TRUE,
  stemming = TRUE
))

sms_dtm

sms_dtm2




#Data preparation – creating training and test 
#datasets


sms_dtm_train <- sms_dtm[1:4169, ]
sms_dtm_test  <- sms_dtm[4170:5559, ]


sms_train_labels <- sms_raw[1:4169, ]$type
sms_test_labels  <- sms_raw[4170:5559, ]$type


prop.table(table(sms_train_labels))

prop.table(table(sms_test_labels))

#Visualizing text data – word clouds

install.packages("wordcloud")

library(wordcloud)


wordcloud(sms_corpus_clean, min.freq = 50, random.order = FALSE)

#Let's use R's subset() function to take a subset of the sms_raw data by the SMS 
#type. First, we'll create a subset where the message type is spam

spam <- subset(sms_raw, type == "spam")

ham <- subset(sms_raw, type == "ham")


wordcloud(spam$text, max.words = 40, scale = c(3, 0.5))
wordcloud(ham$text, max.words = 40, scale = c(3, 0.5))


#Step 3 – training a model on the data



#install.packages("e1071") 
library(e1071)


#The final step in the data preparation process is to transform the sparse matrix into a 
#data structure that can be used to train a Naive Bayes classifier. Currently, the sparse 
#matrix includes over 6,500 features; this is a feature for every word that appears in at 
#least one SMS message. It's unlikely that all of these are useful for classification. To 
#reduce the number of features, we will eliminate any word that appear in less than five 
#SMS messages, or in less than about 0.1 percent of the records in the training data.
# Finding frequent words requires use of the findFreqTerms() function in the  
#tm package. This function takes a DTM and returns a character vector containing  
#the words that appear for at least the specified number of times. For instance,  
#the following command will display the words appearing at least five times in  
#the sms_dtm_train matrix:

sms_freq_term<-findFreqTerms(sms_dtm_train,5)

str(sms_freq_term)


sms_dtm_freq_train<- sms_dtm_train[ , sms_freq_term]
sms_dtm_freq_test<- sms_dtm_test[ , sms_freq_term]



convert_counts <- function(x) {
  x <- ifelse(x > 0, "Yes", "No")
}




sms_train <- apply(sms_dtm_freq_train, MARGIN = 2,
                   convert_counts)


sms_test <- apply(sms_dtm_freq_test, MARGIN = 2,
                  convert_counts)

sms_classifier <- naiveBayes(sms_train, sms_train_labels)


sms_test_pred <- predict(sms_classifier, sms_test)


library(gmodels)


CrossTable(sms_test_pred, sms_test_labels,
           prop.chisq = FALSE,
           dnn = c('predicted', 'actual'))


# Calculate the number of correct predictions
correct_predictions <- sum(sms_test_pred == sms_test_labels)

# Calculate the total number of predictions
total_predictions <- length(sms_test_labels)

# Calculate accuracy
accuracy <- (correct_predictions / total_predictions) * 100

# Print the accuracy
print(paste("Accuracy:", accuracy, "%"))
















# Step 1  download data 

# step 2  exploring and preparing the data
sms_raw <- read.csv(file.choose(), stringsAsFactors = FALSE)
View(sms_raw)
str(sms_raw)

sms_raw$type <- factor(sms_raw$type)
str(sms_raw$type)
table(sms_raw$type)

# Data preparation – cleaning and standardizing text data
library(tm)

sms_corpus <- VCorpus(VectorSource(sms_raw$text))
print(sms_corpus)
inspect(sms_corpus[1:2])
as.character(sms_corpus[[1]])
lapply(sms_corpus[1:2], as.character)

sms_corpus_clean <- tm_map(sms_corpus, content_transformer(tolower))
as.character(sms_corpus[[1]])
as.character(sms_corpus_clean[[1]])

sms_corpus_clean <- tm_map(sms_corpus_clean, removeNumbers)
sms_corpus_clean <- tm_map(sms_corpus_clean, removeWords, stopwords())
sms_corpus_clean <- tm_map(sms_corpus_clean, removePunctuation)

library(SnowballC)
sms_corpus_clean <- tm_map(sms_corpus_clean, stemDocument)
sms_corpus_clean <- tm_map(sms_corpus_clean, stripWhitespace)

sms_dtm <- DocumentTermMatrix(sms_corpus_clean)
sms_dtm2 <- DocumentTermMatrix(sms_corpus, control = list(
  tolower = TRUE,
  removeNumbers = TRUE,
  stopwords = TRUE,
  removePunctuation = TRUE,
  stemming = TRUE
))

sms_dtm
sms_dtm2

# Data preparation – creating training and test datasets
sms_dtm_train <- sms_dtm[1:4169, ]
sms_dtm_test  <- sms_dtm[4170:5559, ]
sms_train_labels <- sms_raw[1:4169, ]$type
sms_test_labels  <- sms_raw[4170:5559, ]$type

prop.table(table(sms_train_labels))
prop.table(table(sms_test_labels))

# Visualizing text data – word clouds
install.packages("wordcloud")
library(wordcloud)

wordcloud(sms_corpus_clean, min.freq = 50, random.order = FALSE)
spam <- subset(sms_raw, type == "spam")
ham <- subset(sms_raw, type == "ham")

wordcloud(spam$text, max.words = 40, scale = c(3, 0.5))
wordcloud(ham$text, max.words = 40, scale = c(3, 0.5))

# Step 3 – training a model on the data
library(e1071)

sms_freq_term <- findFreqTerms(sms_dtm_train, 5)
str(sms_freq_term)

sms_dtm_freq_train <- sms_dtm_train[, sms_freq_term]
sms_dtm_freq_test <- sms_dtm_test[, sms_freq_term]

convert_counts <- function(x) {
  x <- ifelse(x > 0, "Yes", "No")
}

sms_train <- apply(sms_dtm_freq_train, MARGIN = 2, convert_counts)
sms_test <- apply(sms_dtm_freq_test, MARGIN = 2, convert_counts)

sms_classifier <- naiveBayes(sms_train, sms_train_labels)
sms_test_pred <- predict(sms_classifier, sms_test)

library(gmodels)

CrossTable(sms_test_pred, sms_test_labels,
           prop.chisq = FALSE,
           dnn = c('predicted', 'actual'))

# Calculate the number of correct predictions
correct_predictions <- sum(sms_test_pred == sms_test_labels)

# Calculate the total number of predictions
total_predictions <- length(sms_test_labels)

# Calculate accuracy
accuracy <- (correct_predictions / total_predictions) * 100

# Print the accuracy
print(paste("Accuracy:", accuracy, "%"))
