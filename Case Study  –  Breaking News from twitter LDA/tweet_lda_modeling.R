#Load packages
library(tm)
library(topicmodels)
#load document from text file
text <- readLines("/home/geekarea/course_work/web and social media/Case Study 2 – Breaking News/scraped_tweets.txt")


#print(text)
#load document from text file and perform pre-processing 
doc.vec <- VectorSource(text)
doc.corpus <- Corpus(doc.vec)
doc.corpus <- tm_map(doc.corpus, function(x) inconv(enc2utf8(x), sub = "byte"))
doc.corpus <- tm_map(doc.corpus, PlainTextDocument)
doc.corpus <- tm_map(doc.corpus, content_transformer(tolower))
doc.corpus <- tm_map(doc.corpus, removeWords, stopwords('english'))
doc.corpus <- tm_map(doc.corpus, removePunctuation)
doc.corpus <- tm_map(doc.corpus, removeNumbers)
doc.corpus <- tm_map(doc.corpus, stripWhitespace)
print(doc.corpus)

#Remove sparse Terms from the matrix
dtm <- DocumentTermMatrix(doc.corpus)
dtm <- removeSparseTerms(dtm, 0.98)
x <- as.matrix(dtm)
x <- x[which(rowSums(x)>0),]
rownames(x) <- 1:nrow(x)
print(x)
inspect(dtm)
# train LDA model
lda <- LDA(x,6)
inspect(dtm)
terms(lda,6)

# posterior distribution of terms over topics
topic = 6
words = posterior(lda)$terms[topic, ]
topwords = head(sort(words, decreasing = T), n=5)
head(topwords)

# Word could
library(wordcloud)
wordcloud(names(topwords), topwords)
