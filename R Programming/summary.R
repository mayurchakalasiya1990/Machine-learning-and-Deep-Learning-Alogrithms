library(datasets)

#Load data
?iris
head(iris)


# Summary
summary(iris$Species) # Categorical variable
summary(iris$Sepal.Length)  # Quantitative variable
summary(iris)  # Entire table variable

#clean up
detach("package:datasets", unload=TRUE)
