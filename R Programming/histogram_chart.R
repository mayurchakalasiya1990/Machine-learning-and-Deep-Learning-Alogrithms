# Load data set

library(datasets)

?iris
head(iris)

# basic histogram
hist(iris$Sepal.Length)
hist(iris$Sepal.Width)
hist(iris$Petal.Length)
hist(iris$Petal.Width)

#Put graphs in 3 rows and 1 column

par(mfrow = c(3,1))

# Histograms for each species using options

hist(iris$Petal.Width [iris$Species == "setosa"],
     xlim =c(0,3),
     breaks = 9,
     main = "Petal width for Setosa",
     xlab= "",
     col ="red")

hist(iris$Petal.Width [iris$Species == "versicolor"],
     xlim =c(0,3),
     breaks = 9,
     main = "Petal width for versicolor",
     xlab= "",
     col ="purple")

hist(iris$Petal.Width [iris$Species == "virginica"],
     xlim =c(0,3),
     breaks = 9,
     main = "Petal width for virginica",
     xlab= "",
     col ="blue")

# Clear enviroment
rm(list = ls())

