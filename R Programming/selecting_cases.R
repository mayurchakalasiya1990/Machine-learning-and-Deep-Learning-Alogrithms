library(datasets)

?iris
head(iris)


#All data
hist(iris$Petal.Length)
summary(iris$Petal.Length)

summary(iris$Species)


# Select case by Category

hist(iris$Petal.Length[iris$Species == "versicolor"],main="Petal Length: Versicolor")

hist(iris$Petal.Length[iris$Species == "virginica"],main="Petal Length: virginica")

hist(iris$Petal.Length[iris$Species == "setosa"],main="Petal Length: setosa")

hist(iris$Petal.Length[iris$Petal.Length<2], main ="Petal Length < 2")

#Short petals only
hist(iris$Petal.Length[iris$Petal.Length<2], main ="Petal Length < 2")

# Multiple Selectors
hist(iris$Petal.Length[iris$Species == "virginica" & iris$Petal.Length<5.5], main ="Petal Length : Short Virginica")

# Create Sub Sample
#Leave rows or columns blan to select all
i.setosa <- iris[iris$Species == "setosa",]

#Explore subsample
head(i.setosa)
summary(i.setosa)
hist(i.setosa$Petal.Length)

#Clean up
detach("package:datasets", unload=TRUE)
