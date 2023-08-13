library(datasets)

#LOAD DATA
?mtcars
head(mtcars)

#BAR CHARTS

barplot(mtcars$cyl)

# need a table with frequencies for each category
cylinders <- table(mtcars$cyl)
cylinders
barplot(cylinders)
plot(cylinders)

# Clear enviroment
rm(list = ls())
