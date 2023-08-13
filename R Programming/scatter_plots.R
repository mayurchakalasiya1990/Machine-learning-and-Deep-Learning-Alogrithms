library(datasets)

#Load data

?mtcars
head(mtcars)

#Plots

# good to first check univariate distributions
hist(mtcars$wt)
hist(mtcars$mpg)


# Basic X-Y plt for two quantitative variables
plot(mtcars$wt,mtcars$mpg)


plot(mtcars$wt,mtcars$mpg,
     pch = 19, # Solid Circle
     cex = 1.5, # make 150% size
     col ="#cc0000",
     main = "MPG as Functionm of Weight of Cars",
     xlab = "Weight (in 1000 pounds)",
     ylab = "MPG",
     )

detach("package:datasets",unload = TRUE)

