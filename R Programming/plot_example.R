library(datasets)

head(iris)

?plot


plot(iris$Species)  # Categorical variable
plot(iris$Petal.Length)  # Quantitative variable
plot(iris$Species, iris$Petal.Width)  # Cat x quant
plot(iris$Petal.Length, iris$Petal.Width)  # Quant pair
plot(iris)  # Entire data frame

# Plot with options
plot(iris$Sepal.Length,iris$Sepal.Width,
     col = "#cc0000",  # Hex code for datalab.cc red
     pch = 19,         # Use solid circles for points
     main = "Iris: Petal Length vs. Petal Width",
     xlab = "Petal Length",
     ylab = "Petal Width"
)

# Plot Formulas with options
plot(cos,0,2*pi)
plot(exp,1,5)
plot(dnorm,-3,+3)

plot(dnorm,-3,+3,
     col = "#cc0000",
     lwd = 5,
     main = "Standard Normal Distribution",
     xlab = "z-scores",
     ylab = "Density"
     )

# Clear packages
detach("package:datasets", unload = TRUE)

# Clear plots
dev.off()  # But only if there IS a plot

# Clear console
cat("\014")  # ctrl+L
