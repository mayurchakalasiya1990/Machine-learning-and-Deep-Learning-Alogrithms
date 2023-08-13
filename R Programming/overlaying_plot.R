 library(datasets)
 
 ?lynx
 head(lynx)
 
 # Histogram
 
 # default
 hist(lynx)
 
 
 hist(lynx,
      breaks =14,
      freq = FALSE,
      col = "thistle1",
      main = paste("Histogram of Annual Canadian Lynx", "Trappings, 1821-1934"),
      xlab = "Number of Lynx Trapped"
      )
 curve(dnorm(x,mean = mean(lynx),sd=sd(lynx)),
       col="thistle4", # color of curve
       lwd=2, # Line width of 2 pixels
       add=TRUE # Superimpose on previous graph
       )
 
# Add two kernel density estimators
 lines(density(lynx),col="blue",lwd=2)
 lines(density(lynx,adjust = 3),col="purple",lwd=2)
 
 # Add a rug plat 
 rug(lynx, lwd=2,col="gray")
 
 #clean up
 detach("package:datasets", unload=TRUE)
 