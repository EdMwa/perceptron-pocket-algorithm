
###############################Some notes/definitions and assumptions#########################################
#W:"weight" is the transposed vector of w0, w1, w2,...wn
#X:"features" is vector of x0, x1, x3,...xn
#h(x): "hypothesis" is represented by W
#Y: "response" is the variable we are attempting to predict
#Y is assumed to be a binary class +1, -1
#LR: Learning Rate


##############################################WORKFLOW########################################################
#1. Load Iris data from data() & necessary package/s
#2. Data overview
#3. Feature extraction--generate a dataset with selected features & assign Y(response)==+1, -1 to either feature
#4. Hard code the pocket algorithm
#5. Test algorithm
#6. Perform prediction on test set
#7. Assess the performance of the algorithm

#############################################STEP 1 & 2#####################################################
library(MASS)
#data(package = .packages(all.available = TRUE))
#?iris
#Data Overview
names(iris)
summary(iris$Species)
summary(iris)
dim(iris)

################################STEP 3############################################################################

#Considering all possible bi-variate scatter plots
pairs(iris[,1:4], main = "Iris Data", pch = 21, bg = c("red", "pink", "blue")[iris$Species],
      oma=c(4,4,6,12))#set outer margins-bottom,left,top,right
par(xpd=TRUE) #Allow plotting of the legend outside the plots region within the space left to the right
legend(0.85, 0.7, as.vector(unique(iris$Species)),  
       fill=c("red", "pink", "blue"))
#The features Sepal.Width, Petal.Length and Petal.width show Setosa well clustered from the other 2 species
#Use these features for prediction

#create a training set (X) with these 3 features
X <- cbind(iris$Sepal.Width,iris$Petal.Width) #had a bit of a problem when I did 3 features
#Label setosa as +1 and the other 2 species together as -1
Y <- ifelse(iris$Species == 'setosa',+1,-1)
plot(X, cex = 0.5, xlab = '',ylab = '') #Generic plot 
#Set setosa points with '+' and the others with a '-'
points(subset(X, Y == +1), col = 'blue',pch='+', cex = 1)
points(subset(X, Y == -1), col = 'red',pch='-', cex = 1)

###########################################################################################################
##############################PERCEPTRON POCKET ALGORITHM##################################################
###########################################################################################################
# The core perceptron learning algorithm
# 1) Initialize the weight vector W to 0
# 2) Calculate hypothesis: h(x) = sign(transpose of W * (X))
# 3) Pick any misclassified point/s-not accurately predicted (xn, yn)
# 4) Update the weight vector by w <- w + yn * xn
# 5) Repeat until no points are misclassified


perceptron <- function(X, Y, LR = 1){
  converged <- FALSE
  
  #Initialize the weight vector
  W = vector(length = ncol(X))
  
  #number of iterations to run 10,000
  for (i in 1:10000){
    #calculate the hypothesis h
    h <- sign_pred(W %*% t(X))
    
    #compute the misclassified points
    mispredicted <- h != Y
    
    #Get TRUE if converged
    if (sum(mispredicted) == 0){
      converged <- TRUE
      break
    }else{
      
      #correct w for the mispredicted points and continue iterations
      
      mispredicted_X <- X[mispredicted, drop = FALSE]
      mispredicted_Y <- Y[mispredicted]
      
      #Extract a pair of the mispredicted data from above
      mispredicted_index <- sample(dim(mispredicted_X)[1], 1)
      mispredicted_point_X <- mispredicted_X[mispredicted_index, drop = F]
      mispredicted_point_Y <- mispredicted_Y[mispredicted_index]
      
      #update W for the mispredicted pairs above
       W <- W + mispredicted_point_Y %*% mispredicted_point_X
       
    } #repeat iteration hoping for convergence after correction!!
  }
  
    if (converged){
      
      cat('converged!\n')
    }else{
      
      cat('Did not converge!\n')
    }#Go ahead and return the best W so far
  return(W)
}


#######################Define the sign function used above##############################################
sign_pred <- function(nums){
  return(ifelse(nums > 0, +1, -1))
}


################################Line Seperator##########################################################

which_side <- function(line_sep, point){
  nums <- (line_sep[2, 1] - line_sep[1, 1])*(point[, 2] - line_sep[1, 2]) -
     (line_sep[2, 2]) - (line_sep[1, 2])*(point[, 1] - line_sep[1, 1])
  return(sign_pred(nums))
}
##################################Test perceptron#######################################################
pred_W <- perceptron(X, Y)

