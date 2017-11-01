# Load packages
library('ggplot2') # visualization
library('dplyr') # data manipulation
library(rattle)	# Fancy tree plot-just incase
library(randomForest)
#Lets see the data

train <- read.csv("~/R/kaggle-titanic-master/data/train.csv")
test  <- read.csv('~/R/kaggle-titanic-master/data/test.csv')
test$Survived <- NA

train$input <- 1
test$input <- 0

full  <- rbind(train, test) # bind training & test data

str(full)
#------------------------------------------------------------------------------
# Feature Engineering

# Grab title from passenger names
full$Title <- gsub('(.*, )|(\\..*)', '', full$Name)
table(full$Sex, full$Title)

# Titles with very low cell counts combined them all in one level.
rare_title <- c('Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 
'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer')

# Also reassign mlle, ms, and mme accordingly
full$Title[full$Title == 'Mlle']        <- 'Miss' 
full$Title[full$Title == 'Ms']          <- 'Miss'
full$Title[full$Title == 'Mme']         <- 'Mrs' 
full$Title[full$Title %in% rare_title]  <- 'Rare Title'

# title counts by sex
table(full$Sex, full$Title)

# Finally, grab surname from passenger name
full$Surname <- sapply(full$Name,  
function(x) strsplit(x, split = '[,.]')[[1]][1])

## Do families sink or swim together?
#what insight we're getting for families, siblings/spouse and number of children/parents. 
# Fsize <- sibsp & parch + the passenger him/her self.

full$Fsize <- full$SibSp + full$Parch + 1

full$Family <- paste(full$Surname, full$Fsize, sep='_')

# Use ggplot2 to visualize the relationship between family size & survival
ggplot(full[1:891,], aes(x = Fsize, fill = factor(Survived))) +
  geom_bar(stat='count', position='dodge') +
  scale_x_continuous(breaks=c(1:11)) +
  labs(x = 'Family Size')

# family size levels
full$FsizeD[full$Fsize == 1] <- 'single'
full$FsizeD[full$Fsize < 5 & full$Fsize > 1] <- 'small'
full$FsizeD[full$Fsize > 4] <- 'large'

#--------------------------------------------------------------------------------------
mystats <- function(x){
  nmiss<-sum(is.na(x))
  return(c(nmiss=nmiss))}

names(full)
str(full)

vars <- c( "PassengerId","Pclass","Name","Sex","Age","SibSp",
           "Parch","Ticket","Fare","Cabin","Embarked")

dstats <- t(data.frame(apply(full[vars], 2, mystats)))


# Get rid of our missing passenger IDs

table(full$Embarked)
full[full$Embarked=="", "Embarked"] <- "S"

# Replace missing fare value with median fare for class/embarkment
table(is.na(full$Fare))
misfare <- median(full$Fare, na.rm = TRUE)
full$Fare[is.na(full$Fare)] <- misfare

## Predictive imputation for age, as its has 263 missing values.

dstats1 <- t(data.frame(apply(full[vars], 2, mystats)))

# Make variables factors into factors
rawdata <- full
ntmisage <- full[!is.na(rawdata$Age),] #new data set where age is not missing.
misage <- full[is.na(rawdata$Age),] ##new data set where age is missing.

table(is.na(ntmisage$Age))
table(is.na(misage$Age))

#lets predict the age

set.seed(1221)
predicted.age <- randomForest(Age~ Pclass + Sex + SibSp + Parch + Fare + Embarked,
                              data = ntmisage, ntree =500, mtry = 3, nodesize = 0.01 * nrow(ntmisage))

misage$Age <- predict(predicted.age, newdata = misage)
#combining these two data sets

full <- rbind(ntmisage, misage)  #new finaldata with zero NA's in age
table(is.na(full$Age))


# Plot age distributions
par(mfrow=c(1,2))
hist(full$Age, freq=F, 
     col='darkgreen', ylim=c(0,0.04))
hist(misage$Age, freq=F, 
     col='lightgreen', ylim=c(0,0.04))

###adding features in data

# Create the column child, and indicate whether child or adult
full$Child[full$Age < 18] <- 'Child'
full$Child[full$Age >= 18] <- 'Adult'

# Show counts
table(full$Child, full$Survived)

# Adding Mother variable
full$Mother <- 'Not Mother'
full$Mother[full$Sex == 'female' & full$Parch > 0 & full$Age > 18 & full$Title != 'Miss'] <- 'Mother'

table(full$Mother, full$Survived)
full$Child <- as.factor(full$Child)
full$Mother <- as.factor(full$Mother)

### all done, let's move to predicting

md.pattern(full)

# Prediction

## Split into training & test sets

# Split the data back into a train set and a test set
train <- full[full$input== 1,]
test <- full[full$input== 0,]

## Building the model 

#using `randomForest` on the training set.
# Set a random seed
set.seed(75441)

# Build the model
model12 <- randomForest(factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + 
                          Fare + Embarked + Child + Mother,
                        data = train, ntree = 500)


# Show model error
plot(model12, ylim=c(0,0.36))

model12$confusion

# Predict using the test set
pred2 <- predict(model12, test)

# Save the solution to a dataframe with two columns: PassengerId and Survived (prediction)
solution <- data.frame(PassengerID = test$PassengerId, Survived = pred2)

# Write the solution to file
write.csv(solution, file = 'rf_mod_Solution.csv', row.names = F)

##completed##

