setwd("/Users/mattia/Desktop/AI_ass2/spam_filter/data/")
data <- read.csv("spambase.data", header =F)
#nm <- scan(file = "names.txt", what="character")
#names(data) <- nm
spam <- data[data$V58 == 1,]
ham <- data[data$V58 == 0,]

#### Independence of different words in same class #####

chisq.test(spam$V16, spam$V17) #word_freq_free" and "word_freq_business"  
chisq.test(spam$V19, spam$V24) #word_freq_you" and "word_freq_money"  

chisq.test(ham$V16, ham$V17) #word_freq_free" and "word_freq_business"  
chisq.test(ham$V19, ham$V24) #word_freq_you" and "word_freq_money"  
# they are dependent
