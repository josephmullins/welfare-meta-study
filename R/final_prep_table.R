library(knitr)
library(kableExtra)
library(dplyr)

#d <- read.csv("~/Dropbox/WelfareMetaAnalysis/Data/Data_prepped.csv")
d <- read.csv("~/Dropbox/Research Projects/WelfareMetaAnalysis/Data/Data_prepped.csv")
kid_sample <- read.csv("~/Dropbox/Research Projects/WelfareMetaAnalysis/Data/Data_child_prepped.csv") %>%
  select(source,id)
  
sipp <- d %>%
  filter(source == "SIPP") 

d <- d %>%
  merge(kid_sample) %>%
  rbind(sipp)

summary_table <- d %>%
  group_by(source, arm) %>%
  summarize(
    frac_less_hs = mean(less_hs, na.rm = TRUE),
    frac_hs = mean(hs, na.rm = TRUE),
    frac_some_coll = mean(some_coll, na.rm = TRUE),
    frac_coll = mean(coll, na.rm = TRUE),
    frac_AFDC = mean(AFDC, na.rm = TRUE),
    frac_FS = mean(FS, na.rm = TRUE),
    avg_age = mean(age, na.rm = TRUE),
    avg_numkids = mean(numkids, na.rm = TRUE),
    frac_EMP = mean(EMP, na.rm = TRUE),
    avg_EARN = mean(EARN, na.rm = TRUE),
    count = n(), # Number of observations in each group
    individuals = n_distinct(id)
  )
print(summary_table)

transposed_table <- t(summary_table)
transposed_table <- transposed_table[-1, ]   
transposed_table <- apply(transposed_table, 2, function(x) as.numeric(as.character(x))) #convert to numeric to set digits of precision later

rownames(transposed_table) <- c(
  "Arm", "Less than Highschool", "Highschool", "Some College", 
  "College", "AFDC Participation", "Foodstamps Participation", "Mother's age", 
  "Number of Children", "Employed", "Earnings", "Person-Quarter Observations","Individuals"
)
colnames(transposed_table) <- c(
  "CTJF", "CTJF", "FTP", "FTP", "MFIP", "MFIP", "MFIP", "SIPP"
)

transposed_table[, 1:ncol(transposed_table)] <- round(transposed_table[, 1:ncol(transposed_table)], 3) #set digits of precision
transposed_table[1, ] <- format(transposed_table[1, ], nsmall = 0) #remove trailing zeros from integers
print(transposed_table)


# Construct the LaTeX output
latex_output <- capture.output({
  cat(
    kable(
      transposed_table,
      format = "latex",
      booktabs = TRUE,
      caption = "Summary Statistics",
      label = "tab:your_table_label",
      align="r"
    )
  )
})

# Print the LaTeX output
cat(latex_output)





