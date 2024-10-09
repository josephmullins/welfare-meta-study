load("data/CTJF/ctadmrec.rdata")
load("data/CTJF/ctchsvy.rdata")
load("data/CTJF/ctsurvey.rdata")

library(tidyverse)

cpi <- read.csv("data/cpiu_quarterly.csv") %>% mutate(X=NULL)
unemp <- read.csv("data/StateUnemployment.csv") %>% mutate(X=NULL) %>%
  mutate(Q = floor((month-1)/3)+1) %>%
  group_by(fips,year,Q) %>%
  filter(row_number()==1) %>% #<- take the number from first month of the quarter
  mutate(StateFIPS = as.integer(fips)) %>%
  select(StateFIPS,year,Q,unemp)


# Earnings
d1 <- ctadmrec %>%
  select(PUFID,starts_with("ernpq")) %>%
  pivot_longer(cols = starts_with("ernpq"),names_to = "Q",values_to = "EARN",names_prefix="ernpq") %>%
  mutate(Q = -as.integer(Q))

cols = c()
for (q in 0:21) {
  cols <- c(cols,paste("ernq",q,sep=""))
}

d2 <- ctadmrec %>%
  select(PUFID,all_of(cols)) %>%
  pivot_longer(cols = starts_with("ernq"),names_to = "Q",values_to = "EARN",names_prefix="ernq") %>%
  mutate(Q = as.integer(Q))

earn <- rbind(d1,d2)

# AFDC
d1 <- ctadmrec %>%
  select(PUFID,starts_with("adcpq")) %>%
  pivot_longer(cols = starts_with("adcpq"),names_to = "Q",values_to = "AFDC",names_prefix="adcpq") %>%
  mutate(Q = -as.integer(Q))

cols = c()
for (q in 0:21) {
  cols <- c(cols,paste("adcq",q,sep=""))
}

d2 <- ctadmrec %>%
  select(PUFID,all_of(cols)) %>%
  pivot_longer(cols = starts_with("adcq"),names_to = "Q",values_to = "AFDC",names_prefix="adcq") %>%
  mutate(Q = as.integer(Q))

AFDC <- rbind(d1,d2)


# Food Stamps
d1 <- ctadmrec %>%
  select(PUFID,starts_with("fstpq")) %>%
  pivot_longer(cols = starts_with("fstpq"),names_to = "Q",values_to = "FS",names_prefix="fstpq") %>%
  mutate(Q = -as.integer(Q))

cols = c()
for (q in 0:21) {
  cols <- c(cols,paste("fstq",q,sep=""))
}

d2 <- ctadmrec %>%
  select(PUFID,all_of(cols)) %>%
  pivot_longer(cols = starts_with("fstq"),names_to = "Q",values_to = "FS",names_prefix="fstq") %>%
  mutate(Q = as.integer(Q))

FS <- rbind(d1,d2)

# demographics
# age, race, educ, numkids, age youngest, applicant status, treatment status
# APPLCANT: 0 - recipient, 1 - applicant
# county: 1: Manchester, 2: New Haven
demogs <- ctadmrec %>%
  filter(NOCHILD==0,FEMALE==1) %>%
  mutate(Y0 = 1996 + RAQTR197,Q0 = 1*(RAQTR196+RAQTR197) + 2*RAQTR296 + 3*RAQTR396 + 4*RAQTR496) %>%
  mutate(AGE = 20 + 7 * AGE2534 + 15 * AGE3544) %>% #<- CHECK THIS!!! is there a higher category?
  mutate(county = MANCHSTR + 2*NEWHAVEN) %>%
  select(PUFID,AGE,HIGRADE,BLACK,KIDCOUNT,TREATMNT,YNGCHTRU,Y0,Q0,APPLCANT,county) %>%
  drop_na() %>%
  rename(arm = TREATMNT,ageyng = YNGCHTRU,numkids = KIDCOUNT,black = BLACK,age = AGE) %>%
  rename(app_status = APPLCANT) %>%
  mutate(less_hs = HIGRADE<12,hs = HIGRADE==12,some_coll = (HIGRADE>12) & (HIGRADE<16),coll = HIGRADE>=16) %>%
  mutate(StateFIPS = 9,SOI = 7)


# main interview:
chcare <- ctsurvey %>%
  rename(chcare = E6Z) %>%
  #mutate(chcare = na_if(na_if(E6,999997),999999)) %>%
  select(PUFID,chcare) %>%
  mutate(Q = 12)

chcare <- ctchsvy %>%
  filter(!is.na(CFCEBPIS)) %>% #<- filter non-focal children
  mutate(pay_care = !is.na(Q4G) | !is.na(Q4H) | !is.na(Q4I) | !is.na(Q4J) | !is.na(Q4K) | !is.na(Q4L)) %>%
  select(PUFID,pay_care) %>%
  merge(chcare,all.y = TRUE)

ctchsvy %>%
  filter(!is.na(CFCEBPIS)) %>% #<- filter non-focal children
  select(PUFID,starts_with("Q4")) %>%
  merge(ctsurvey) %>%
  select(PUFID,starts_with("Q4"),E6a,E6,E6Z)
  

# [1] "id"        "age"       "month"     "StateFIPS" "educ"      "ageyng"    "numkids"   "black"     "Q"         "year"      "EMP"       "EARN"     
# [13] "AFDC"      "FS"        "chcare"    "usecare"

data <- demogs %>%
  merge(AFDC) %>%
  merge(FS) %>%
  merge(earn) %>%
  merge(chcare,all.x = TRUE) %>%
  mutate(year = Y0 + (floor((Q0-1+Q)/4))) %>%
  arrange(PUFID,Q) %>%
  rename(id = PUFID,AFDCt = AFDC,FSt = FS,rQ = Q) %>%
  mutate(EMP = EARN>0,AFDC = AFDCt>0,FS = FSt>0,Q = (Q0+rQ-1)%%4+1) %>%
  inner_join(unemp) %>%
  inner_join(cpi) %>%
  mutate(EARN = EARN/cpi,chcare = chcare/cpi,AFDCt = AFDCt/cpi,FSt = FSt/cpi)

write.csv(data,"data/CTJF_prepped.csv")

ctchsvy %>%
  merge(demogs) %>%
  group_by(ageyng,Q5) %>%
  summarize(n())


# child:
# age, BPI, suspensions, grade repeat, achievement, PBS, engagement?
# CANT FIND AGE HERE
# Q17_A is a teacher report.. more reliable than the parent report? #<- check for teacher report in all?
# AGE
# PRCUFORM: primary current care formal

# age all below 12
ctchsvy %>%
  rename(GRADE = Q5, ENGAG = CFCSHENS,SUSP = CACSUSX,REPEAT = CACREPTX,ACHIEVE = CAVPO,
         BPI = CFCBPIS,BPIE = CFCEBPIS,BPIN = CFCIBPIS,PBS = CFCPBSCS,
         TCH_AVG = Q17_A,TCH_READ = Q17_B, TCH_MATH = Q17_C) %>%
  select(PUFID,ENGAG,SUSP,REPEAT,ACHIEVE,BPIE,BPIN,PBS,TCH_AVG) %>%
  filter(!is.na(BPIE)) %>% #<- drops non-focal chilren
  merge(demogs) %>%
  mutate(AGEKID = ageyng+4) %>% #<- NOTE: AGE NOT AVAILABLE IN THESE DATA, we impute with age of youngest at RA + 4 years (time of follow up)
  write.csv("data/CTJF_child_prepped.csv")


# CFCSHENS	Numeric	4-12 scale of school engagement
# CAVPO	Integer	School performance measurement 1-5
# CACSUSX	Integer	Ever suspended
# CACEXPX	Integer	Ever expelled
# CACARRSX	Integer	Ever arrested
# CFCPBSCS	Numeric	PBS Social Competence subscale
# CACREPTX	Integer	Child ever repeated a grade
# CACSPEDX	Integer	Child in special ed
# CFCBPIS	Numeric	Behavioral Problems Index 0-56
