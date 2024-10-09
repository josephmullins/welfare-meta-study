load("data/MFIP/PUF.RData")
load("data/MFIP/PUFCH.RData")
load("data/MFIP/PUFCR.RData")
library(tidyverse)

cpi <- read.csv("data/cpiu_quarterly.csv") %>% mutate(X=NULL)
unemp <- read.csv("data/StateUnemployment.csv") %>% mutate(X=NULL) %>%
  mutate(Q = floor((month-1)/3)+1) %>%
  group_by(fips,year,Q) %>%
  filter(row_number()==1) %>% #<- take the number from first month of the quarter
  mutate(StateFIPS = as.integer(fips)) %>%
  select(StateFIPS,year,Q,unemp)


# --- Step 1: build a dataset with quarterly earnings
d1 <- PUF %>%
  select(PUBID,starts_with("PEARN")) %>%
  pivot_longer(cols = starts_with("PEARN"),names_to = "Q",values_to = "EARN",names_prefix="PEARN") %>%
  mutate(Q = -as.integer(Q))

cols = c()
for (q in 1:18) {
  cols <- c(cols,paste("EARN",q,sep=""))
}

d2 <- PUF %>%
  select(PUBID,all_of(cols)) %>%
  pivot_longer(cols = starts_with("EARN"),names_to = "Q",values_to = "EARN",names_prefix="EARN") %>%
  mutate(Q = as.integer(Q) - 1)

earn <- rbind(d1,d2)

# ------------ 

# ----- Step 2: build a dataset with welfare receipt
#NOTE: ADC IS THE SUM OF ALL, AFC IS AFDC ONLY, MFP IS MFIP
# ---- Total Benefits
d1 <- PUF %>%
  select(PUBID,starts_with("ADCPM")) %>%
  pivot_longer(cols = starts_with("ADCPM"),names_to = "M",values_to = "BenTot",names_prefix="ADCPM") %>%
  mutate(Q = -floor((as.integer(M)-1)/3)-1) %>%
  mutate(M = -as.integer(M))

d2 <- PUF %>%
  select(PUBID,starts_with("ADCM")) %>%
  pivot_longer(cols = starts_with("ADCM"),names_to = "M",values_to = "BenTot",names_prefix="ADCM") %>%
  mutate(M = as.integer(M)-1) %>%
  mutate(Q = floor(M/3))

BenTot <- rbind(d1,d2) %>%
  mutate(BenTot = na_if(BenTot,-99999)) %>%
  group_by(PUBID,Q) %>%
  summarize(BenTot = sum(BenTot))

# PUF %>%
#   select(PUBID,P_RESGP,FAMSTAT,URBANRUR,APPRECIP,MFIPTRIG) %>%
#   merge(afdc) %>%
#   filter(FAMSTAT==1,URBANRUR==1,!P_RESGP=="C2") %>%
#   group_by(P_RESGP,Q,APPRECIP) %>%
#   summarize(A = mean(AFDC>0)) %>%
#   ggplot(aes(x=Q,y=A,color=P_RESGP)) + geom_line() + geom_point() + facet_grid(. ~ APPRECIP)

# --- Step 3: Food Stamps
d1 <- PUF %>%
  select(PUBID,starts_with("FSPM")) %>%
  pivot_longer(cols = starts_with("FSPM"),names_to = "M",values_to = "FS",names_prefix="FSPM") %>%
  mutate(Q = -floor((as.integer(M)-1)/3)-1) %>%
  mutate(M = -as.integer(M))

d2 <- PUF %>%
  select(PUBID,starts_with("FSM")) %>%
  pivot_longer(cols = starts_with("FSM"),names_to = "M",values_to = "FS",names_prefix="FSM") %>%
  mutate(M = as.integer(M)-1) %>%
  mutate(Q = floor(M/3))

FS <- rbind(d1,d2) %>%
  group_by(PUBID,Q) %>%
  summarize(FS=sum(FS))


# --- Step 4: AFDC and MFIP... convert to quarters
d1 <- PUF %>%
  select(PUBID,starts_with("AFCPM")) %>%
  pivot_longer(cols = starts_with("AFCPM"),names_to = "M",values_to = "AFDC",names_prefix="AFCPM") %>%
  mutate(Q = -floor((as.integer(M)-1)/3)-1) %>%
  mutate(M = -as.integer(M))

d2 <- PUF %>%
  select(PUBID,starts_with("AFCM")) %>%
  pivot_longer(cols = starts_with("AFCM"),names_to = "M",values_to = "AFDC",names_prefix="AFCM") %>%
  mutate(M = as.integer(M)-1) %>%
  mutate(Q = floor(M/3))


d3 <- PUF %>%
  select(PUBID,starts_with("MFPM")) %>%
  pivot_longer(cols = starts_with("MFPM"),names_to = "M",values_to = "MFIP",names_prefix="MFPM") %>%
  mutate(M = as.integer(M)-1) %>%
  mutate(Q = floor(M/3))

d2 <- merge(d2,d3) %>%
  mutate(AFDC = AFDC + MFIP) %>%
  select(-MFIP)

afdc <- rbind(d1,d2) %>%
  group_by(PUBID,Q) %>%
  summarize(AFDC=sum(AFDC))


# [1] "id"        "age"       "month"     "StateFIPS" "educ"      "ageyng"    "numkids"   "black"     "Q"         "year"      "EMP"       "EARN"     
# [13] "AFDC"      "FS"        "chcare"    "usecare"

# step 5: create demographics and do some selection
# APPRECIP: 1 - new applicant, 2 - re-applicant, 3 - recipient
# MFIPTRIG: receipt long enough to trigger services
# county: 1 - Hennepin, 2 - Anoka, 3 - Dakota

demogs <- PUF %>%
  filter(FAMSTAT==1,URBANRUR==1) %>%
  mutate(Y0 = 1994*(RA2Q94+RA3Q94+RA4Q94) + 1995*(RA1Q95+RA2Q95+RA3Q95+RA4Q95) + 1996*RA1Q96) %>%
  mutate(Q0 = 1*(RA1Q95+RA1Q96) + 2*(RA2Q94+RA2Q95) + 3*(RA3Q94+RA3Q95) + 4*(RA4Q94+RA4Q95)) %>%
  select(PUBID,P_RESGP,AGE25T34,AGE35UP,YGAGECAT,HIGRADE,KID1TRUX,BLACK,NUMKIDS,APPRECIP,MFIPTRIG,Y0,Q0,HENNEPIN,ANOKA,DAKOTA) %>%
  mutate(age = 20 + 7 *AGE25T34 + 15 * AGE35UP) %>% #<- ages round to 
  # age categories: 1: 0-2, 2: 3-5, 3: 6-18
  mutate(ageyng = case_when(YGAGECAT==1 ~ 1,YGAGECAT==2 ~ 4,YGAGECAT==3 ~ 10)) %>%
  mutate(arm = case_when(P_RESGP=="C1" ~ 0,P_RESGP=="E1" ~ 1,P_RESGP=="E2" ~ 2)) %>%
  mutate(less_hs = HIGRADE<12,hs = HIGRADE==12,some_coll = (HIGRADE>12) & (HIGRADE<16),coll = HIGRADE>=16) %>%
  mutate(county = HENNEPIN + 2*ANOKA + 3*DAKOTA) %>%
  rename(app_status = APPRECIP) %>%
  select(PUBID,arm,less_hs,hs,some_coll,coll,age,ageyng,BLACK,NUMKIDS,app_status,MFIPTRIG,Y0,Q0,county,app_status) %>%
  drop_na()
  

# convert to monthly childcare expense
chcare <- PUFCR %>% 
  mutate(chcare = case_when(H9A==0 ~ as.double(H9A),H9BA==4 ~ as.double(H9A),H9AA==1 ~ as.double(H9A*4))) %>%
  mutate(pay_care = (H61A==1) | (H62A==1) | (H63A==1) | (H64A==1) | (H65A==1) | (H66A==1)) %>% # | (H67A==1)) %>% #<- exclude relatives from paid care to make comparable with FTP and CTJF
  mutate(pay_care = replace_na(pay_care,FALSE)) %>%
  select(PUBID,chcare,pay_care) %>% #<- something is weird with how childcare is read in (missing a decimal place?) so we divide by 100
  mutate(Q = 12,chcare = chcare/100) #<- 36 month follow up means the 12th quarter since RA


# [1] "id"        "age"       "month"     "StateFIPS" "educ"      "ageyng"    "numkids"   "black"     "Q"         "year"      "EMP"       "EARN"     
# [13] "AFDC"      "FS"        "chcare"    "usecare"

data <- demogs %>%
  merge(afdc) %>%
  merge(FS) %>%
  merge(earn) %>%
  merge(BenTot) %>%
  merge(chcare,all.x = TRUE) %>%
  mutate(year = Y0 + (floor((Q0-1+Q)/4))) %>%
  arrange(PUBID,Q) %>%
  rename(id = PUBID,AFDCt = AFDC,FSt = FS,black = BLACK,numkids = NUMKIDS,rQ = Q) %>%
  mutate(EMP = EARN>0,AFDC = AFDCt>0,FS = FSt>0,Q = (Q0+rQ-1)%%4+1) %>%
  mutate(StateFIPS = 27,SOI = 24) %>%
  inner_join(unemp) %>%
  inner_join(cpi) %>%
  mutate(EARN = EARN/cpi,chcare = chcare/cpi,AFDCt = AFDCt/cpi,FSt = FSt/cpi)

write.csv(data,"data/MFIP_prepped.csv")

PUFCH %>%
  select(-PBS) %>%
  rename(AGEKID = AGE36M,BPI = BPI,BPIE = BPIEXT, BPIN = BPIINT,ACHIEVE = GRADE,
         PBS = PBSSOC, SUSPEND = SUSPFK, REPEAT = REPFK, ENGAGE = ENGAG) %>%
  select(PUBID,AGEKID,BPIE,BPIN,PBS,ENGAGE,REPEAT,SUSPEND,ACHIEVE) %>%
  merge(demogs) %>%
  drop_na() %>%
  write.csv("data/MFIP_child_prepped.csv")


  
