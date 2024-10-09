load("data/FTP/ftp_ar.rdata")
load("data/FTP/ftp_chd.rdata")
load("data/FTP/ftp_srv.rdata")

cpi <- read.csv("data/cpiu_quarterly.csv") %>% mutate(X=NULL)
unemp <- read.csv("data/StateUnemployment.csv") %>% mutate(X=NULL) %>%
  mutate(Q = floor((month-1)/3)+1) %>%
  group_by(fips,year,Q) %>%
  filter(row_number()==1) %>% #<- take the number from first month of the quarter
  mutate(StateFIPS = as.integer(fips)) %>%
  select(StateFIPS,year,Q,unemp)


# Earnings
d1 <- ftp_ar %>%
  select(SAMPLEID,starts_with("PEARN")) %>%
  pivot_longer(cols = starts_with("PEARN"),names_to = "Q",values_to = "EARN",names_prefix="PEARN") %>%
  mutate(Q = -as.integer(Q))

cols = c()
for (q in 1:22) {
  cols <- c(cols,paste("EARN",q,sep=""))
}

d2 <- ftp_ar %>%
  select(SAMPLEID,all_of(cols)) %>%
  pivot_longer(cols = starts_with("EARN"),names_to = "Q",values_to = "EARN",names_prefix="EARN") %>%
  mutate(Q = as.integer(Q) - 1)

earn <- rbind(d1,d2)

# AFDC
d1 <- ftp_ar %>%
  select(SAMPLEID,starts_with("ADCPC")) %>%
  pivot_longer(cols = starts_with("ADCPC"),names_to = "Q",values_to = "AFDC",names_prefix="ADCPC") %>%
  mutate(Q = -as.integer(Q))

cols = c()
for (q in 1:21) {
  cols <- c(cols,paste("ADCC",q,sep=""))
}

d2 <- ftp_ar %>%
  select(SAMPLEID,all_of(cols)) %>%
  pivot_longer(cols = starts_with("ADCC"),names_to = "Q",values_to = "AFDC",names_prefix="ADCC") %>%
  mutate(Q = as.integer(Q) - 1)

afdc <- rbind(d1,d2)

# Food Stamps
d1 <- ftp_ar %>%
  select(SAMPLEID,starts_with("FSPC")) %>%
  pivot_longer(cols = starts_with("FSPC"),names_to = "Q",values_to = "FS",names_prefix="FSPC") %>%
  mutate(Q = -as.integer(Q))

cols = c()
for (q in 1:21) {
  cols <- c(cols,paste("FSC",q,sep=""))
}

d2 <- ftp_ar %>%
  select(SAMPLEID,all_of(cols)) %>%
  pivot_longer(cols = starts_with("FSC"),names_to = "Q",values_to = "FS",names_prefix="FSC") %>%
  mutate(Q = as.integer(Q) - 1)

FS <- rbind(d1,d2)

# demogs
# APPLCANT: 0 - recipient, 1 - applicant
demogs <- ftp_ar %>%
  filter(NOCHILD==0) %>%
  mutate(Y0 = 1994,Q0 = RARELQT) %>%
  mutate(numkids = 1*ONECHILD + 2*TWOCHILD + 3*THRCHILD + 4*GE4CHILD) %>%
  mutate(age = 20 + 7*AGE2534 + 15*AGE3544 + 25*AGEGE45) %>%
  rename(ageyng = YNGCHAGE) %>%
  select(SAMPLEID,BLACK,HIGRADE,numkids,ageyng,B_AIDST,age,E,Y0,Q0,APPLCANT) %>%
  drop_na() %>%
  rename(arm = E,numkids = numkids,black = BLACK,app_status = APPLCANT) %>%
  mutate(less_hs = HIGRADE<12,hs = HIGRADE==12,some_coll = (HIGRADE>12) & (HIGRADE<16),coll = HIGRADE>=16) %>%
  mutate(StateFIPS = 12,SOI = 10)


# main survey: childcare
chcare <- ftp_srv %>%
  rename(chcare = FME6) %>%
  select(SAMPLEID,chcare) %>%
  mutate(Q = 16) #<- 48 month followup
 
chcare <- ftp_chd %>%
  filter(FCFLAG==1) %>%
  select(SAMPLEID,starts_with("FME5I")) %>%
  mutate(pay_care = (FME5I1>6 & FME5I1<14) | (FME5I2>6 & FME5I2<14) | (FME5I3>6 & FME5I3<14)) %>%
  mutate(pay_care = replace_na(pay_care,FALSE)) %>%
  select(SAMPLEID,pay_care) %>%
  merge(chcare,all.y = TRUE) 

data <- demogs %>%
  merge(afdc) %>%
  merge(FS) %>%
  merge(earn) %>%
  merge(chcare,all.x = TRUE) %>%
  mutate(year = Y0 + (floor((Q0-1+Q)/4))) %>%
  arrange(SAMPLEID,Q) %>%
  rename(id = SAMPLEID,AFDCt = AFDC,FSt = FS,rQ = Q) %>%
  mutate(EMP = EARN>0,AFDC = AFDCt>0,FS = FSt>0,Q = (Q0+rQ-1)%%4+1) %>%
  inner_join(unemp) %>%
  inner_join(cpi) %>%
  mutate(EARN = EARN/cpi,chcare = chcare/cpi,AFDCt = AFDCt/cpi,FSt = FSt/cpi)

write.csv(data,"data/FTP_prepped.csv")



# FCAAGE3	Binary	Child age at last birthday 0-4 / 2: 5-12 / 3: 13+
# FCF12	Integer	Weekly cost of primary child care
# FME6	Numeric	Cost of child care last month
# FME5K	Integer	Hours in child care per week
# FCAFORMZ	Integer	Had Formal care in last week
# FME50	Integer	School Progress measurement
# FME5Q	Integer	Ever suspended/expelled
# FCI4A	Integer	What grade is child in?
# FCI9	Integer	Child ever repeated a grade
# FCI19	Integer	Child in special ed
# FCS3E	Integer	Behavioral Problems Index
# child survey:

ftp_chd %>%
  filter(FCFLAG==1) %>%
  # note: for the age we can narrow it down using age restrictions on focal child?
  mutate(AGEKID = case_when(FCAAGE3==1 ~ 1,FCAAGE3==2 ~ 8,FCAAGE3==3 ~ 15)) %>%
  rename(REPEAT = FCI9,ACHIEVE = FCASCHLZ,PBS = FCPBSSCS) %>%
  rename(BPI = FCBPIS,BPIE = FCEBPIS,BPIN = FCIBPIS) %>%
  rename(SUSPEND = FCASUSPZ,ENGAGE = FCSCHENM) %>%
  select(SAMPLEID,AGEKID,ACHIEVE,BPIE,BPIN,SUSPEND,REPEAT,ENGAGE,PBS) %>%
  merge(demogs) %>%
  write.csv("data/FTP_child_prepped.csv")
