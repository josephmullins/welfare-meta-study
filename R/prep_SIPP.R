library(tidyverse)

D <- read.csv("data/SIPP_panel_92_93.csv") %>% mutate(X=NULL)
panel_92_93 <- read.csv("data/SIPP_chcare_92_93.csv") %>% mutate(X=NULL)
cpi <- read.csv("data/cpiu_quarterly.csv") %>% mutate(X=NULL)
unemp <- read.csv("data/StateUnemployment.csv") %>% mutate(X=NULL) %>%
  mutate(Q = floor((month-1)/3)+1) %>%
  group_by(fips,year,Q) %>%
  filter(row_number()==1) %>% #<- take the number from first month of the quarter
  mutate(StateFIPS = as.integer(fips)) %>%
  select(StateFIPS,year,Q,unemp)

statecodes <- read.csv("data/StateCodes.csv") %>% mutate(X=NULL) %>%
 rename(StateFIPS = fips) %>%
 select(StateFIPS,SOI)

# (1) get the sample
sample <- D %>%
  filter(wave==1,srefmon=="Month before interview month") %>%
  mutate(Y0 = year,Q0 = floor((month-1)/3)+1) %>% #<-
  select(panel,id,age,StateFIPS,educ,syngst,rfoklt18,black,metro,Y0,Q0) %>%
  unique() %>%
  rename(ageyng = syngst,numkids = rfoklt18) %>%
  drop_na() %>%
  filter(StateFIPS!="19, 38, 46",StateFIPS!="23, 50",StateFIPS!="02, 16, 30, 56",StateFIPS!="56, 38, 46") %>% #<- drop the FIPS codes that are lumped together
  mutate(StateFIPS = as.integer(StateFIPS)) %>%
  mutate(less_hs = educ=="less than high school",hs = educ=="high school grad",some_coll = educ=="some college",coll = educ=="college degree") %>%
  select(-educ) %>%
  merge(statecodes)

N = nrow(sample)


# (2) create quarterly measures of choices

Dq <- D %>%
  filter(srefmon=="Month before interview month") %>%
  mutate(Q = floor((month-1)/3)+1) %>%
  select(panel,id,year,Q,earn,tanf_a,fs_a,hours) %>%
  mutate(EMP = hours>0,EARN = 3*earn,AFDC = tanf_a>0,FS = fs_a>0) %>%
  arrange(id,year,Q) %>%
  group_by(id) %>%
  #mutate(source="v2") %>%
  select(-c("hours","earn","tanf_a","fs_a")) %>%
  as.data.frame()


# (3) create a balanced panel for time-varying observations
# fix this if want to include 92
panel <- sample %>%
  slice(kronecker(row_number(),rep(1,16))) %>%
  mutate(Q = rep(0:15%%4+1,N),year = rep(1992 + floor(0:15/4),N)) %>%
  left_join(Dq) %>%
  arrange(id,year,Q) %>%
  filter(((year-1)*4 + Q) >= ((Y0-1)*4 + Q0)) %>%
  filter((panel==1992 & year<=1994) | (panel==1993 & year>=1993) ) %>%
  inner_join(unemp) %>%
  inner_join(cpi) %>%
  arrange(id,year,Q)

# (4) 
chcare <- panel_92_93 %>%
  mutate(year = panel + floor((wave*4-1)/12),Q = floor((wave*4-1) %% 12) / 3) %>%
  filter(!is.na(used_cc)) %>% 
  mutate(chcare = replace_na(tot_fam1,0) + replace_na(tot_fam2,0) + replace_na(tot_for1,0) + replace_na(tot_for2,0)
         + replace_na(tot_rel1,0)+ replace_na(tot_rel2,0)+ replace_na(tot_nan1,0)+ replace_na(tot_nan2,0), usecare = used_cc>0) %>%
  mutate(Q = floor((wave-1)/3)+1) %>%
  select(id,chcare,year,Q,amt_fam1,amt_fam2,amt_for1,amt_for1,amt_rel1,amt_rel2,amt_nan1,amt_nan2,usecare,paid_cc,ccuniv1,ccuniv2)

panel <- panel %>%
  left_join(chcare) %>% 
  mutate(EARN = EARN/cpi,chcare = 4*chcare/cpi) #<- deflate values and convert childcare to monthly

write.csv(panel,"data/SIPP_prepped.csv")
write.csv(chcare,"data/SIPP_chcare.csv")
