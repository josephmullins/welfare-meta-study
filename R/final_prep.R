library(tidyverse)

cols_keep = c("id","source","panel","Y0","Q0","t0","age","ageyng","numkids","arm","SOI","less_hs","hs","some_coll","coll","year","Q","unemp","cpi","chcare","pay_care","EARN","AFDC","FS","EMP","TOTINC","FSt","AFDCt","county","app_status")

# drop SIPP people who move state
#sipp <- read.csv("~/Dropbox/Research Projects/WelfareMetaAnalysis/Data/SIPP_prepped.csv") %>% mutate(X=NULL) %>%
sipp <- read.csv("~/Dropbox/WelfareMetaAnalysis/Data/SIPP_prepped.csv") %>% mutate(X=NULL) %>%
  filter(age<=50) %>%
  mutate(source = "SIPP",arm = 0,t0 = (Y0-panel)*4 + (Q0-1),county = 0,app_status = 0) %>%
  mutate(TOTINC = NA,FSt = NA, AFDCt = NA) %>%
  mutate(pay_care = chcare>0) %>%
  select(all_of(cols_keep)) %>%
  group_by(id) %>%
  filter(length(unique(SOI))==1) %>%
  ungroup()


#ftp <- read.csv("~/Dropbox/Research Projects/WelfareMetaAnalysis/Data/FTP_prepped.csv") %>% mutate(X=NULL) %>%
ftp <- read.csv("~/Dropbox/WelfareMetaAnalysis/Data/FTP_prepped.csv") %>% mutate(X=NULL) %>#%
  mutate(source = "FTP",ageyng = round(ageyng),t0 = Q0 - 1,county = 0,panel = 1994) %>%
  mutate(TOTINC = EARN+AFDCt+FSt) %>%
  filter(rQ>=0) %>%
  select(all_of(cols_keep))

#ctjf <- read.csv("~/Dropbox/Research Projects/WelfareMetaAnalysis/Data/CTJF_prepped.csv") %>% mutate(X=NULL) %>%
ctjf <- read.csv("~/Dropbox/WelfareMetaAnalysis/Data/CTJF_prepped.csv") %>% mutate(X=NULL) %>%
  mutate(source = "CTJF",t0 = (Y0-1996)*4 + Q0 - 1, panel = 1996) %>%
  mutate(TOTINC = EARN+AFDCt+FSt) %>%
  filter(rQ>=0) %>%
  select(all_of(cols_keep))

#mfip <- read.csv("~/Dropbox/Research Projects/WelfareMetaAnalysis/Data/MFIP_prepped.csv") %>% mutate(X=NULL) %>%
mfip <- read.csv("~/Dropbox/WelfareMetaAnalysis/Data/MFIP_prepped.csv") %>% mutate(X=NULL) %>%
  mutate(source = "MFIP",t0 = (Y0-1994)*4 + Q0 - 2,panel = 1994) %>%
  mutate(TOTINC = EARN+AFDCt+FSt) %>%
  filter(rQ>=0) %>%
  select(all_of(cols_keep))

# # 1 - create a case index for the SIPP that is unique for each individual:
# sipp <- sipp %>%
#   select(id) %>%
#   unique() %>%
#   mutate(case_idx = row_number()) %>%
#   inner_join(sipp)
# 
# case_max = max(sipp$case_idx)

data <- ftp %>%
  rbind(ctjf) %>%
  rbind(mfip) 

# data <- data %>%
#   select(source,SOI,arm,age,ageyng,numkids) %>%
#   unique() %>%
#   mutate(case_idx = case_max+row_number()) %>%
#   inner_join(data)


sipp %>%
  rbind(data) %>%
  mutate(EARN = EARN/3,TOTINC = TOTINC/3,FSt = FSt/3,AFDCt = AFDCt/3) %>% #<- convert quarterly to monthly earnings (for the sake of the budget function which takes monthly earnings)
  #arrange(case_idx,id,year,Q) %>%
  arrange(source,id,year,Q) %>%
  write.csv("~/Dropbox/Research Projects/WelfareMetaAnalysis/Data/Data_prepped.csv")


# ---- now compile the child data?
cols_keep = c("id","source","AGEKID","BPIE","BPIN","PBS","ENGAGE","REPEAT","SUSPEND","ACHIEVE","TCH_AVG")

ftp <- read.csv("~/Dropbox/Research Projects/WelfareMetaAnalysis/Data/FTP_child_prepped.csv") %>% mutate(X=NULL) %>%
  mutate(source = "FTP",TCH_AVG = NA,SUSPEND = SUSPEND / 100) %>%
  rename(id = SAMPLEID) %>%
  select(cols_keep)
ctjf <- read.csv("~/Dropbox/Research Projects/WelfareMetaAnalysis/Data/CTJF_child_prepped.csv") %>% mutate(X=NULL) %>%
  mutate(source = "CTJF") %>%
  mutate(REPEAT = REPEAT / 100,SUSPEND = SUSP / 100) %>% #<- the 1 here coded as 100 (some mistake in data import)
  rename(ENGAGE = ENGAG) %>%
  rename(id = PUFID) %>%
  select(cols_keep)
mfip <- read.csv("~/Dropbox/Research Projects/WelfareMetaAnalysis/Data/MFIP_child_prepped.csv") %>% mutate(X=NULL) %>%
  mutate(source = "MFIP",TCH_AVG = NA) %>%
  rename(id = PUBID) %>%
  select(cols_keep)

child <- ctjf %>%
  rbind(ftp) %>%
  rbind(mfip)
  
# --- save to file:
child %>%
  write.csv("~/Dropbox/Research Projects/WelfareMetaAnalysis/Data/Data_child_prepped.csv")

child %>%
  select(-TCH_AVG) %>%
  pivot_longer(cols=-c("id","source","AGEKID")) %>%
  group_by(source,AGEKID,name) %>%
  summarize(m = mean(value,na.rm = TRUE),s = sd(value,na.rm = TRUE)) %>%
  ggplot(aes(x=AGEKID,y=m,color=source)) + geom_line() + geom_point() + facet_wrap(. ~ name,scales="free_y")

child %>%
  select(-TCH_AVG,-REPEAT,-SUSPEND) %>%
  pivot_longer(cols=-c("id","source","AGEKID")) %>%
  group_by(source,AGEKID,name) %>%
  ggplot(aes(x=value,fill=source,color=source)) + geom_histogram(alpha=0.3) + facet_wrap(. ~ name,scales="free")

child %>%
  group_by(source,BPIE) %>%
  summarize(PBS = mean(PBS,na.rm = TRUE),N = n()) %>%
  ggplot(aes(x=BPIE,y=PBS,color=source,size=N)) + geom_point()


break
# ---- run a regression to construct an initial guess:
d <- read.csv("~/Dropbox/Research Projects/WelfareMetaAnalysis/Data/Data_prepped.csv")

library(fixest)

d %>%
  filter(EARN>0) %>%
  mutate(ageQ = (age-18)*4) %>%
  feols(log(EARN) ~ ageQ + unemp | id,data=.) %>%
  summary()

d %>%
  filter(EARN>0) %>%
  mutate(tQ = (year-1992)*4 + Q) %>%
  group_by(source,tQ) %>%
  summarize(logE = mean(log(EARN))) %>%
  ggplot(aes(x=tQ,y=logE,color=source)) + geom_line()
 