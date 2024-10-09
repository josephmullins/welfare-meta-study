library(tidyverse)

# this is mostly all set up to do the comparison with panel data

model <- read.csv("~/Dropbox/Research Projects/WelfareMetaAnalysis/julia/output/model_stats_K5.csv") %>%
#model <- read.csv("~/Dropbox/Research Projects/WelfareMetaAnalysis/julia/output/modelfit_exante.csv") %>%
  mutate(date = as.Date(paste(year, "-", 3 * Q, "-01", sep = ""))) %>%
  #select(-LOGINC,-LOGFULL) %>%
  select(-LOGFULL) %>%
  pivot_longer(cols=c("EMP","AFDC","EARN")) %>%
  mutate(case = "Model",sd = NA,est_sample = est_sample=="true") %>%
  select(-year,-Q)

#c_sub <- read.csv("~/Dropbox/Research Projects/WelfareMetaAnalysis/Code/julia/output/case_subset.csv")
c_sub <- read.csv("~/Dropbox/Research Projects/WelfareMetaAnalysis/Data/Data_child_prepped.csv") %>%
  select(source,id)

data <- read.csv("~/Dropbox/Research Projects/WelfareMetaAnalysis/Data/Data_prepped.csv") %>%
  filter(!is.na(FS),!is.na(AFDC),!is.na(EMP))

data %>%
  filter(EMP) %>%
  group_by(source,arm) %>%
  summarize(mean(chcare>0,na.rm = TRUE))

sipp <- data %>%
  filter(source=="SIPP") %>%
  mutate(EARN = replace_na(EARN,0.))
  #mutate(EARN = case_when(EMP ~ EARN,!EMP ~ 0.))

data <- data %>%
  inner_join(c_sub) %>%
  rbind(sipp) %>%
  mutate(est_sample = ! ((source=="MFIP" & arm>0 & county>1 & app_status==1) | (source=="CTJF" & arm>0 & county==1 & app_status==1))) %>%
  mutate(date = as.Date(paste(year, "-", 3 * Q, "-01", sep = ""))) %>%
  mutate(arm = as.factor(arm),app_status = as.factor(app_status)) %>%
  select(date,source,arm,app_status,est_sample,EMP,AFDC,EARN) %>%
  pivot_longer(cols = c("EMP","AFDC","EARN")) %>%
  group_by(date,source,arm,app_status,est_sample,name) %>%
  summarize(sd = sd(value,na.rm = TRUE)/sqrt(sum(!is.na(value))), value = mean(value,na.rm = TRUE)) %>%
  mutate(case = "Data") %>%
  as.data.frame()

# recode applicant status:
data <- data %>%
  mutate(applicant = case_when(source=="CTJF" & app_status==0 ~ "Recipient",source=="CTJF" & app_status==1 ~ "Applicant",
                               source=="FTP" & app_status==0 ~ "Recipient",source=="FTP" & app_status==1 ~ "Applicant",
                               source=="MFIP" & app_status==3 ~ "Recipient",source=="MFIP" & app_status==2 ~ "Re-applicant",source=="MFIP" & app_status==1 ~ "New Applicant",
                               source=="SIPP" ~ "Rep. sample"))
model <- model %>%
  mutate(applicant = case_when(source=="CTJF" & app_status==0 ~ "Recipient",source=="CTJF" & app_status==1 ~ "Applicant",
                               source=="FTP" & app_status==0 ~ "Recipient",source=="FTP" & app_status==1 ~ "Applicant",
                               source=="MFIP" & app_status==3 ~ "Recipient",source=="MFIP" & app_status==2 ~ "Re-applicant",source=="MFIP" & app_status==1 ~ "New Applicant",
                               source=="SIPP" ~ "Rep. sample"))

# now get treatment effects:
data_ctrl <- data %>%
  filter(arm==0) %>%
  select(-arm) %>%
  rename(value0 = value,sd0 = sd)

data_TE <- data %>%
  filter(arm==1 | arm==2) %>%
  merge(data_ctrl) %>%
  mutate(value = value-value0, sd = sqrt(sd^2+sd0^2))

model_ctrl <- model %>%
  filter(arm==0) %>%
  select(-arm) %>%
  rename(value0 = value)

model_TE <- model %>%
  filter(arm==1 | arm==2) %>%
  merge(model_ctrl) %>%
  mutate(value = value-value0,sd0 = NA)


# ------ Now let's make some graphs
colors = c("#221E1D","#63AA9C","#E9633B") # color scheme
cback = "#fff5EE"  ###"#f5f4ef"
th_ = theme_minimal() + theme(rect = element_rect(fill = cback),panel.background = element_rect(fill = cback, color=cback),legend.position = "bottom",axis.title.y=element_text(vjust=1.2),legend.background = element_rect(fill=cback,color=cback),panel.border = element_blank())

library(tikzDevice)


# graph 1: model fit of means:

g <- data %>%
  rbind(model) %>%
  filter(est_sample) %>%
  ggplot(aes(x=date,y=value,ymin=value-1.96*sd,ymax=value+1.96*sd,linetype=case,color=arm,fill=arm)) + geom_ribbon(color=NA,alpha=0.2) + geom_line() + facet_grid(name ~ source*applicant,scales="free_y") + th_ + scale_color_manual(name="Arm",values = colors) + scale_fill_manual(name="Arm",values = colors) + xlab("") + ylab("") + scale_linetype_discrete(name=NULL) + theme(strip.text = element_text(size=7,margin = margin(t=0,r=0,b=0,l=0)),axis.text = element_text(size=6,angle=45))

tikz(file = "~/Dropbox/Research Projects/WelfareMetaAnalysis/Figures/ModelFit.tex",width=6.2,height = 3)
print(g)
dev.off()

g <- data %>%
  rbind(model) %>%
  filter(est_sample) %>%
  ggplot(aes(x=date,y=value,ymin=value-1.96*sd,ymax=value+1.96*sd,linetype=case,color=arm,fill=arm)) + geom_ribbon(color=NA,alpha=0.2) + geom_line(size=1.2) + facet_grid(name ~ source*applicant,scales="free_y") + theme_minimal() + theme(legend.position = "bottom") + scale_color_manual(name="Arm",values = colors) + scale_fill_manual(name="Arm",values = colors) + xlab("") + ylab("") + scale_linetype_discrete(name=NULL) + theme(strip.text = element_text(size=7,margin = margin(t=0,r=0,b=0,l=0)),axis.text = element_text(size=6,angle=45))

tikz(file = "~/Dropbox/Research Projects/WelfareMetaAnalysis/Figures/ModelFitPaper.tex",width=7,height = 4.5)
print(g)
dev.off()


break

# graph 2: model fit of treatment effects:

g <- data_TE %>%
  rbind(model_TE) %>%
  filter(est_sample) %>%
  ggplot(aes(x=date,y=value,ymin=value-1.96*sd,ymax=value+1.96*sd,linetype=case,color=arm,fill=arm)) + geom_point(size=0.5) + geom_ribbon(color=NA,alpha=0.2) + geom_line() + facet_grid(name ~ source*applicant,scales="free_y") + scale_color_manual(name="Arm",values = colors[2:3]) + scale_fill_manual(name="Arm",values = colors[2:3]) + scale_linetype_discrete(name=NULL) + th_

break
# graph 3: fit of holdout sample treatment effects

g <- data %>%
  rbind(model) %>%
  filter(!est_sample) %>%
  ggplot(aes(x=date,y=value,ymin=value-1.96*sd,ymax=value+1.96*sd,linetype=case,color=arm,fill=arm)) + geom_point(size=0.5) + geom_ribbon(color=NA,alpha=0.2) + geom_line() + facet_grid(name ~ source*applicant,scales="free_y") + th_ + scale_color_manual(name="Arm",values = colors[2:3]) + scale_fill_manual(name="Arm",values = colors[2:3]) + ylab("") + xlab("")  + scale_linetype_discrete(name=NULL) + theme(axis.text.y = element_text(size=6))

tikz(file = "~/Dropbox/Research Projects/WelfareMetaAnalysis/Figures/OutModelFit.tex",width=5,height = 3)
print(g)
dev.off()



break
d %>%
  select(-eta2) %>%
  rbind(d2) %>%
  pivot_longer(cols = -c("source","arm","year","Q","date","case")) %>%
  mutate(arm = as.factor(arm)) %>%
  ggplot(aes(x=date,y = value,color=arm,linetype=case)) + facet_grid(name ~ source,scales = "free_y") + geom_point() + geom_line()

break
d %>%
  mutate(arm = as.factor(arm)) %>%
  ggplot(aes(x=date,y = eta2,color=arm,linetype=case)) + facet_grid(. ~ source) + geom_point() + geom_line()

#experimenting here with how to look at model fit

read.csv("~/Dropbox/Research Projects/WelfareMetaAnalysis/Data/Data_prepped.csv") %>%
  filter(!is.na(FS),!is.na(AFDC),!is.na(EMP)) %>%
  filter(source!="SIPP") %>%
  #inner_join(c_sub) %>%
  group_by(source,arm,year,Q,app_status) %>%
  summarize(EMP = mean(EMP,na.rm = TRUE),AFDC = mean(AFDC,na.rm = TRUE),EARN = mean(EARN,na.rm = TRUE)) %>%
  mutate(date = as.Date(paste(year, "-", 3 * Q, "-01", sep = ""))) %>%
  pivot_longer(cols = -c("source","arm","year","Q","date","app_status")) %>%
  mutate(arm = as.factor(arm),app_status = as.factor(app_status)) %>%
  ggplot(aes(x=date,y = value,color=app_status,linetype=arm)) + facet_grid(name ~ source,scales = "free_y") + geom_point() + geom_line()

read.csv("~/Dropbox/Research Projects/WelfareMetaAnalysis/Data/Data_prepped.csv") %>%
  filter(!is.na(FS),!is.na(AFDC),!is.na(EMP)) %>%
  filter(source!="SIPP",source!="FTP") %>%
  #inner_join(c_sub) %>%
  filter(! (source=="MFIP" & app_status==2)) %>%
  #mutate(county = pmin(county,2)) %>% #<- group anoka and dakota counties:
  group_by(source,arm,year,Q,app_status,county) %>%
  summarize(EMP = mean(EMP,na.rm = TRUE),AFDC = mean(AFDC,na.rm = TRUE),EARN = mean(EARN,na.rm = TRUE)) %>%
  mutate(date = as.Date(paste(year, "-", 3 * Q, "-01", sep = ""))) %>%
  pivot_longer(cols = -c("source","arm","year","Q","date","app_status","county")) %>%
  mutate(arm = as.factor(arm),app_status = as.factor(app_status)) %>%
  ggplot(aes(x=date,y = value,color=app_status,linetype=arm)) + facet_grid(name ~ source*county,scales = "free_y") + geom_point() + geom_line()


d2 %>%
  filter(est_sample) %>%
  ggplot(aes(x=date,y=value,ymin=value-1.96*sd,ymax=value+1.96*sd,linetype=arm)) + geom_point() + geom_ribbon(alpha=0.2) + geom_line() + facet_grid(name ~ source*app_status,scales="free_y")

TE %>%
  filter(est_sample) %>%
  ggplot(aes(x=date,y=value,ymin=value-1.96*sd,ymax=value+1.96*sd,linetype=arm)) + geom_point() + geom_ribbon(alpha=0.2) + geom_line() + facet_grid(name ~ source*app_status,scales="free_y") + geom_hline(yintercept = 0.,linetype = "dashed")


# ----- Check fit of log income:
d <- read.csv("~/Dropbox/Research Projects/WelfareMetaAnalysis/Data/Data_prepped.csv") %>%
  mutate(X=NULL) %>%
  merge(c_sub) %>%
  #filter(TOTINC>0) %>%
  #filter(EARN>0) %>%
  group_by(source,arm,app_status,year,Q) %>%
  #summarize(lY = mean(log(TOTINC))) %>%
  summarize(lY = mean(TOTINC)) %>%
  #summarize(lY = mean(log(EARN))) %>%
  #summarize(lY = mean(FSt+AFDCt)) %>%
  #summarize(lY = mean(EARN)) %>%
  mutate(case="data") %>%
  ungroup() %>%
  mutate(date = as.Date(paste(year, "-", 3 * Q, "-01", sep = ""))) %>%
  as.data.frame()
  
model <- read.csv("~/Dropbox/Research Projects/WelfareMetaAnalysis/Code/julia/output/model_stats_childsample.csv") %>%
  mutate(date = as.Date(paste(year, "-", 3 * (Q+1), "-01", sep = ""))) %>%
  select(source,arm,app_status,date,year,Q,LOGINC) %>%
  rename(lY = LOGINC) %>%
  mutate(case="model")

model %>%
  rbind(d) %>%
  ggplot(aes(x=date,y=lY,color=case,linetype=as.factor(arm))) + geom_line() + facet_grid(source ~ app_status)
  
# no net effect on income for CTJF and FTP
read.csv("~/Dropbox/Research Projects/WelfareMetaAnalysis/Code/julia/output/model_stats_childsample.csv") %>%
  mutate(date = as.Date(paste(year, "-", 3 * (Q+1), "-01", sep = "")))


read.csv("~/Dropbox/Research Projects/WelfareMetaAnalysis/Data/Data_prepped.csv") %>%
  mutate(X=NULL) %>%
  merge(c_sub) %>%
  mutate(date = as.Date(paste(year, "-", 3 * Q, "-01", sep = ""))) %>%
  group_by(source,arm,app_status,date) %>%
  #summarize(lY = mean(log(TOTINC))) %>%
  summarize(lY = mean(TOTINC==0)) %>%
  ggplot(aes(x=date,y=lY,linetype=as.factor(arm))) + geom_line() + facet_grid(source ~ app_status)



# data2 <- data %>%
#   inner_join(c_sub) %>%
#   rbind(sipp) %>%
#   mutate(date = as.Date(paste(year, "-", 3 * Q, "-01", sep = ""))) %>%
#   mutate(arm = as.factor(arm),app_status = as.factor(app_status),county = as.factor(county)) %>%
#   select(date,source,arm,app_status,county,EMP,AFDC,EARN) %>%
#   pivot_longer(cols = c("EMP","AFDC","EARN")) %>%
#   group_by(date,source,arm,app_status,county,name) %>%
#   summarize(sd = sd(value,na.rm = TRUE)/sqrt(sum(!is.na(value))), value = mean(value,na.rm = TRUE)) %>%
#   mutate(case = "Data") %>%
#   as.data.frame()
# 
# data2 %>%
#   ggplot(aes(x=date,y=value,ymin=value-1.96*sd,ymax=value+1.96*sd,color=arm,fill=arm)) + geom_ribbon(color=NA,alpha=0.2) + geom_line() + facet_grid(name ~ source*app_status*county,scales="free_y") + th_ + scale_color_manual(name="Arm",values = colors) + scale_fill_manual(name="Arm",values = colors) + xlab("") + ylab("") + scale_linetype_discrete(name=NULL) + theme(strip.text = element_text(size=7,margin = margin(t=0,r=0,b=0,l=0)),axis.text = element_text(size=6,angle=45))

#regression here to show a small employment effect?
data %>%
  filter(source=="MFIP") %>%
  filter(year==1998,Q==3) %>%
  mutate(arm = as.factor(arm),app_status = as.factor(app_status),county = as.factor(county)) %>%
  lm(EMP ~ arm + app_status*county,data=.) %>%
  summary()

