library(tidyverse)

# --- Part 1: In-sample model fit

model <- read.csv("output/model_stats_K5.csv") %>%
  mutate(date = as.Date(paste(year, "-", 3 * Q, "-01", sep = ""))) %>%
  select(-LOGFULL) %>%
  pivot_longer(cols=c("EMP", "AFDC", "EARN")) %>%
  mutate(case = "Model", sd = NA, est_sample = est_sample == "true") %>%
  select(-year, -Q)



c_sub <- read.csv("../Data/Data_child_prepped.csv") %>%
  select(source, id)

data <- read.csv("../Data/Data_prepped.csv") %>%
  filter(!is.na(FS), !is.na(AFDC), !is.na(EMP))

data %>%
  filter(EMP) %>%
  group_by(source, arm) %>%
  summarize(mean(chcare > 0, na.rm = TRUE))

sipp <- data %>%
  filter(source == "SIPP") %>%
  mutate(EARN = replace_na(EARN,0.))

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

# paper friendly version of model fit
g <- data %>%
  rbind(model) %>%
  filter(est_sample) %>%
  filter(name!="EARN") %>%
  ggplot(aes(x=date,y=value,ymin=value-1.96*sd,ymax=value+1.96*sd,linetype=case,shape=arm,color=arm)) + 
  #geom_ribbon(color=NA,alpha=0.2) + geom_line(size=2) + 
  geom_errorbar(width = 0.) + geom_line(size=1.5) + 
  #geom_point(size=3) +
  facet_grid(source*applicant ~ name,scales="free_y") + 
  theme_minimal() + theme(legend.position = "bottom") + 
  scale_shape_discrete(name="Arm") + 
  scale_color_manual(name="Arm",values = colors) + 
  scale_linetype_discrete(name=NULL) + 
  xlab("") + ylab("") + 
  theme(strip.text = element_text(size=7,margin = margin(t=0,r=0,b=0,l=0)),axis.text = element_text(size=6,angle=45))

ggsave("output/figures/ModelFitPaper.eps",width = 7, height = 9)


# ------- Part 2: model fit out of sample:
model <- read.csv("output/modelfit_exante_K5.csv") %>%
  mutate(date = as.Date(paste(year, "-", 3 * (Q+1), "-01", sep = ""))) %>%
  select(-LOGFULL) %>%
  pivot_longer(cols=c("EMP","AFDC","EARN")) %>%
  mutate(case = "Model",sd = NA,est_sample = est_sample=="true") %>%
  select(-year,-Q)

model_ctrl <- model %>%
  filter(arm==0) %>%
  select(-arm) %>%
  rename(value0 = value)

model_TE <- model %>%
  filter(arm==1 | arm==2) %>%
  merge(model_ctrl) %>%
  mutate(value = value-value0,sd0 = NA)

g <- data %>%
  select(-c("app_status","applicant")) %>%
  rbind(model) %>%
  filter(!est_sample) %>%
  ggplot(aes(x=date,y=value,ymin=value-1.96*sd,ymax=value+1.96*sd,linetype=case)) + 
  geom_errorbar(width=0) +
  geom_line(size=2) + 
  facet_grid(name ~ source*arm,scales="free_y") + theme_minimal() + 
  theme(legend.position = "bottom") + 
  scale_color_manual(name="Arm",values = colors[2:3]) + 
  scale_fill_manual(name="Arm",values = colors[2:3]) + 
  scale_shape_discrete(name=NULL) + 
  ylab("") + xlab("")  + 
  scale_linetype_discrete(name=NULL) + 
  theme(axis.text.y = element_text(size=6))

ggsave("output/figures/OutModelFitPaper.eps",width = 8,height = 8)
