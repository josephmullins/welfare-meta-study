library(tidyverse)

# this is mostly all set up to do the comparison with panel data

#model <- read.csv("~/Dropbox/Research Projects/WelfareMetaAnalysis/julia/output/model_stats_K5.csv") %>%
model <- read.csv("~/Dropbox/Research Projects/WelfareMetaAnalysis/julia/output/modelfit_exante_K5.csv") %>%
  mutate(date = as.Date(paste(year, "-", 3 * (Q+1), "-01", sep = ""))) %>%
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
  mutate(date = as.Date(paste(year, "-", 3 * (Q+1), "-01", sep = ""))) %>%
  mutate(arm = as.factor(arm),app_status = as.factor(app_status)) %>%
  select(date,source,arm,app_status,est_sample,EMP,AFDC,EARN) %>%
  pivot_longer(cols = c("EMP","AFDC","EARN")) %>%
  group_by(date,source,arm,est_sample,name) %>%
  summarize(sd = sd(value,na.rm = TRUE)/sqrt(sum(!is.na(value))), value = mean(value,na.rm = TRUE)) %>%
  mutate(case = "Data") %>%
  as.data.frame()


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
  ggplot(aes(x=date,y=value,ymin=value-1.96*sd,ymax=value+1.96*sd,linetype=case,color=arm,fill=arm)) + geom_ribbon(color=NA,alpha=0.2) + geom_line() + facet_grid(name ~ source,scales="free_y") + th_ + scale_color_manual(name="Arm",values = colors) + scale_fill_manual(name="Arm",values = colors) + xlab("") + ylab("") + scale_linetype_discrete(name=NULL) + theme(strip.text = element_text(size=7,margin = margin(t=0,r=0,b=0,l=0)),axis.text = element_text(size=6,angle=45))

g

break
tikz(file = "~/Dropbox/Research Projects/WelfareMetaAnalysis/Figures/ModelFitExAnte.tex",width=6.2,height = 3)
print(g)
dev.off()



# graph 2: model fit of treatment effects:

g <- data_TE %>%
  rbind(model_TE) %>%
  filter(est_sample) %>%
  ggplot(aes(x=date,y=value,ymin=value-1.96*sd,ymax=value+1.96*sd,linetype=case,color=arm,fill=arm)) + geom_point(size=0.5) + geom_ribbon(color=NA,alpha=0.2) + geom_line() + facet_grid(name ~ source,scales="free_y") + scale_color_manual(name="Arm",values = colors[2:3]) + scale_fill_manual(name="Arm",values = colors[2:3]) + scale_linetype_discrete(name=NULL) + th_

break
# graph 3: fit of holdout sample treatment effects

g <- data %>%
  rbind(model) %>%
  filter(!est_sample) %>%
  ggplot(aes(x=date,y=value,ymin=value-1.96*sd,ymax=value+1.96*sd,linetype=case,color=arm,fill=arm)) + geom_point(size=0.5) + geom_ribbon(color=NA,alpha=0.2) + geom_line() + facet_grid(name ~ source,scales="free_y") + th_ + scale_color_manual(name="Arm",values = colors[2:3]) + scale_fill_manual(name="Arm",values = colors[2:3]) + ylab("") + xlab("")  + scale_linetype_discrete(name=NULL) + theme(axis.text.y = element_text(size=6))

tikz(file = "~/Dropbox/Research Projects/WelfareMetaAnalysis/Figures/OutModelFit.tex",width=5,height = 3)
print(g)
dev.off()

# version for paper:

g <- data %>%
  rbind(model) %>%
  filter(!est_sample) %>%
  ggplot(aes(x=date,y=value,ymin=value-1.96*sd,ymax=value+1.96*sd,linetype=case,color=arm,fill=arm)) + geom_point(size=0.5) + geom_ribbon(color=NA,alpha=0.2) + geom_line(size=1.2) + facet_grid(name ~ source,scales="free_y") + theme_minimal() + theme(legend.position = "bottom") + scale_color_manual(name="Arm",values = colors[2:3]) + scale_fill_manual(name="Arm",values = colors[2:3]) + ylab("") + xlab("")  + scale_linetype_discrete(name=NULL) + theme(axis.text.y = element_text(size=6))

tikz(file = "~/Dropbox/Research Projects/WelfareMetaAnalysis/Figures/OutModelFitPaper.tex",width=5,height = 5)
print(g)
dev.off()


break
# junk here:

read.csv("~/Dropbox/Research Projects/WelfareMetaAnalysis/julia/output/statefit.csv") %>%
  filter(est_sample=="true") %>%
  mutate(date = as.Date(paste(year, "-", 3 * (Q+1), "-01", sep = ""))) %>%
  pivot_longer(cols=c("E","E1","E2","E3","E4","J","W")) %>%
  ggplot(aes(x=date,y=value,color=as.factor(arm),linetype=case)) + geom_line() + facet_grid(name ~ source,scales="free_y")

unemp <- read.csv("~/Dropbox/Research Projects/WelfareMetaAnalysis/Data/StateUnemployment.csv") %>% mutate(X=NULL) %>%
  mutate(Q = floor((month-1)/3)+1) %>%
  group_by(fips,year,Q) %>%
  filter(row_number()==1) %>% #<- take the number from first month of the quarter
  mutate(StateFIPS = as.integer(fips)) %>%
  select(StateFIPS,year,Q,unemp)

unemp %>%
  mutate(date = as.Date(paste(year, "-", 3 * Q, "-01", sep = ""))) %>%
  filter(StateFIPS==9 | StateFIPS==12 | StateFIPS==27) %>%
  ggplot(aes(x=date,y=unemp,color=as.factor(StateFIPS))) + geom_line(size=1.3)
  
S <- d %>%
  select(abstheta,absalpha,absbeta,absgamma) %>%
  drop_na() %>%
  cov()

m <- d %>%
  select(treat,abstheta,absalpha,absbeta,absgamma) %>%
  drop_na() %>%
  group_by(treat) %>%
  pivot_longer(cols=-treat) %>%
  group_by(treat,name) %>%
  summarize(m = mean(value)) %>%
  pivot_wider(id_cols = name,values_from = m,names_from = treat,names_prefix="value") %>%
  mutate(diff = value1 - value0)

n <- d %>%
  select(treat,abstheta,absalpha,absbeta,absgamma) %>%
  drop_na() %>%
  group_by(treat) %>%
  summarize(n = n())

dm = m$diff
S = S * (1/n$n[1] + 1/n$n[2])
chisq = t(dm) %*% solve(S) %*% dm
