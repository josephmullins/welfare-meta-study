library(tidyverse)

model <- read.csv("~/Dropbox/Research Projects/WelfareMetaAnalysis/julia/output/modelfit_exante_K5.csv") %>%
  filter(est_sample=="true") %>%
  mutate(date = as.Date(paste(year, "-", 3 * (Q+1), "-01", sep = ""))) %>%
  select(-LOGFULL) %>%
  pivot_longer(cols=c("EMP","AFDC","EARN")) %>%
  mutate(case = "Full Sample") %>%
  select(-year,-Q)

model2 <- read.csv("~/Dropbox/Research Projects/WelfareMetaAnalysis/julia/output/modelfit_exante_K5_noexp.csv") %>%
  filter(est_sample=="true") %>%
  mutate(date = as.Date(paste(year, "-", 3 * (Q+1), "-01", sep = ""))) %>%
  select(-LOGFULL) %>%
  pivot_longer(cols=c("EMP","AFDC","EARN")) %>%
  mutate(case = "No Experimental Data") %>%
  select(-year,-Q)

model <- model %>%
  rbind(model2)

model_ctrl <- model %>%
  filter(arm==0) %>%
  select(-arm) %>%
  rename(value0 = value)

model_TE <- model %>%
  filter(arm==1 | arm==2) %>%
  merge(model_ctrl) %>%
  mutate(value = value-value0)

colors = c("#221E1D","#63AA9C","#E9633B") # color scheme
cback = "#fff5EE"  ###"#f5f4ef"
th_ = theme_minimal() + theme(rect = element_rect(fill = cback),panel.background = element_rect(fill = cback, color=cback),legend.position = "bottom",axis.title.y=element_text(vjust=1.2),legend.background = element_rect(fill=cback,color=cback),panel.border = element_blank())

library(tikzDevice)


g <- model_TE %>%
  group_by(source,arm,name,case) %>%
  filter(row_number()<n()) %>%
  ggplot(aes(x=date,y=value,linetype=case,color=as.factor(arm))) + geom_line() + facet_grid(name ~ source)+ geom_line(size=1.2) + facet_grid(name ~ source,scales="free_y") + scale_color_manual(name="Arm",values = colors[2:3]) + scale_linetype_discrete(name=NULL) + theme_minimal() + theme(legend.position = "bottom") + geom_hline(yintercept = 0, linetype = "dashed") + ylab("") + xlab("Date")

tikz(file = "~/Dropbox/Research Projects/WelfareMetaAnalysis/Figures/NoExpData.tex",width=5,height = 4)
print(g)
dev.off()

