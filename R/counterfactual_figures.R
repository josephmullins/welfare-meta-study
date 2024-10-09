library(tidyverse)
library(tikzDevice)

colors = c("#221E1D","#63AA9C","#E9633B") # color scheme
cback = "#fff5EE"  ###"#f5f4ef"
th_ = theme_minimal() + theme(rect = element_rect(fill = cback),panel.background = element_rect(fill = cback, color=cback),legend.position = "bottom",axis.title.y=element_text(vjust=1.2),legend.background = element_rect(fill=cback,color=cback),panel.border = element_blank())


d <- read.csv("../output/decomp_counterfactual.csv") %>%
  mutate(date = as.Date(paste(year, "-", 3 * (Q+1), "-01", sep = "")))
  
g <- d %>%
  ggplot(aes(x=date,y=value,ymin=q25,ymax=q75,color=case,fill=case)) + geom_line(linewidth=1.1) + facet_grid(variable ~ source,scales="free_y") + geom_ribbon(color=NA,alpha=0.1) + th_ + theme(legend.title = element_blank()) + ylab(NULL) + xlab(NULL)

tikz(file = "../output/figures/DecompCounterfactual.tex",width=5,height = 3.5)
print(g)
dev.off()

tikz(file = "../output/figures/DecompCounterfactualPaper.tex",width=6,height = 4)
print(g+theme_minimal() + theme(legend.position = "bottom",legend.title = element_blank()))
dev.off()



d <- read.csv("../output/decomp_counterfactual2.csv")

g <- d %>%
  ggplot(aes(x=case,y=value,ymin=q25,ymax=q75,color=case)) + geom_point() + geom_errorbar(size=1.1,width=0.5) + facet_grid(variable ~ source,scales="free_y") + th_ + theme(axis.text.x = element_blank(),legend.title = element_blank(),strip.text.y = element_text(size=8)) + ylab("\\%") + xlab(NULL) + geom_hline(yintercept = 0,alpha=0.8,linetype="dashed")

tikz(file = "../output/figures/DecompCounterfactual2.tex",width=5,height = 3.5)
print(g)
dev.off()

# -- now depict the non_selected counterfactual --- #s
d <- read.csv("../output/non_selected_counterfactual.csv") %>%
  select(-source) %>%
  rename(source = case) %>%
  mutate(case = "Non-Selected")

d0 <- read.csv("../output/decomp_counterfactual.csv") %>%
  filter(case=="Treatment") %>%
  mutate(case = "Selected")

g <- d %>%
  rbind(d0) %>%
  mutate(date = as.Date(paste(year, "-", 3 * (Q+1), "-01", sep = ""))) %>%
  ggplot(aes(x=date,y=value,ymin=q25,ymax=q75,color=case,fill=case)) + geom_line(size=1.2) + facet_grid(variable ~ source) + geom_ribbon(color=NA,alpha=0.1) + th_ + theme(legend.title = element_blank()) + ylab(NULL) + xlab(NULL) + scale_color_manual(name=NULL,values=colors[2:3]) + scale_fill_manual(name=NULL,values=colors[2:3])

tikz(file = "../output/figures/non_selected_counterfactual.tex",width=5,height = 4)
print(g + theme_minimal() + theme(legend.position = "bottom"))
dev.off()


