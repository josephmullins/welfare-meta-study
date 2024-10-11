library(tidyverse)
library(tikzDevice)

colors = c("#221E1D","#63AA9C","#E9633B") # color scheme
cback = "#fff5EE"  ###"#f5f4ef"
th_ = theme_minimal() + theme(rect = element_rect(fill = cback),panel.background = element_rect(fill = cback, color=cback),legend.position = "bottom",axis.title.y=element_text(vjust=1.2),legend.background = element_rect(fill=cback,color=cback),panel.border = element_blank())


d <- read.csv("output/production_ests.csv")
names(d) <- c("dI","dth","g1","g2","version","skill")
d <- d %>%
  mutate(version = case_when(version=="mle" ~ "control function",version=="iv" ~ "iv"))



g <- d %>%
  ggplot(aes(x=dI,color=version,fill=version,linetype=version)) + 
  geom_density(bw=0.01,linewidth=2,alpha=0.3) + theme_minimal() + 
  theme(legend.position = "bottom") + 
  facet_grid(. ~ skill) + xlim(0,0.3) + 
  scale_color_manual(name=NULL,values=colors[2:3]) + 
  scale_fill_manual(name=NULL,values=colors[2:3]) + 
  scale_linetype(name=NULL) +
  ylab(NULL) + xlab(NULL)

# tikz(file = "output/figures/dI_ests_paper.tex",width=5,height = 3.5)
# print(g)
# dev.off()


ggsave("output/figures/dI_ests_paper.eps",width=5,height = 3.5)

g <- d %>%
  ggplot(aes(x=g1,color=version,linetype=version)) + 
  geom_density(bw = 0.01,linewidth=2) + theme_minimal() + 
  theme(legend.position = "bottom") + facet_grid(. ~ skill) + 
  scale_color_manual(name=NULL,values=colors[2:3]) + 
  scale_linetype(name=NULL) +
  ylab(NULL) + xlab(NULL) + xlim(-0.75,0.75) + 
  geom_vline(xintercept = 0,linetype="dashed") + 
  theme(axis.text.y = element_blank())

ggsave("output/figures/g1_ests_paper.eps",width = 6,height = 4)

g <- d %>%
  ggplot(aes(x=g2,color=version,linetype=version)) + 
  geom_density(bw = 0.01,linewidth=2) + theme_minimal() + 
  theme(legend.position = "bottom") + facet_grid(. ~ skill) + 
  scale_color_manual(name=NULL,values=colors[2:3]) + 
  scale_linetype(name=NULL) +
  ylab(NULL) + xlab(NULL) + xlim(-0.75,0.75) + 
  geom_vline(xintercept = 0,linetype="dashed") + 
  theme(axis.text.y = element_blank())
ggsave("output/figures/g2_ests_paper.eps",width = 6,height = 4)

g <- d %>%
  ggplot(aes(x=dth,color=version,fill=version)) + 
  geom_density(bw=0.05,alpha=0.3,linewidth=2) + theme_minimal() + 
  theme(legend.position = "bottom") + facet_grid(. ~ skill) + 
  scale_color_manual(name=NULL,values=colors[2:3]) + 
  scale_fill_manual(name=NULL,values=colors[2:3]) + 
  scale_linetype(name=NULL) +
  ylab(NULL) + xlab(NULL)

ggsave("output/figures/dth_ests_paper.eps",width = 6,height = 3.5)


d <- read.csv("output/production_ests_hetero.csv")
names(d) <- c("dI","dth","g1_1","g1_2","g1_3","g1_4","g1_5","g2_1","g2_2","g2_3","g2_4","g2_5","skill")

d1 <- d %>%
  select(starts_with("g1"),skill) %>%
  pivot_longer(cols = -skill) %>%
  mutate(Type = case_when(name=="g1_1" ~ 1,name=="g1_2" ~ 2,name=="g1_3" ~ 3,name=="g1_4" ~ 4,name=="g1_5" ~ 5),parameter = "${g}_{1}$" ) 

d2 <- d %>%
  select(starts_with("g2"),skill) %>%
  pivot_longer(cols = -skill) %>%
  mutate(Type = case_when(name=="g2_1" ~ 1,name=="g2_2" ~ 2,name=="g2_3" ~ 3,name=="g2_4" ~ 4,name=="g2_5" ~ 5),parameter = "${g}_{2}$")

g <- d1 %>%
  rbind(d2) %>%
  ggplot(aes(x=value)) + 
  geom_density(color=colors[2],fill = colors[2],alpha=0.5,bw = 0.005,linewidth=2) + 
  facet_grid(Type ~ skill*parameter) + th_ + 
  ylab(NULL) + xlab(NULL) + 
  geom_vline(xintercept = 0,linetype="dashed",alpha=0.5) + 
  theme(axis.text.y = element_blank())

g <- g + theme_minimal() + theme(legend.position = "bottom")
tikz(file = "output/figures/g_est_hetero_paper.tex",width=6,height = 8)
print(g)
dev.off()
#ggsave("output/figures/g_est_hetero_paper.pdf",width=6,height = 8)
