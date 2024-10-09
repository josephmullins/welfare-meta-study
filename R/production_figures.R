library(tidyverse)
library(tikzDevice)

colors = c("#221E1D","#63AA9C","#E9633B") # color scheme
cback = "#fff5EE"  ###"#f5f4ef"
th_ = theme_minimal() + theme(rect = element_rect(fill = cback),panel.background = element_rect(fill = cback, color=cback),legend.position = "bottom",axis.title.y=element_text(vjust=1.2),legend.background = element_rect(fill=cback,color=cback),panel.border = element_blank())


d <- read.csv("../output/production_ests.csv")
names(d) <- c("dI","dth","g1","g2","version","skill")
d <- d %>%
  mutate(version = case_when(version=="mle" ~ "control function",version=="iv" ~ "iv"))


g <- d %>%
  ggplot(aes(x=dI,color=version,fill=version)) + geom_density(bw=0.01,alpha=0.3) + th_ + facet_grid(. ~ skill) + xlim(0,0.4) + scale_color_manual(name=NULL,values=colors[2:3]) + scale_fill_manual(name=NULL,values=colors[2:3]) + ylab(NULL) + xlab("$\\delta_{I}$")

tikz(file = "../output/figures/dI_ests",width=5,height = 3.5)
print(g)
dev.off()

# --- version for the paper ----- #

g <- d %>%
  ggplot(aes(x=dI,color=version,fill=version)) + geom_density(bw=0.01,alpha=0.3) + theme_minimal() + theme(legend.position = "bottom") + facet_grid(. ~ skill) + xlim(0,0.3) + scale_color_manual(name=NULL,values=colors[2:3]) + scale_fill_manual(name=NULL,values=colors[2:3]) + ylab(NULL) + xlab("$\\delta_{I}$")

tikz(file = "../output/figures/dI_ests_paper",width=5,height = 3.5)
print(g)
dev.off()



g <- d %>%
  ggplot(aes(x=g1,color=version,fill=version)) + geom_density(bw = 0.01,alpha=0.3) + th_ + facet_grid(skill ~ .) + scale_color_manual(name=NULL,values=colors[2:3]) + scale_fill_manual(name=NULL,values=colors[2:3]) + ylab(NULL) + xlab("${g}_{1}$") + xlim(-0.75,0.75) + geom_vline(xintercept = 0,linetype="dashed",alpha=0.5) + theme(axis.text.y = element_blank())

tikz(file = "../output/figures/g1_ests",width=5,height = 3.5)
print(g)
dev.off()

g <- d %>%
  ggplot(aes(x=g1,color=version,fill=version)) + geom_density(bw = 0.01,alpha=0.3) + theme_minimal() + theme(legend.position = "bottom") + facet_grid(skill ~ .) + scale_color_manual(name=NULL,values=colors[2:3]) + scale_fill_manual(name=NULL,values=colors[2:3]) + ylab(NULL) + xlab("${g}_{1}$") + xlim(-0.75,0.75) + geom_vline(xintercept = 0,linetype="dashed",alpha=0.5) + theme(axis.text.y = element_blank())

tikz(file = "../output/figures/g1_ests_paper",width=5,height = 3.5)
print(g )
dev.off()



g <- d %>%
  ggplot(aes(x=g2,color=version,fill=version)) + geom_density(bw = 0.01,alpha=0.3) + th_ + facet_grid(skill ~ .) + scale_color_manual(name=NULL,values=colors[2:3]) + scale_fill_manual(name=NULL,values=colors[2:3]) + ylab(NULL) + xlab("${g}_{2}$") + geom_vline(xintercept = 0,linetype="dashed",alpha=0.5) + theme(axis.text.y = element_blank())
tikz(file = "../output/figures/g2_ests",width=5,height = 3.5)
print(g + xlim(-0.75,0.75))
dev.off()

g <- d %>%
  ggplot(aes(x=g2,color=version,fill=version)) + geom_density(bw = 0.01,alpha=0.3) + theme_minimal() + theme(legend.position = "bottom") + facet_grid(skill ~ .) + scale_color_manual(name=NULL,values=colors[2:3]) + scale_fill_manual(name=NULL,values=colors[2:3]) + ylab(NULL) + xlab("${g}_{2}$") + geom_vline(xintercept = 0,linetype="dashed",alpha=0.5) + theme(axis.text.y = element_blank())
tikz(file = "../output/figures/g2_ests_paper",width=5,height = 3.5)
print(g + xlim(-0.75,0.75))
dev.off()


g <- d %>%
  ggplot(aes(x=dth,color=version,fill=version)) + geom_density(bw=0.05,alpha=0.3) + th_ + facet_grid(. ~ skill) + scale_color_manual(name=NULL,values=colors[2:3]) + scale_fill_manual(name=NULL,values=colors[2:3]) + ylab(NULL) + xlab("$\\delta_{\\theta}$")


tikz(file = "../output/figures/dth_ests",width=5,height = 3.5)
print(g)
dev.off()

g <- d %>%
  ggplot(aes(x=dth,color=version,fill=version)) + geom_density(bw=0.05,alpha=0.3) + theme_minimal() + theme(legend.position = "bottom") + facet_grid(. ~ skill) + scale_color_manual(name=NULL,values=colors[2:3]) + scale_fill_manual(name=NULL,values=colors[2:3]) + ylab(NULL) + xlab("$\\delta_{\\theta}$")


tikz(file = "../output/figures/dth_ests_paper",width=6,height = 3.5)
print(g)
dev.off()



d <- read.csv("../output/production_ests_hetero.csv")
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
  ggplot(aes(x=value)) + geom_density(color=colors[2],fill = colors[2],alpha=0.5,bw = 0.005) + facet_grid(Type ~ skill*parameter) + th_ + ylab(NULL) + xlab("${g}_{1}$") + geom_vline(xintercept = 0,linetype="dashed",alpha=0.5) + theme(axis.text.y = element_blank())

tikz(file = "../output/figures/g_est_hetero.tex",width=5,height = 3.)
print(g)
dev.off()

tikz(file = "../output/figures/g_est_hetero_paper.tex",width=6.5,height = 4.)
print(g+theme_minimal()+theme(legend.position = "bottom"))
dev.off()


g <- d %>%
  select(starts_with("g1"),skill) %>%
  pivot_longer(cols = -skill) %>%
  mutate(Type = case_when(name=="g1_1" ~ 1,name=="g1_2" ~ 2,name=="g1_3" ~ 3,name=="g1_4" ~ 4,name=="g1_5" ~ 5)) %>%
  ggplot(aes(x=value)) + geom_density(color=colors[2],fill = colors[2],alpha=0.5,bw = 0.005) + facet_grid(Type ~ skill) + th_ + ylab(NULL) + xlab("${g}_{1}$") + geom_vline(xintercept = 0,linetype="dashed",alpha=0.5) + theme(axis.text.y = element_blank())

tikz(file = "../output/figures/g1_est_hetero.tex",width=4,height = 3.)
print(g)
dev.off()

tikz(file = "../output/figures/g1_est_hetero_paper.tex",width=4,height = 4.)
print(g+theme_minimal()+theme(legend.position = "bottom"))
dev.off()


g <- d %>%
  select(starts_with("g2"),skill) %>%
  pivot_longer(cols = -skill) %>%
  mutate(Type = case_when(name=="g2_1" ~ 1,name=="g2_2" ~ 2,name=="g2_3" ~ 3,name=="g2_4" ~ 4,name=="g2_5" ~ 5)) %>%
  ggplot(aes(x=value)) + geom_density(color=colors[2],fill = colors[2],alpha=0.5,bw = 0.005) + facet_grid(Type ~ skill) + th_ + ylab(NULL) + xlab("${g}_{2}$") + geom_vline(xintercept = 0,linetype="dashed",alpha=0.5) + theme(axis.text.y = element_blank())

tikz(file = "../output/figures/g2_est_hetero.tex",width=4,height = 3.)
print(g)
dev.off()

tikz(file = "../output/figures/g2_est_hetero_paper.tex",width=4,height = 4.)
print(g+theme_minimal()+theme(legend.position = "bottom"))
dev.off()