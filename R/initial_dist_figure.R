colors = c("#221E1D","#63AA9C","#E9633B") # color scheme
cback = "#fff5EE"  ###"#f5f4ef"
th_ = theme_minimal() + theme(rect = element_rect(fill = cback),panel.background = element_rect(fill = cback, color=cback),legend.position = "bottom",axis.title.y=element_text(vjust=1.2),legend.background = element_rect(fill=cback,color=cback),panel.border = element_blank())

library(tikzDevice)
library(tidyverse)
d <- read.csv("~/Dropbox/Research Projects/WelfareMetaAnalysis/julia/output/initial_dists.csv")

# make the figure for slides
g <- d %>%
  ggplot(aes(x=value,y=dist,fill = source,color = source,stat_count(identity))) + geom_bar(stat = "identity",position = position_dodge()) + facet_grid(. ~ var,scales="free_x") + th_ + scale_color_manual(name=NULL,values=colors[2:3]) + scale_fill_manual(name=NULL,values=colors[2:3]) + ylab("Prob.") + xlab(NULL)

tikz(file = "~/Dropbox/Research Projects/WelfareMetaAnalysis/Figures/InitialDistributions.tex",width=4,height = 3)
print(g)
dev.off()

# make the figure for paper
g <- d %>%
  ggplot(aes(x=value,y=dist,fill = source,color = source)) + geom_bar(stat = "identity",position = position_dodge()) + facet_grid(. ~ var,scales="free_x") + theme_minimal() + scale_color_manual(name=NULL,values=colors[2:3]) + scale_fill_manual(name=NULL,values=colors[2:3]) + ylab("Prob.") + xlab(NULL)

g <- d %>%
  ggplot(aes(x=value,y=dist)) + geom_bar(stat = "identity",position = position_dodge()) + facet_grid(source ~ var,scales="free_x") + theme_minimal() + ylab("Prob.") + xlab(NULL)


tikz(file = "~/Dropbox/Research Projects/WelfareMetaAnalysis/Figures/InitialDistributions-Paper.tex",width=4,height = 4)
print(g)
dev.off()


