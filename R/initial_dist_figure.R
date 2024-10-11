colors = c("#221E1D","#63AA9C","#E9633B") # color scheme
cback = "#fff5EE"  ###"#f5f4ef"
th_ = theme_minimal() + theme(rect = element_rect(fill = cback),panel.background = element_rect(fill = cback, color=cback),legend.position = "bottom",axis.title.y=element_text(vjust=1.2),legend.background = element_rect(fill=cback,color=cback),panel.border = element_blank())

library(tikzDevice)
library(tidyverse)
d <- read.csv("output/initial_dists.csv")

g <- d %>%
  ggplot(aes(x=value,y=dist)) + geom_bar(stat = "identity",position = position_dodge()) + facet_grid(source ~ var,scales="free_x") + theme_minimal() + ylab("Prob.") + xlab(NULL)


ggsave("output/figures/InitialDistributions-Paper.eps",width=4,height = 4)
