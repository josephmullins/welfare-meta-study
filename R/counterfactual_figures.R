library(tidyverse)

colors = c("#221E1D","#63AA9C","#E9633B") # color scheme
cback = "#fff5EE"  ###"#f5f4ef"
th_ = theme_minimal() + theme(rect = element_rect(fill = cback),panel.background = element_rect(fill = cback, color=cback),legend.position = "bottom",axis.title.y=element_text(vjust=1.2),legend.background = element_rect(fill=cback,color=cback),panel.border = element_blank())


d <- read.csv("output/decomp_counterfactual.csv") %>%
  mutate(date = as.Date(paste(year, "-", 3 * (Q+1), "-01", sep = "")))
  
g <- d %>%
  ggplot(aes(x=date,y=value,ymin=q25,ymax=q75,shape=case,color=case)) + 
  geom_line(linewidth=1.5) + 
  geom_point(size=3) +
  facet_grid(variable ~ source,scales="free_y") + 
  #geom_ribbon(alpha=0,color="black",linetype="dashed") + 
  geom_errorbar(width=0) +
  theme_minimal() + 
  theme(legend.position = "bottom",legend.title = element_blank()) + 
  ylab(NULL) + xlab(NULL)


ggsave("output/figures/DecompCounterfactualPaper.eps",width = 10, height = 8)

# -- now depict the non_selected counterfactual --- #
d <- read.csv("output/non_selected_counterfactual.csv") %>%
  select(-source) %>%
  rename(source = case) %>%
  mutate(case = "Non-Selected")

d0 <- read.csv("output/decomp_counterfactual.csv") %>%
  filter(case=="Treatment") %>%
  mutate(case = "Selected")

g <- d %>%
  rbind(d0) %>%
  mutate(date = as.Date(paste(year, "-", 3 * (Q+1), "-01", sep = ""))) %>%
  ggplot(aes(x=date,y=value,ymin=q25,ymax=q75,shape=case,color=case)) + 
  geom_line(linewidth=1.5) + geom_point(size=3) +
  facet_grid(variable ~ source,scales="free_y") + 
  geom_errorbar(width=0) +
  theme_minimal() + 
  theme(legend.position="bottom",legend.title = element_blank())+ 
  ylab(NULL) + xlab(NULL) + 
  scale_color_manual(name=NULL,values=colors[2:3]) +
  scale_shape(name=NULL)

ggsave("output/figures/non_selected_counterfactual.eps",width = 10, height = 8)

