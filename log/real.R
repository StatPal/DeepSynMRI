pred <- as.matrix(read.csv("values/real-performances.csv", header=F))


perf_LS <- pred[c(6,7,3), ]
perf_MLE <- pred[c(13,14,10), ]
perf_DL_LS <- pred[c(20,21,17), ]
perf_DL_MLE <- pred[c(27,28,24), ]


tmp_LS <- (perf_DL_LS / perf_LS - 1) * 100
tmp_LS[1:2,] <- -tmp_LS[1:2,]

xtable::xtable(tmp_LS)


perf_new <- rbind(perf_LS, perf_MLE, perf_DL_LS, perf_DL_MLE)

method <- rep(c("LS", "MLE"), each=3, times=2)  # The second will be 3
measures <- rep(c("MAPE", "RMSPE", "SSIM"), 2 * 2)
DL <- rep(c(F, T), each = 3 * 2)


all_dat <- data.frame(measures, method, DL, perf_new)

rm(DL)
rm(measures)


## Plots
library(tidyverse)
library(tidyr)
library(ggplot2)
library(xtable)

colnames(all_dat)[4:12] <- paste0('Test image ', 1:9)
all_dat_check <- tibble(all_dat)
all_dat_old <- all_dat_check






tmp <- all_dat_check %>%
  pivot_longer(!c(DL, measures, method), names_to = "img", values_to = "vals") %>% 
  filter(measures == "MAPE") %>% 
  pivot_wider(names_from = img, values_from = vals) %>% 
  select(-c(measures)) %>% 
  arrange(method) %>%
  relocate(DL, .after=method) %>% 
  mutate(DL = ifelse(DL, "DL", "")) %>% 
  mutate(method = interaction(DL, method)) %>% 
  mutate_if(is.numeric, ~ . * 100) %>%
  select(-DL)

print(xtable(tmp, digits=c(1,0,2,2,2,2,2,2,2,2,2)), include.rownames=FALSE)








library(stringr)

TE_TR <- c("15/0.6", "20/0.6", "10/1", "30/1", "40/1", "10/2", "40/2", "60/3", "100/3")
TE_TR_names  <- paste0('TE/TR = ', TE_TR[1:9])

tmp_dat_old <- all_dat %>%
  pivot_longer(!c(DL, measures, method), names_to = "img", values_to = "vals") %>% 
  # filter(img != "Test image 6") %>%   # Remove 6-th test image as it ewas flagged
  mutate(method_old = method) %>%
  mutate(method = ifelse(DL, paste0("DL+", method), method )) 

tmp_dat <- tmp_dat_old
tmp_dat$img_vals  <-  TE_TR_names[as.numeric( str_split_fixed(tmp_dat_old$img, ' ', n=3)[,3] )]



library(RColorBrewer)
my_cols <-  brewer.pal(5, 'Dark2')
new_styles <- 15:18



tmp_dat %>%
  filter(measures == "MAPE") %>%
  ggplot() + 
    aes(x = img, y = vals, shape=method, group = method, linetype = method) +
    geom_point(size=3) + 
    scale_shape_manual(values = new_styles[1:4]) + 
    scale_colour_manual(values=my_cols) + 
    geom_line()




p <- tmp_dat %>%
  filter(measures == "MAPE") %>%
  ggplot() + 
    aes(x = img_vals, y = vals, shape=method, group = method, linetype = method, color = method) +
    geom_point(size=3) + 
    scale_shape_manual(values = new_styles[1:4]) + 
    scale_colour_manual(values=my_cols) + 
    geom_line() +
    theme_minimal() + theme(panel.background = element_rect(fill = "white", colour = "white", size = 0.5, linetype = "solid"), 
                                plot.background = element_rect(fill = "white"),
                                axis.text.x = element_text(angle = 60, vjust = 1, hjust=1))
p1 <- p + 
  guides(size = FALSE) + 
  labs(color  = "Method", linetype = "Method", shape = "Method") + 
  xlab('Image') + ylab("MAPE")
p1
ggsave('real-MAPE.jpg')


p <- tmp_dat %>%
  filter(measures == "RMSPE") %>%
  ggplot() + 
    aes(x = img_vals, y = vals, shape=method, group = method, linetype = method, color = method) +
    geom_point(size=3) + 
    scale_shape_manual(values = new_styles[1:4]) + 
    scale_colour_manual(values=my_cols) + 
    geom_line() +
    theme_minimal() + theme(panel.background = element_rect(fill = "white", colour = "white", size = 0.5, linetype = "solid"), 
                                plot.background = element_rect(fill = "white"),
                                axis.text.x = element_text(angle = 60, vjust = 1, hjust=1))
p2 <- p + 
  guides(size = FALSE) + 
  labs(color  = "Method", linetype = "Method", shape = "Method") + 
  xlab('\nImage') + ylab("RMSPE")
p2
ggsave('real-RMSPE.jpg')



p <- tmp_dat %>%
  filter(measures == "SSIM") %>%
  ggplot() + 
    aes(x = img_vals, y = vals, shape=method, group = method, linetype = method, color = method) +
    geom_point(size=3) + 
    scale_shape_manual(values = new_styles[1:4]) + 
    scale_colour_manual(values=my_cols) + 
    geom_line() +
    theme_minimal() + theme(panel.background = element_rect(fill = "white", colour = "white", size = 0.5, linetype = "solid"), 
                                plot.background = element_rect(fill = "white"),
                                axis.text.x = element_text(angle = 60, vjust = 1, hjust=1))
p3 <- p + 
  guides(size = FALSE) + 
  labs(color  = "Method", linetype = "Method", shape = "Method") + 
  xlab('\nImage') + ylab("SSIM")
p3
ggsave('real-SSIM.jpg')


library(patchwork)
p1_new <- p1 + theme(legend.position="none", 
              axis.title.x=element_blank(), 
              axis.text.x=element_blank()) 
p2_new <- p2 + theme(legend.position="none", 
              strip.background = element_blank(),
              strip.text.x = element_blank(),
              axis.title.x=element_blank(), 
              axis.text.x=element_blank())
p3_new <- p3 + theme(legend.position="none", 
              strip.background = element_blank(),
              strip.text.x = element_blank(), 
              axis.text.x = element_text(angle = 30, vjust = 1, hjust=1))

p1_new / p2_new / p3_new

ggsave('all-real.png', scale=1.0)




tmp <- cowplot::get_legend(p3 + 
    theme(legend.key.width = unit(1.5,"cm")) + 
    guides(
        size = guide_legend(override.aes = list(size = 2)), 
        linetype = guide_legend(override.aes = list(size = 1.1))))
ggpubr::as_ggplot(tmp)

library(grid)
grid.ls(grid.force())
grid.gedit("key-[-0-9]-1-1", size = unit(5, "mm"))
g <- grid.grab()
print(g)
legend1 <- ggpubr::as_ggplot(g)
legend1
ggsave('real-Legend.png', scale=1.0)



# all_plot <- (p1_new / p2_new / p3_new) + coord_fixed()

layout <- c(
  area(t = 0, l = 0, b = 2, r = 4),
  area(t = 3, l = 0, b = 4, r = 4),
  area(t = 5, l = 0, b = 6, r = 4),
  area(t = 2, l = 5, b = 6, r = 5)
)
p1_new + p2_new + p3_new + legend1 + 
  plot_layout(design = layout)

ggsave('all-real-Legend.pdf', scale=1.0)

system("pdfcrop all-real-Legend.pdf")
