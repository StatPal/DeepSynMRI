
# new_styles <- -1*c(9818:9824,9829,9830,9831,9832,9833)
# new_styles <- -1*c(9828,9829,9830,9831,9832)
# new_styles <- -1*c(9824,9827,9829,9830)
# plot(sort(rnorm(25)), pch=new_styles)
# plot(sort(rnorm(25)), pch=c("☺", "❤", "✌", "❄", "✈"))

# plot(sort(rnorm(25)), pch=c("❤", "♠", "♦", "♣"))

# new_styles <- -1*c(9824,9827,9829,9830)
# # plot(, type="b", pch=new_styles)


new_styles <- 15:18



noise_vec <- c(1, 2.5, 5, 7.5, 10)
all_normal <- c()
all_DL     <- c()

for(i in 1:5){
    f1 <- paste0("values/pred-noise-", noise_vec[i], "-INU-00.csv")
    f2 <- paste0("values/DL-pred-noise-", noise_vec[i], "-INU-00.csv")

    all_normal <- rbind(all_normal, as.matrix(read.csv(f1, header = F))[c(6,7,3, 13,14,10), ])   ## Only the scaled versions
    all_DL     <- rbind(all_DL    , as.matrix(read.csv(f2, header = F))[c(6,7,3, 13,14,10), ])
}

all_normal <- data.frame(all_normal)
all_DL <- data.frame(all_DL)

method <- rep(c("LS", "MLE"), each=3, times=5)
measures <- rep(c("MAPE", "RMSPE", "SSIM"), 5 * 2)
errs <- rep(c(1, 2.5, 5, 7.5, 10), each = 3 * 2)
DL <- rep(c(F, T), each = 5 * 3 * 2)

all_normal <- cbind(measures, errs, method, all_normal)
all_DL     <- cbind(measures, errs, method, all_DL    )

all_dat <- cbind(DL, rbind(all_normal, all_DL))

rm(DL)
rm(measures)
rm(errs)



## Plots
library(tidyverse)
library(tidyr)
library(ggplot2)








# ##

# tmp <- all_dat %>%
#   pivot_longer(!c(DL, measures, errs, method), names_to = "img", values_to = "vals") %>% 
#   filter(measures == "MAPE") %>% 
#   filter(method == "LS") %>% 
#   pivot_wider(names_from = img, values_from = vals) %>% 
#   arrange(errs) %>%
#   select(-c(measures, method)) 

# xtable::xtable(tmp)





tmp <- all_dat %>%
  pivot_longer(!c(DL, measures, errs, method), names_to = "img", values_to = "vals") %>% 
  filter(measures == "RMSPE") %>% 
  pivot_wider(names_from = img, values_from = vals) %>% 
  select(-c(measures)) %>% 
  arrange(method) %>%
  arrange(errs) %>%
  relocate(DL, .after=method) %>% 
  mutate(DL = ifelse(DL, "DL", "")) %>% 
  mutate(method = interaction(DL, method)) %>% 
  # mutate_if(is.numeric, ~ . * 100) %>%
  # select(starts_with("V"), ~ . * 100) %>%
  select(-DL)

library(xtable)
print(xtable(tmp, digits=c(1,1,0,2,2,2,2,2,2,2,2,2)), include.rownames=FALSE)




library(stringr)

names(all_dat)[5:13] <- paste0('Test image ', 1:9)
# TE_TR <- c("15/0.6", "20/0.6", "10/1", "30/1", "40/1", "10/2", "40/2", "60/3", "100/3")
# TE_TR <- c(".015/.6", ".020/.6", ".01/1", ".03/1", ".04/1", ".01/2", ".04/2", ".06/3", ".1/3")  # sec
TE_TR <- c("0.015/0.6", "0.020/0.6", "0.01/1", "0.03/1", "0.04/1", "0.01/2", "0.04/2", "0.06/3", "0.1/3")  # sec
# TE_TR <- c("15/600", "20/600", "10/1000", "30/1000", "40/1000", "10/2000", "40/2000", "60/3000", "100/3000")
TE_TR_names  <- paste0('TE/TR = ', TE_TR[1:9])


tmp_dat_old <- all_dat %>%
  pivot_longer(!c(DL, measures, errs, method), names_to = "img", values_to = "vals") %>% 
  filter(img == "Test image 1" | img == "Test image 2" | img == "Test image 6" | img == "Test image 8") %>% 
  mutate(method_old = method) %>%
  mutate(method = ifelse(DL, paste0("DL+", method), method )) 

tmp_dat <- tmp_dat_old
tmp_dat$img_vals  <-  TE_TR_names[as.numeric( str_split_fixed(tmp_dat_old$img, ' ', n=3)[,3] )]


# ## Checking order
# all_dat %>%
#   filter(measures == "MAPE") %>%
#   pivot_longer(!c(errs, measures, method, DL), names_to = "img", values_to = "vals") %>% 
#   filter(img == "Test image 1" | img == "Test image 2") %>% 
#   filter(errs == 1 | errs == 5) %>% 
#   mutate(method_old = method) %>%
#   mutate(method = ifelse(DL, paste0("DL-", method), method ))








library(RColorBrewer)
my_cols <-  brewer.pal(5, 'Dark2')


# as.matrix(tmp_dat %>%
#   filter(measures == "MAPE"))

# as.matrix(tmp_dat %>%
#   filter(errs == 1 | errs == 5) %>% 
#   filter(measures == "MAPE") %>%
#   arrange(rev(method)) %>%
#   arrange(errs))


tmp_dat %>%
  # filter(errs == 1 | errs == 5) %>% 
  filter(measures == "MAPE") %>%
  mutate(img = img) %>% 
  ggplot() + 
    aes(x = interaction(DL, method_old), y = vals, group = errs, shape=method, color=factor(errs), linetype = factor(errs)) +
    geom_point(size=3) + 
    scale_shape_manual(values = new_styles[1:4]) + 
    scale_colour_manual(values=my_cols) + 
    facet_grid(cols = vars(img)) +
    geom_line() +
    theme_minimal() + theme(panel.background = element_rect(fill = "white", colour = "white", size = 0.5, linetype = "solid"), 
                                plot.background = element_rect(fill = "white")) + 
  labs(colour = "Noise percentage", shape = "Method") + 
  # guides(linetype = FALSE, size="none") + 
  guides(shape = guide_legend(override.aes = list(size = 5))) + 
  guides(size = FALSE) + 
  xlab('Method') + ylab("MAPE") + 
  labs(color  = "Guide name", linetype = "Guide name", shape = "Guide name")







p <- tmp_dat %>%
  filter(measures == "MAPE") %>%
  ggplot() + 
    aes(x = method, y = vals, group = errs, shape=method, color=factor(errs), linetype = factor(errs)) +
    geom_point(size=3) + 
    scale_shape_manual(values = new_styles[1:4]) + 
    scale_colour_manual(values=my_cols) + 
    facet_grid(cols = vars(img_vals)) +
    geom_line() +
    theme_minimal() + theme(panel.background = element_rect(fill = "white", colour = "white", size = 0.5, linetype = "solid"), 
                                plot.background = element_rect(fill = "white"))

p1 <- p + 
  guides(shape = guide_legend(override.aes = list(size = 5))) + 
  guides(size = FALSE) + 
  labs(color  = "Noise percentage", linetype = "Noise percentage", shape = "Method") + 
  xlab('Method') + ylab("MAPE")
p1
ggsave('MAPE.png')


p <- tmp_dat %>%
  filter(measures == "RMSPE") %>%
    ggplot() + 
    aes(x = method, y = vals, group = errs, shape=method, color=factor(errs), linetype = factor(errs)) +
    geom_point(size=3) + 
    scale_shape_manual(values = new_styles[1:4]) + 
    scale_colour_manual(values=my_cols) + 
    facet_grid(cols = vars(img_vals)) +
    geom_line() +
    theme_minimal() + theme(panel.background = element_rect(fill = "white", colour = "white", size = 0.5, linetype = "solid"), 
                                plot.background = element_rect(fill = "white"))

p2 <- p + 
  guides(shape = guide_legend(override.aes = list(size = 5))) + 
  guides(size = FALSE) + 
  labs(color  = "Noise percentage", linetype = "Noise percentage", shape = "Method") + 
  xlab('Method') + ylab("RMSPE")
p2
ggsave('RMSPE.png')



p <- tmp_dat %>%
  filter(measures == "SSIM") %>%
    ggplot() + 
    aes(x = method, y = vals, group = errs, shape=method, color=factor(errs), linetype = factor(errs)) +
    geom_point(size=3) + 
    scale_shape_manual(values = new_styles[1:4]) + 
    scale_colour_manual(values=my_cols) + 
    facet_grid(cols = vars(img_vals)) +
    geom_line() +
    theme_minimal() + theme(panel.background = element_rect(fill = "white", colour = "white", size = 0.5, linetype = "solid"), 
                                plot.background = element_rect(fill = "white"))

p3 <- p + 
  guides(shape = guide_legend(override.aes = list(size = 5))) + 
  guides(size = FALSE) + 
  labs(color  = "Noise percentage", linetype = "Noise percentage", shape = "Method") + 
  xlab('\nMethod') + ylab("SSIM") 
p3


p3 + theme(legend.position="none", 
              strip.background = element_blank(),
              strip.text.x = element_blank(), 
              axis.text.x = element_text(angle = 45, vjust = 0.5, hjust=1),
              axis.title.y = element_text(margin = margin(t = 0, r = 0, b = 20, l = 0)))

ggsave('SSIM.png', scale=0.8, width=6, height=4)


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
              axis.text.x = element_text(angle = 45, vjust = 0.5, hjust=1),
              axis.title.y = element_text(margin = margin(t = 0, r = 0, b = 20, l = 0)))

p3_new

p1_new / p2_new / p3_new

ggsave('all-SE.png', scale=0.75)



leg1 <- cowplot::get_legend(p2 + theme(legend.position="bottom") + 
  labs(color  = "Noise %: ", linetype = "Noise %: ", shape = "Method: ") + 
  theme(legend.key.width = unit(2,"cm")) +
  guides(shape = FALSE, size="none",
          linetype = guide_legend(override.aes = list(size = 2.0, shape=NA))))
## https://stackoverflow.com/questions/48361948/remove-box-and-points-in-legend

legend1 <- ggpubr::as_ggplot(leg1) + theme(panel.background = element_rect(fill = "white", colour = "white", size = 0.5, linetype = "solid"), 
                                plot.background = element_rect(fill = "white"), 
                                legend.position="bottom")
ggsave('Legend1.png', scale=1.0)

leg2 <- cowplot::get_legend(p2 + theme(legend.position="bottom") + labs(shape = "Method: ") +
  guides(color = FALSE, linetype = FALSE, size="none"))
legend2 <- ggpubr::as_ggplot(leg2) + theme(panel.background = element_rect(fill = "white", colour = "white", size = 0.5, linetype = "solid"), 
                                plot.background = element_rect(fill = "white"), 
                                legend.position="bottom")
ggsave('Legend2.png', scale=1.0)







p3_new <- p3 + theme(legend.position="none", 
              strip.background = element_blank(),
              strip.text.x = element_blank(), 
              axis.text.x = element_text(angle = 45, vjust = 0.5, hjust=1),
              axis.title.y = element_text(margin = margin(t = 0, r = 0, b = 20, l = 0)))

library(patchwork)

layout <- c(
area(1, 1, 3, 4),
area(3, 1, 3, 4),
area(4, 1, 5, 4)
)

p3_new / legend1 / legend2 +  plot_layout(design = layout)

ggsave('SSIM-new.png', scale=0.75)










# p <- tmp_dat %>%
#   ggplot(aes(x = interaction(method, DL), y = vals, group = errs, shape=interaction(method, DL))) +
#     geom_point(aes(color=factor(errs), size=3)) + 
#     scale_shape_manual(values = new_styles[1:4]) + 
#     scale_colour_manual(values=my_cols) + 
#     facet_grid(rows = vars(measures), cols = vars(img), scale='free_y') + 
#     geom_line(aes(linetype = factor(errs), color = factor(errs))) +
#     theme_minimal() + theme(panel.background = element_rect(fill = "white", colour = "white", size = 0.5, linetype = "solid"), 
#                                 plot.background = element_rect(fill = "white"),
#                                 strip.text.y = element_text(angle = 0)) + 
#   labs(colour = "Noise percentage", shape = "Method") + 
#   guides(linetype = FALSE, size="none") + 
#   guides(shape = guide_legend(override.aes = list(size = 5))) + 
#   xlab('Method') + ylab("SSIM")







# tmp_dat %>%
#   ggplot(aes(x = interaction(method, DL), y = vals, group = errs, shape=interaction(method, DL))) +
#     geom_point(aes(color=factor(errs), size=3)) + 
#     scale_shape_manual(values = new_styles[1:4]) + 
#     scale_colour_manual(values=my_cols) + 
#     facet_wrap(~measures * img, strip.position = c("left")) + 
#     geom_line(aes(linetype = factor(errs), color = factor(errs))) +
#     theme_minimal() + theme(panel.background = element_rect(fill = "white", colour = "white", size = 0.5, linetype = "solid"), 
#                                 plot.background = element_rect(fill = "white"),
#                                 strip.text.y = element_text(angle = 0)) + 
#                                 ylab(NULL) +
#      theme(strip.background = element_blank(),
#            strip.placement = "outside")



sys.call("convert Legend1.png -trim Legend1.trim.png")
sys.call("convert Legend2.png -trim Legend2.trim.png")
