
new_styles <- -1*c(9818:9824,9829,9830,9831,9832,9833)
new_styles <- -1*c(9828,9829,9830,9831,9832)
new_styles <- -1*c(9824,9827,9829,9830,9831)
plot(sort(rnorm(25)), pch=new_styles)
plot(sort(rnorm(25)), pch=c("☺", "❤", "✌", "❄", "✈"))

plot(sort(rnorm(25)), pch=c("❤", "♠", "♦", "♣"))





noise_vec <- c(1, 2.5, 5, 7.5, 10)
all_normal <- c()
all_DL     <- c()

for(i in 1:5){
    f1 <- paste0("values/pred-noise-", noise_vec[i], "-INU-00.csv")
    f2 <- paste0("values/DL-pred-noise-", noise_vec[i], "-INU-00.csv")

    all_normal <- rbind(all_normal, as.matrix(read.csv(f1, header = F)))
    all_DL     <- rbind(all_DL    , as.matrix(read.csv(f2, header = F)))
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
  filter(measures == "MAPE") %>% 
  pivot_wider(names_from = img, values_from = vals) %>% 
  select(-c(measures)) %>% 
  arrange(method) %>%
  arrange(errs) %>%
  relocate(DL, .after=method) %>% 
  mutate(DL = ifelse(DL, "DL", "")) %>% 
  mutate(method = interaction(DL, method)) %>% 
  select(-DL)

library(xtable)
print(xtable(tmp, digits=c(1,1,0,2,2,2,2,2,2,2,2,2)), include.rownames=FALSE)









# all_normal %>%
#   filter(measures == "MAPE") %>%
#   pivot_longer(!c(measures, errs, method), names_to = "img", values_to = "vals") %>% 
#   filter(img == "V1" & method == "LS") %>%
#   ggplot(aes(x = errs, y = vals)) +
#     geom_line()


# all_normal %>%
#   filter(measures == "MAPE") %>%
#   pivot_longer(!c(measures, errs, method), names_to = "img", values_to = "vals") %>% 
#   filter(method == "LS") %>%
#   ggplot() +
#     geom_line(aes(x = errs, y = vals)) + 
#     facet_wrap(~img)



# all_dat %>%
#   filter(measures == "MAPE") %>%
#   pivot_longer(!c(DL, measures, errs, method), names_to = "img", values_to = "vals") %>% 
#   filter(method == "LS") %>%
#   group_by(DL) %>%
#   ggplot(aes(x = errs, y = vals, group = DL)) +
#     geom_line(aes(linetype = DL)) + 
#     facet_wrap(~img, ncol=9) + 
#     theme_minimal() + 
#     theme(panel.spacing = unit(0,'lines'))





# all_dat %>%
#   filter(measures == "MAPE") %>%
#   pivot_longer(!c(DL, measures, errs, method), names_to = "img", values_to = "vals") %>% 
#   filter(method == "LS") %>%
#   group_by(DL) %>%
#   ggplot(aes(x = errs, y = vals, group = DL)) +
#     geom_line(aes(linetype = DL)) + 
#     facet_wrap(~img, ncol=9, nrow=3) + 
#     theme_minimal() + 
#     theme(panel.spacing = unit(0,'lines'))





# ## 9 * 3 basic image - LS only
# # all_dat %>%
# #   pivot_longer(!c(DL, measures, errs, method), names_to = "img", values_to = "vals") %>% 
# #   filter(method == "LS") %>%
# #   filter(img == "V1" | img == "V2") %>%
# #   group_by(DL) %>%
# #   ggplot(aes(x = errs, y = vals, group = DL)) +
# #     geom_line(aes(linetype = DL)) + 
# #     facet_wrap(~measures * img, ncol=2, nrow=3, scales = 'free_x') + 
# #     theme_minimal() + 
# #     theme(panel.spacing = unit(0,'lines'))



# ## facet_grid?? 
# ## 9 * 3 basic image - LS only
# all_dat %>%
#   pivot_longer(!c(DL, measures, errs, method), names_to = "img", values_to = "vals") %>% 
#   filter(method == "LS") %>% filter(img == "V1" | img == "V2") %>%
#   group_by(DL) %>%
#   ggplot(aes(x = errs, y = vals, group = DL)) +
#     geom_line(aes(linetype = DL)) + 
#     facet_grid(rows = vars(measures), cols = vars(img), scale='free_y') + 
#     theme_minimal() + 
#     theme(panel.spacing = unit(0,'lines'))



# pdf("Rplots.pdf", height=3*2, width=9*2)
# all_dat %>%
#   pivot_longer(!c(DL, measures, errs, method), names_to = "img", values_to = "vals") %>% 
#   # filter(img == "V1" | img == "V2") %>%
#   group_by(vars(DL)) %>%
#   ggplot(aes(x = errs, y = vals, group = interaction(DL, method), shape=interaction(DL, method))) +
#     geom_point(aes(color=method, size=3)) +
#     scale_shape_manual(values = new_styles[1:4]) + 
#     geom_line(aes(linetype = DL, color=method)) + 
#     facet_grid(rows = vars(measures), cols = vars(img), scale='free_y') + 
#     theme_minimal() + 
#     theme(panel.spacing = unit(0,'lines'))
# dev.off()


# all_dat %>%
#   pivot_longer(!c(DL, measures, errs, method), names_to = "img", values_to = "vals") %>% 
#   filter(img == "V1" | img == "V2" | img == "V6" | img == "V8") %>%
#   filter(measures == "MAPE") %>%
#   group_by(vars(DL)) %>%
#   ggplot(aes(x = errs, y = vals, group = interaction(DL, method), shape=interaction(DL, method))) +
#     geom_point(aes(color=factor(errs), size=3)) + scale_shape_manual(values = new_styles[1:4]) + 
#     facet_grid(cols = vars(img)) +
#     geom_line(aes(linetype = interaction(DL, method), color = interaction(DL, method))) +
#     theme_minimal() +
#     theme(panel.spacing = unit(0,'lines'))


# tmp <- all_dat %>%
#   pivot_longer(!c(DL, measures, errs, method), names_to = "img", values_to = "vals") %>% 
#   filter(img == "V1" | img == "V2")





names(all_dat)[5:13] <- paste0('Test image ', 1:9)
# all_dat$DL <- ifelse(all_dat$DL, "DL", "No DL")





tmp_dat <- all_dat %>%
  pivot_longer(!c(DL, measures, errs, method), names_to = "img", values_to = "vals") %>% 
  filter(img == "Test image 1" | img == "Test image 2" | img == "Test image 6" | img == "Test image 8") %>% 
  mutate(method_old = method) %>%
  mutate(method = ifelse(DL, paste0("DL-", method), method )) 


library(RColorBrewer)
my_cols <-  brewer.pal(5, 'Dark2')



p <- tmp_dat %>%
  filter(measures == "MAPE") %>%
  ggplot(aes(x = method, y = vals, group = errs, shape=method)) +
    geom_point(aes(color=factor(errs), size=1)) + 
    scale_shape_manual(values = new_styles[1:4]) + 
    scale_colour_manual(values=my_cols) + 
    facet_grid(cols = vars(img)) +
    geom_line(aes(linetype = factor(errs), color = factor(errs))) +
    theme_minimal() + theme(panel.background = element_rect(fill = "white", colour = "white", size = 0.5, linetype = "solid"), 
                                plot.background = element_rect(fill = "white"))

p1 <- p + labs( colour = "Noise percentage", shape = "Method") + 
  guides(linetype = FALSE, size="none") + 
  guides(shape = guide_legend(override.aes = list(size = 5))) + xlab('Method') + ylab("MAPE")
p1
ggsave('MAPE.jpg')


p <- tmp_dat %>%
  filter(measures == "RMSPE") %>%
  ggplot(aes(x = method, y = vals, group = errs, shape=method)) +
    geom_point(aes(color=factor(errs), size=1)) + 
    scale_shape_manual(values = new_styles[1:4]) + 
    scale_colour_manual(values=my_cols) + 
    facet_grid(cols = vars(img)) +
    geom_line(aes(linetype = factor(errs), color = factor(errs))) +
    theme_minimal() + theme(panel.background = element_rect(fill = "white", colour = "white", size = 0.5, linetype = "solid"), 
                                plot.background = element_rect(fill = "white"))

p2 <- p + labs( colour = "Noise percentage", shape = "Method") + 
  guides(linetype = FALSE, size="none") + 
  guides(shape = guide_legend(override.aes = list(size = 5))) + xlab('Method') + ylab("RMSPE")
p2
ggsave('RMSPE.jpg')



p <- tmp_dat %>%
  filter(measures == "SSIM") %>%
  ggplot(aes(x = method, y = vals, group = errs, shape=method)) +
    geom_point(aes(color=factor(errs), size=1)) + 
    scale_shape_manual(values = new_styles[1:4]) + 
    scale_colour_manual(values=my_cols) + 
    facet_grid(cols = vars(img)) +
    geom_line(aes(linetype = factor(errs), color = factor(errs))) +
    theme_minimal() + theme(panel.background = element_rect(fill = "white", colour = "white", size = 0.5, linetype = "solid"), 
                                plot.background = element_rect(fill = "white"))

p3 <- p + labs(colour = "Noise percentage", shape = "Method") + 
  guides(linetype = FALSE, size="none") + 
  guides(shape = guide_legend(override.aes = list(size = 5))) + xlab('Method') + ylab("SSIM")
p3
ggsave('SSIM.jpg')


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
              axis.text.x = element_text(angle = 45, vjust = 0.5, hjust=1))

p1_new / p2_new / p3_new

ggsave('all-SE.png', scale=0.75)








p <- tmp_dat %>%
  ggplot(aes(x = interaction(method, DL), y = vals, group = errs, shape=interaction(method, DL))) +
    geom_point(aes(color=factor(errs), size=3)) + 
    scale_shape_manual(values = new_styles[1:4]) + 
    scale_colour_manual(values=my_cols) + 
    facet_grid(rows = vars(measures), cols = vars(img), scale='free_y') + 
    geom_line(aes(linetype = factor(errs), color = factor(errs))) +
    theme_minimal() + theme(panel.background = element_rect(fill = "white", colour = "white", size = 0.5, linetype = "solid"), 
                                plot.background = element_rect(fill = "white"),
                                strip.text.y = element_text(angle = 0)) + 
  labs(colour = "Noise percentage", shape = "Method") + 
  guides(linetype = FALSE, size="none") + 
  guides(shape = guide_legend(override.aes = list(size = 5))) + 
  xlab('Method') + ylab("SSIM")







tmp_dat %>%
  ggplot(aes(x = interaction(method, DL), y = vals, group = errs, shape=interaction(method, DL))) +
    geom_point(aes(color=factor(errs), size=3)) + 
    scale_shape_manual(values = new_styles[1:4]) + 
    scale_colour_manual(values=my_cols) + 
    facet_wrap(~measures * img, strip.position = c("left")) + 
    geom_line(aes(linetype = factor(errs), color = factor(errs))) +
    theme_minimal() + theme(panel.background = element_rect(fill = "white", colour = "white", size = 0.5, linetype = "solid"), 
                                plot.background = element_rect(fill = "white"),
                                strip.text.y = element_text(angle = 0)) + 
                                ylab(NULL) +
     theme(strip.background = element_blank(),
           strip.placement = "outside")
