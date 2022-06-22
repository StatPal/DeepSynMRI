
# new_styles <- -1*c(9824,9827,9829,9830)
new_styles <- 15:18


INU_vec <- c("00", "10", "20")
all_normal <- c()
all_DL     <- c()

for(i in 1:3){
    f1 <- paste0("values/pred-noise-5-INU-", INU_vec[i], ".csv")
    f2 <- paste0("values/DL-pred-noise-5-INU-", INU_vec[i], ".csv")

    all_normal <- rbind(all_normal, as.matrix(read.csv(f1, header = F))[c(6,7,3, 11,12,10), ])
    all_DL     <- rbind(all_DL    , as.matrix(read.csv(f2, header = F))[c(6,7,3, 11,12,10), ])
}

all_normal <- data.frame(all_normal)
all_DL <- data.frame(all_DL)

method <- rep(c("LS", "MLE"), each=3, times=3)  # The second will be 3
measures <- rep(c("MAPE", "RMSPE", "SSIM"), 3 * 2)
INU <- rep(INU_vec, each = 3 * 2)
DL <- rep(c(F, T), each = 3 * 3 * 2)

all_normal <- cbind(measures, INU, method, all_normal)
all_DL     <- cbind(measures, INU, method, all_DL    )

all_dat <- cbind(DL, rbind(all_normal, all_DL))
all_dat$INU <- as.numeric(all_dat$INU)

rm(DL)
rm(measures)
rm(INU)


names(all_dat)[5:13] <- paste0('Test image ', 1:9)
all_dat_old <- all_dat
# all_dat$DL <- ifelse(all_dat$DL, "DL", "No DL")


## Plots
library(tidyverse)
library(tidyr)
library(ggplot2)
library(xtable)

tmp <- all_dat %>%
  pivot_longer(!c(DL, measures, INU, method), names_to = "img", values_to = "vals") %>% 
  filter(measures == "MAPE") %>% 
  pivot_wider(names_from = img, values_from = vals) %>% 
  select(-c(measures)) %>% 
  arrange(method) %>%
  arrange(INU) %>%
  relocate(DL, .after=method) %>% 
  mutate(DL = ifelse(DL, "DL", "")) %>% 
  mutate(method = interaction(DL, method)) %>% 
  select(-DL)

print(xtable(tmp, digits=c(1,1,0,2,2,2,2,2,2,2,2,2)), include.rownames=FALSE)

tmp <- all_dat %>%
  pivot_longer(!c(DL, measures, INU, method), names_to = "img", values_to = "vals") %>% 
  filter(measures == "RMSPE") %>% 
  pivot_wider(names_from = img, values_from = vals) %>% 
  select(-c(measures)) %>% 
  arrange(method) %>%
  arrange(INU) %>%
  relocate(DL, .after=method) %>% 
  mutate(DL = ifelse(DL, "DL", "")) %>% 
  mutate(method = interaction(DL, method)) %>% 
  select(-DL)

print(xtable(tmp, digits=c(1,1,0,2,2,2,2,2,2,2,2,2)), include.rownames=FALSE)

tmp <- all_dat %>%
  pivot_longer(!c(DL, measures, INU, method), names_to = "img", values_to = "vals") %>% 
  filter(measures == "SSIM") %>% 
  pivot_wider(names_from = img, values_from = vals) %>% 
  select(-c(measures)) %>% 
  arrange(method) %>%
  arrange(INU) %>%
  relocate(DL, .after=method) %>% 
  mutate(DL = ifelse(DL, "DL", "")) %>% 
  mutate(method = interaction(DL, method)) %>% 
  select(-DL)

print(xtable(tmp, digits=c(1,1,0,2,2,2,2,2,2,2,2,2)), include.rownames=FALSE)






all_dat <- all_dat_old
names(all_dat)[5:13] <- paste0('Test image ', 1:9)




library(stringr)

names(all_dat)[5:13] <- paste0('Test image ', 1:9)
TE_TR <- c("15/0.6", "20/0.6", "10/1", "30/1", "40/1", "10/2", "40/2", "60/3", "100/3")
TE_TR_names  <- paste0('TE/TR = ', TE_TR[1:9])


tmp_dat_old <- all_dat %>%
  pivot_longer(!c(DL, measures, INU, method), names_to = "img", values_to = "vals") %>% 
  filter(img == "Test image 1" | img == "Test image 2" | img == "Test image 6" | img == "Test image 8") %>% 
  mutate(method_old = method) %>%
  mutate(method = ifelse(DL, paste0("DL+", method), method )) 

tmp_dat <- tmp_dat_old
tmp_dat$img_vals  <-  TE_TR_names[as.numeric( str_split_fixed(tmp_dat_old$img, ' ', n=3)[,3] )]



library(RColorBrewer)
my_cols <-  brewer.pal(5, 'Dark2')



p <- tmp_dat %>%
  filter(measures == "MAPE") %>%
  ggplot() + 
    aes(x = method, y = vals, group = INU, shape=method, color=factor(INU), linetype = factor(INU)) +
    geom_point(size=3.5) + 
    scale_shape_manual(values = new_styles[1:4]) + 
    scale_colour_manual(values=my_cols) + 
    facet_grid(cols = vars(img_vals)) +
    geom_line() +
    theme_minimal() + theme(panel.background = element_rect(fill = "white", colour = "white", size = 0.5, linetype = "solid"), 
                                plot.background = element_rect(fill = "white"))

p1 <- p + 
  guides(shape = guide_legend(override.aes = list(size = 5))) + 
  guides(size = FALSE) + 
  labs(color  = "INU %", linetype = "INU %", shape = "Method") + 
  xlab('Method') + ylab("MAPE")
p1
ggsave('SE-INU-MAPE.jpg')


p <- tmp_dat %>%
  filter(measures == "RMSPE") %>%
  ggplot() + 
    aes(x = method, y = vals, group = INU, shape=method, color=factor(INU), linetype = factor(INU)) +
    geom_point(size=3.5) + 
    scale_shape_manual(values = new_styles[1:4]) + 
    scale_colour_manual(values=my_cols) + 
    facet_grid(cols = vars(img_vals)) +
    geom_line() +
    theme_minimal() + theme(panel.background = element_rect(fill = "white", colour = "white", size = 0.5, linetype = "solid"), 
                                plot.background = element_rect(fill = "white"))

p2 <- p + 
  guides(shape = guide_legend(override.aes = list(size = 5))) + 
  guides(size = FALSE) + 
  labs(color  = "INU %", linetype = "INU %", shape = "Method") + 
  xlab('Method') + ylab("RMSPE")
p2
ggsave('SE-INU-FLASH-RMSPE.jpg')



p <- tmp_dat %>%
  filter(measures == "SSIM") %>%
  ggplot() + 
    aes(x = method, y = vals, group = INU, shape=method, color=factor(INU), linetype = factor(INU)) +
    geom_point(size=3.5) + 
    scale_shape_manual(values = new_styles[1:4]) + 
    scale_colour_manual(values=my_cols) + 
    facet_grid(cols = vars(img_vals)) +
    geom_line() +
    theme_minimal() + theme(panel.background = element_rect(fill = "white", colour = "white", size = 0.5, linetype = "solid"), 
                                plot.background = element_rect(fill = "white"))

p3 <- p + 
  guides(shape = guide_legend(override.aes = list(size = 5))) + 
  guides(size = FALSE) + 
  labs(color  = "INU %", linetype = "INU %", shape = "Method") + 
   xlab('\nMethod') + ylab("SSIM")
p3
ggsave('SE-INU-FLASH-SSIM.jpg')


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

ggsave('all-SE-INU.png', scale=0.75)


# p2 + theme(legend.position="bottom") + 
#   labs(color  = "INU %: ", linetype = "INU %: ", shape = "Method: ") + 
#   theme(legend.key.width = unit(2,"cm")) +
#   guides(shape = FALSE, size="none", 
#         linetype = guide_legend(override.aes = list(size = 1.5)),
#         )












leg1 <- cowplot::get_legend(p2 + theme(legend.position="bottom") + 
  labs(color  = "INU %: ", linetype = "INU %: ", shape = "Method: ") + 
  theme(legend.key.width = unit(2,"cm")) +
  guides(shape = FALSE, size="none", 
        linetype = guide_legend(override.aes = list(size = 2.0, shape=NA))))
legend1 <- ggpubr::as_ggplot(leg1) + theme(panel.background = element_rect(fill = "white", colour = "white", size = 0.5, linetype = "solid"), 
                                plot.background = element_rect(fill = "white"), 
                                legend.position="bottom")
ggsave('SE-INU-Legend1.png', scale=1.0)

leg2 <- cowplot::get_legend(p2 + theme(legend.position="bottom") + labs(shape = "Method: ") +
  guides(color = FALSE, linetype = FALSE, size="none"))
legend2 <- ggpubr::as_ggplot(leg2) + theme(panel.background = element_rect(fill = "white", colour = "white", size = 0.5, linetype = "solid"), 
                                plot.background = element_rect(fill = "white"), 
                                legend.position="bottom")
ggsave('SE-INU-Legend2.png', scale=1.0)



