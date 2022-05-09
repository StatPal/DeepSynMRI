
new_styles <- -1*c(9824,9827,9829,9830)

INU_vec <- c("00", "10", "20")
all_normal <- c()
all_DL     <- c()

for(i in 1:3){
    f1 <- paste0("values/FLASH-pred-noise-5-INU-", INU_vec[i], ".csv")
    f2 <- paste0("values/FLASH-DL-pred-noise-5-INU-", INU_vec[i], ".csv")

    all_normal <- rbind(all_normal, as.matrix(read.csv(f1, header = F)))
    all_DL     <- rbind(all_DL    , as.matrix(read.csv(f2, header = F)))
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

rm(DL)
rm(measures)
rm(INU)


names(all_dat)[5:13] <- paste0('Test image ', 1:9)
all_dat_old <- all_dat
all_dat$DL <- ifelse(all_dat$DL, "DL", "No DL")


## Plots
library(tidyverse)
library(tidyr)
library(ggplot2)

tmp <- all_dat %>%
  pivot_longer(!c(DL, measures, INU, method), names_to = "img", values_to = "vals") %>% 
  filter(measures == "SSIM") %>% 
  pivot_wider(names_from = img, values_from = vals) %>% 
  arrange(method) %>%
  arrange(INU) %>%
  select(-c(measures)) %>% 
  relocate(DL, .after=method)

library(xtable)
print(xtable(tmp, digits=c(1,1,0,0,2,2,2,2,2,2,2,2,2)), include.rownames=FALSE)








all_dat <- all_dat_old
names(all_dat)[5:13] <- paste0('Test image ', 1:9)





tmp_dat <- all_dat %>%
  pivot_longer(!c(DL, measures, INU, method), names_to = "img", values_to = "vals") %>% 
  filter(img == "Test image 1" | img == "Test image 2" | img == "Test image 6" | img == "Test image 8") %>% 
  mutate(method_old = method) %>%
  mutate(method = ifelse(DL, paste0("DL-", method), method )) 


library(RColorBrewer)
my_cols <-  brewer.pal(5, 'Dark2')



p <- tmp_dat %>%
  filter(measures == "MAPE") %>%
  ggplot(aes(x = method, y = vals, group = INU, shape=method)) +
    geom_point(aes(color=factor(INU), size=1)) + 
    scale_shape_manual(values = new_styles[1:4]) + 
    scale_colour_manual(values=my_cols) + 
    facet_grid(cols = vars(img)) +
    geom_line(aes(linetype = factor(INU), color = factor(INU))) +
    theme_minimal() + theme(panel.background = element_rect(fill = "white", colour = "white", size = 0.5, linetype = "solid"), 
                                plot.background = element_rect(fill = "white"))

p1 <- p + labs( colour = "Noise percentage", shape = "Method") + 
  guides(linetype = FALSE, size="none") + 
  guides(shape = guide_legend(override.aes = list(size = 5))) + xlab('Method') + ylab("MAPE")
p1
ggsave('MAPE.jpg')


p <- tmp_dat %>%
  filter(measures == "RMSPE") %>%
  ggplot(aes(x = method, y = vals, group = INU, shape=method)) +
    geom_point(aes(color=factor(INU), size=1)) + 
    scale_shape_manual(values = new_styles[1:4]) + 
    scale_colour_manual(values=my_cols) + 
    facet_grid(cols = vars(img)) +
    geom_line(aes(linetype = factor(INU), color = factor(INU))) +
    theme_minimal() + theme(panel.background = element_rect(fill = "white", colour = "white", size = 0.5, linetype = "solid"), 
                                plot.background = element_rect(fill = "white"))

p2 <- p + labs( colour = "Noise percentage", shape = "Method") + 
  guides(linetype = FALSE, size="none") + 
  guides(shape = guide_legend(override.aes = list(size = 5))) + xlab('Method') + ylab("RMSPE")
p2
ggsave('RMSPE.jpg')



p <- tmp_dat %>%
  filter(measures == "SSIM") %>%
  ggplot(aes(x = method, y = vals, group = INU, shape=method)) +
    geom_point(aes(color=factor(INU), size=1)) + 
    scale_shape_manual(values = new_styles[1:4]) + 
    scale_colour_manual(values=my_cols) + 
    facet_grid(cols = vars(img)) +
    geom_line(aes(linetype = factor(INU), color = factor(INU))) +
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

ggsave('all-FLASH.png', scale=0.75)

