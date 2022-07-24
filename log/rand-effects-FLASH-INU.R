## FLASH

INU_vec <- c("00", "10", "20")
all_normal <- c()
all_DL     <- c()

for(i in 1:3){
    f1 <- paste0("values/FLASH-pred-noise-5-INU-", INU_vec[i], ".csv")
    f2 <- paste0("values/FLASH-DL-pred-noise-5-INU-", INU_vec[i], ".csv")

    all_normal <- rbind(all_normal, as.matrix(read.csv(f1, header = F))[c(6,7,3, 13,14,10), ])
    all_DL     <- rbind(all_DL    , as.matrix(read.csv(f2, header = F))[c(6,7,3, 13,14,10), ])
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









library(tidyverse)
library(lme4)


tmp_dat <- all_dat %>%
    filter(measures == "SSIM") %>%
    filter(method == "LS") %>%
    pivot_longer(!c(DL, measures, INU, method), names_to = "img", values_to = "vals") 


tmp_dat$INUf <- factor(tmp_dat$INU)
(mod_test <- lmer(vals ~ DL * INUf + (1 | img), tmp_dat))   # random effect for intercept

library(emmeans)
emmeans_1 <- emmeans(mod_test, ~ DL | INUf)
pairs(emmeans_1)
xtable::xtable(pairs(emmeans_1))

library(ggResidpanel)
resid_panel(mod_test)


library(qqplotr)
type = "response"
model = mod_test
model_values <- data.frame(Residual = ggResidpanel:::helper_resid(type = "response", model = mod_test))
r_label <- ggResidpanel:::helper_label(type, model)
data_add <- ggResidpanel:::helper_plotly_label(model)
model_values <- cbind(model_values, data_add)
names(model_values)[which(names(model_values) == "data_add")] <- "Data"
model_values <- model_values[order(model_values$Residual), ]
plot <- ggplot(data = model_values, mapping = aes_string(sample = "Residual", 
    label = "Data")) + stat_qq_point() + labs(x = "Theoretical Quantiles", 
    y = "Sample Quantiles")
plot_data <- ggplot_build(plot)
model_values$Theoretical <- plot_data[[1]][[1]]$theoretical
model_values$Residual_Plot <- model_values$Residual


p_SSIM <- ggplot(data = model_values, mapping = aes_string(sample = "Residual_Plot", 
    label = "Data")) + stat_qq_point() + 
    labs(x = "Theoretical Quantiles", y = "Sample Quantiles") + 
    stat_qq_line(color = "blue", size = 0.5) + 
    theme_minimal() + theme(panel.background = element_rect(fill = "white", colour = "white", size = 0.5, linetype = "solid"), 
                                plot.background = element_rect(fill = "white"),
                                axis.text = element_text(size = 15),
                                axis.title = element_text(size = 20))

p_SSIM_band <- ggplot(data = model_values, mapping = aes_string(sample = "Residual_Plot", 
    label = "Data")) + stat_qq_band() + stat_qq_point() + 
    labs(x = "Theoretical Quantiles", y = "Sample Quantiles") + 
    stat_qq_line(color = "blue", size = 0.5) + 
    theme_minimal() + theme(panel.background = element_rect(fill = "white", colour = "white", size = 0.5, linetype = "solid"), 
                                plot.background = element_rect(fill = "white"),
                                axis.text = element_text(size = 15),
                                axis.title = element_text(size = 20))

p_SSIM
p_SSIM_band









tmp_dat <- all_dat %>%
    filter(measures == "MAPE") %>%
    filter(method == "LS") %>%
    pivot_longer(!c(DL, measures, INU, method), names_to = "img", values_to = "vals") 


tmp_dat$INUf <- factor(tmp_dat$INU)
(mod_test <- lmer(vals ~ DL * INUf + (1 | img), tmp_dat))   # random effect for intercept

library(emmeans)
emmeans_1 <- emmeans(mod_test, ~ DL | INUf)
pairs(emmeans_1)
xtable::xtable(pairs(emmeans_1))

library(ggResidpanel)
resid_panel(mod_test)


library(qqplotr)
type = "response"
model = mod_test
model_values <- data.frame(Residual = ggResidpanel:::helper_resid(type = "response", model = mod_test))
r_label <- ggResidpanel:::helper_label(type, model)
data_add <- ggResidpanel:::helper_plotly_label(model)
model_values <- cbind(model_values, data_add)
names(model_values)[which(names(model_values) == "data_add")] <- "Data"
model_values <- model_values[order(model_values$Residual), ]
plot <- ggplot(data = model_values, mapping = aes_string(sample = "Residual", 
    label = "Data")) + stat_qq_point() + labs(x = "Theoretical Quantiles", 
    y = "Sample Quantiles")
plot_data <- ggplot_build(plot)
model_values$Theoretical <- plot_data[[1]][[1]]$theoretical
model_values$Residual_Plot <- model_values$Residual


p_MAPE <- ggplot(data = model_values, mapping = aes_string(sample = "Residual_Plot", 
    label = "Data")) + stat_qq_point() + 
    labs(x = "Theoretical Quantiles", y = "Sample Quantiles") + 
    stat_qq_line(color = "blue", size = 0.5) + 
    theme_minimal() + theme(panel.background = element_rect(fill = "white", colour = "white", size = 0.5, linetype = "solid"), 
                                plot.background = element_rect(fill = "white"),
                                axis.text = element_text(size = 15),
                                axis.title = element_text(size = 20))

p_MAPE_band <- ggplot(data = model_values, mapping = aes_string(sample = "Residual_Plot", 
    label = "Data")) + stat_qq_band() + stat_qq_point() + 
    labs(x = "Theoretical Quantiles", y = "Sample Quantiles") + 
    stat_qq_line(color = "blue", size = 0.5) + 
    theme_minimal() + theme(panel.background = element_rect(fill = "white", colour = "white", size = 0.5, linetype = "solid"), 
                                plot.background = element_rect(fill = "white"),
                                axis.text = element_text(size = 15),
                                axis.title = element_text(size = 20))

p_MAPE
p_MAPE_band





tmp_dat <- all_dat %>%
    filter(measures == "RMSPE") %>%
    filter(method == "LS") %>%
    pivot_longer(!c(DL, measures, INU, method), names_to = "img", values_to = "vals") 


tmp_dat$INUf <- factor(tmp_dat$INU)
(mod_test <- lmer(vals ~ DL * INUf + (1 | img), tmp_dat))   # random effect for intercept

library(emmeans)
emmeans_1 <- emmeans(mod_test, ~ DL | INUf)
pairs(emmeans_1)
xtable::xtable(pairs(emmeans_1))

library(ggResidpanel)
resid_panel(mod_test)


library(qqplotr)
type = "response"
model = mod_test
model_values <- data.frame(Residual = ggResidpanel:::helper_resid(type = "response", model = mod_test))
r_label <- ggResidpanel:::helper_label(type, model)
data_add <- ggResidpanel:::helper_plotly_label(model)
model_values <- cbind(model_values, data_add)
names(model_values)[which(names(model_values) == "data_add")] <- "Data"
model_values <- model_values[order(model_values$Residual), ]
plot <- ggplot(data = model_values, mapping = aes_string(sample = "Residual", 
    label = "Data")) + stat_qq_point() + labs(x = "Theoretical Quantiles", 
    y = "Sample Quantiles")
plot_data <- ggplot_build(plot)
model_values$Theoretical <- plot_data[[1]][[1]]$theoretical
model_values$Residual_Plot <- model_values$Residual


p_RMSPE <- ggplot(data = model_values, mapping = aes_string(sample = "Residual_Plot", 
    label = "Data")) + stat_qq_point() + 
    labs(x = "Theoretical Quantiles", y = "Sample Quantiles") + 
    stat_qq_line(color = "blue", size = 0.5) + 
    theme_minimal() + theme(panel.background = element_rect(fill = "white", colour = "white", size = 0.5, linetype = "solid"), 
                                plot.background = element_rect(fill = "white"),
                                axis.text = element_text(size = 15),
                                axis.title = element_text(size = 20))

p_RMSPE_band <- ggplot(data = model_values, mapping = aes_string(sample = "Residual_Plot", 
    label = "Data")) + stat_qq_band() + stat_qq_point() + 
    labs(x = "Theoretical Quantiles", y = "Sample Quantiles") + 
    stat_qq_line(color = "blue", size = 0.5) + 
    theme_minimal() + theme(panel.background = element_rect(fill = "white", colour = "white", size = 0.5, linetype = "solid"), 
                                plot.background = element_rect(fill = "white"),
                                axis.text = element_text(size = 15),
                                axis.title = element_text(size = 20))

p_RMSPE
p_RMSPE_band





pdf('qqplots-FLASH-INU.pdf')
p_MAPE_band
p_RMSPE_band
p_SSIM_band

p_MAPE
p_RMSPE
p_SSIM

dev.off()


system("pdfcrop qqplots-FLASH-INU.pdf")





