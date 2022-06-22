noise_vec <- c(1, 2.5, 5, 7.5, 10)
all_normal <- c()
all_DL     <- c()

for(i in 1:5){
    f1 <- paste0("values/pred-noise-", noise_vec[i], "-INU-00.csv")
    f2 <- paste0("values/DL-pred-noise-", noise_vec[i], "-INU-00.csv")

    all_normal <- rbind(all_normal, as.matrix(read.csv(f1, header = F))[c(6,7,3, 11,12,10), ])  # to take only the normalized? versions c(4,5,3, 9,10,8)
    all_DL     <- rbind(all_DL    , as.matrix(read.csv(f2, header = F))[c(6,7,3, 11,12,10), ])
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


library(tidyverse)
tmp_dat <- all_dat %>%
    filter(measures == "MAPE") %>%
    filter(method == "LS") %>%
    pivot_longer(!c(DL, measures, errs, method), names_to = "img", values_to = "vals") 


library(lme4)
mod1 <- lmer(vals ~ DL + (DL | img), tmp_dat)
mod1


## Random effects model to the real data
ggplot(tmp_dat) + 
    aes(x = DL, y = vals, group = errs, shape=method, color=factor(errs), linetype = factor(errs)) +
    geom_point(aes(size=1)) + 
    facet_grid(cols = vars(img)) +
    geom_line()


tmp_dat$errf <- factor(tmp_dat$errs)

(mod_test <- lmer(vals ~ DL * errf + (1 | img), tmp_dat))   # random effect for intercept   # make new column, errf

library(emmeans)
emmeans_1 <- emmeans(mod_test, ~ DL | errf)
pairs(emmeans_1)
xtable::xtable(pairs(emmeans_1))

library(ggResidpanel)
resid_panel(mod_test)
resid_panel(mod_test, type='response')
# resid_panel(mod_test, type='standardized')

p_MAPE <- resid_panel(mod_test, type='response', plots = "qq") + 
    theme_minimal() + theme(panel.background = element_rect(fill = "white", colour = "white", size = 0.5, linetype = "solid"), 
                                plot.background = element_rect(fill = "white"),
                                axis.text=element_text(size=40),
        axis.title=element_text(size=20,face="bold"))

p_MAPE


library(qqplotr)
qq <- ggResidpanel:::plot_qq(model = mod_test, type = "response", theme = "bw", 
            axis.text.size = 10, title.text.size = 20, 
            title.opt = 'Q-Q Plot', qqline = T, qqbands = T)



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


# plot <- ggplot(data = model_values, mapping = aes_string(sample = "Residual_Plot", 
#     label = "Data")) + stat_qq_point() + geom_point(mapping = aes_string(x = "Theoretical", 
#     y = "Residual")) + labs(x = "Theoretical Quantiles", 
#     y = "Sample Quantiles")

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
                                # plot.title = element_text(size = 10, face = "bold"), 
                                axis.text = element_text(size = 15),
                                axis.title = element_text(size = 20))

p_MAPE
p_MAPE_band



(design_X <- getME(mod_test, "X"))
design_X %*% t(design_X)

diag( design_X %*% solve(t(design_X) %*% design_X) %*% t(design_X) )

summary(mod_test)
coef(mod_test)$img





(mod_test <- lmer(vals ~ DL * errf + (errf | img), tmp_dat))
emmeans_1 <- emmeans(mod_test, ~ DL | errf)
pairs(emmeans_1)

summary(mod_test)
coef(mod_test)$img
# Makes more sense without singularity????



(mod_test <- lmer(vals ~ DL * errs + (DL | img), tmp_dat))
# Singular

summary(mod_test)
coef(mod_test)$img





(mod_test <- lmer(vals ~ DL * errs + (DL * errs | img), tmp_dat))

summary(mod_test)
coef(mod_test)$img
# Singular














tmp_dat <- all_dat %>%
    filter(measures == "RMSPE") %>%
    filter(method == "LS") %>%
    pivot_longer(!c(DL, measures, errs, method), names_to = "img", values_to = "vals") 


tmp_dat$errf <- factor(tmp_dat$errs)
(mod_test <- lmer(vals ~ DL * errf + (1 | img), tmp_dat))   # random effect for intercept

emmeans_1 <- emmeans(mod_test, ~ DL | errf)
pairs(emmeans_1)
xtable::xtable(pairs(emmeans_1))

library(ggResidpanel)
resid_panel(mod_test)

# p_RMSPE <- resid_panel(mod_test, type='response', plots = "qq") + 
#     theme_minimal() + theme(panel.background = element_rect(fill = "white", colour = "white", size = 0.5, linetype = "solid"), 
#                                 plot.background = element_rect(fill = "white"))

# p_RMSPE


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








summary(mod_test)
coef(mod_test)$img


(mod_test <- lmer(vals ~ DL * errs + (DL | img), tmp_dat))
# Singular
summary(mod_test)
coef(mod_test)$img


(mod_test <- lmer(vals ~ DL * errs + (errs | img), tmp_dat))
summary(mod_test)
coef(mod_test)$img
# Makes more sense without singularity????


(mod_test <- lmer(vals ~ DL * errs + (DL * errs | img), tmp_dat))

summary(mod_test)
coef(mod_test)$img
# Singular






tmp_dat <- all_dat %>%
    filter(measures == "SSIM") %>%
    filter(method == "LS") %>%
    pivot_longer(!c(DL, measures, errs, method), names_to = "img", values_to = "vals") 


tmp_dat$errf <- factor(tmp_dat$errs)
(mod_test <- lmer(vals ~ DL * errf + (1 | img), tmp_dat))   # random effect for intercept

emmeans_1 <- emmeans(mod_test, ~ DL | errf)
pairs(emmeans_1)

library(ggResidpanel)
resid_panel(mod_test)

# p_SSIM <- resid_panel(mod_test, type='response', plots = "qq") + 
#     theme_minimal() + theme(panel.background = element_rect(fill = "white", colour = "white", size = 0.5, linetype = "solid"), 
#                                 plot.background = element_rect(fill = "white"))

# p_SSIM


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




summary(mod_test)
coef(mod_test)$img


(mod_test <- lmer(vals ~ DL * errs + (DL | img), tmp_dat))
# Singular
summary(mod_test)
coef(mod_test)$img


(mod_test <- lmer(vals ~ DL * errs + (errs | img), tmp_dat))
summary(mod_test)
coef(mod_test)$img
# Makes more sense without singularity????


(mod_test <- lmer(vals ~ DL * errs + (DL * errs | img), tmp_dat))

summary(mod_test)
coef(mod_test)$img
# Singular




pdf('qqplots.pdf')
p_MAPE_band
p_RMSPE_band
p_SSIM_band

p_MAPE
p_RMSPE
p_SSIM

# p_MAPE + ggtitle('Q-Q Plot for residuals of radom effects model fitted to MAPE measures')
# p_RMSPE + ggtitle('Q-Q Plot for residuals of radom effects model fitted to RMSPE measures')
# p_SSIM + ggtitle('Q-Q Plot for residuals of radom effects model fitted to SSIM measures')
dev.off()

