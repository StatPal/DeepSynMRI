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

library(ggResidpanel)
resid_panel(mod_test)

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

library(ggResidpanel)
resid_panel(mod_test)

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


