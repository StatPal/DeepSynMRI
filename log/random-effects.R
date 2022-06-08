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

effect_mat <- matrix( tmp_dat %>%
    group_by(interaction(DL, errs)) %>%
    dplyr::summarize(Mean = mean(vals, na.rm=TRUE)) %>% pull(Mean), ncol=2, byrow = TRUE)

effect_mat



library(lme4)
mod1 <- lmer(vals ~ DL + (DL | img), tmp_dat)
mod1




# fm1 <- lmer(Reaction ~ Days + (Days | Subject), sleepstudy)
# fm1

# mol <- filter(pld, phylum == "Mollusca")
# mix.int <- lmer(l.pld ~ l.temp + (1 | species), data = mol)
# summary(mix.int)



# ## Create an example:
# x <- rep(1:3, 5)
# b <- rep(1:5, each = 3)
# eff <- rnorm(5, 0, 10)
# y <- 2 + x * (-20) + eff[b] + rnorm(15)

# cbind(y, x, b)
# mod_test <- lmer(y ~ x + (1 | b))   # random effect for intercept
# mod_test
# # Linear mixed model fit by REML ['lmerMod']
# # Formula: y ~ x + (1 | b)
# # REML criterion at convergence: 53.2821
# # Random effects:
# #  Groups   Name        Std.Dev.
# #  b        (Intercept) 8.2490  
# #  Residual             0.5769  
# # Number of obs: 15, groups:  b, 5
# # Fixed Effects:
# # (Intercept)            x  
# #       6.888      -20.310  

# summary(mod_test)

# coef(mod_test)

# # mod_test <- lmer(y ~ x + (x | b))  # random effects on both terms
# # mod_test









# ## 2 class?


# ## Create an example:
# x <- rep(1:2, 5)
# b <- rep(1:5, each = 2)
# eff <- rnorm(5, 0, 10)
# y <- 2 + x * (-20) + eff[b] + rnorm(10)

# cbind(y, x, b)
# mod_test <- lmer(y ~ x + (1 | b))   # random effect for intercept
# mod_test

# summary(mod_test)
# # Random effects:
# #  Groups   Name        Variance Std.Dev.
# #  b        (Intercept) 233.1317 15.2686 
# #  Residual               0.5742  0.7577 
# # Number of obs: 10, groups:  b, 5

# # Fixed effects:
# #             Estimate Std. Error t value
# # (Intercept)   6.6631     6.8703    0.97
# # x           -19.1425     0.4792  -39.94
#             # this ~ -20
# coef(mod_test)$b
# eff










# ## Create an example with multiple factors:
# DL <- rep(1:2, each = 5 * 3)             # DL means DL or not, 3 subjects
# errs <- rep(1:5, each = 3, times = 2)   # error
# sub <- rep(1:3, times = 2 * 5)          # This is the random part
# err_vals <- c(1, 2.5, 5, 7.5, 10)
# # effect_mat <- matrix(c(0.5, 1,   1.25, 1.1,   2.5, 2,    4, 3,   6, 4), ncol=2, byrow = T)

# eff_sub <- rnorm(3, 0, 1)
# y <- 2 + eff_sub[sub] + rnorm(30, 0, 0.05)
# for(i in 1:30){
#     y[i] <- y[i] + 1 * effect_mat[errs[i], DL[i]]
# }
# simu_dat <- data.frame(cbind(y, DL, errs, sub))
# ggplot(simu_dat) + 
#     aes(x = DL, y = y, group = errs, shape=factor(DL), color=factor(errs), linetype = factor(errs)) +
#     geom_point(aes(size=1)) + 
#     facet_grid(cols = vars(sub)) +
#     geom_line()




# (mod_test <- lmer(y ~ DL * err_vals[errs] + (1 | sub)))   # random effect for intercept

# summary(mod_test)
# coef(mod_test)$sub
# eff_sub


# (mod_test <- lmer(y ~ DL * err_vals[errs] + (DL * err_vals[errs] | sub)))   # random effect for everything

# summary(mod_test)
# coef(mod_test)$sub
# eff_sub


# (mod_test <- lmer(y ~ DL  + (DL | sub)))   # random effect for sub only

# summary(mod_test)
# coef(mod_test)$sub
# eff_sub


# (mod_test <- lmer(y ~ DL * err_vals[errs] + (DL | sub)))   # random effect for everything
# # How meaningful is this?

# summary(mod_test)
# coef(mod_test)$sub
# eff_sub







## Random effects model to the real data
DL <- rep(1:2, each = 5 * 9)             # DL means DL or not, 3 subjects
errs <- rep(1:5, each = 9, times = 2)   # error
sub <- rep(1:9, times = 2 * 5)          # This is the random part
err_vals <- c(1, 2.5, 5, 7.5, 10)

ggplot(tmp_dat) + 
    aes(x = DL, y = vals, group = errs, shape=method, color=factor(errs), linetype = factor(errs)) +
    geom_point(aes(size=1)) + 
    facet_grid(cols = vars(img)) +
    geom_line()

(mod_test <- lmer(vals ~ DL * errs + (1 | img), tmp_dat))   # random effect for intercept

summary(mod_test)
coef(mod_test)$img




(mod_test <- lmer(vals ~ DL * errs + (DL * errs | img), tmp_dat))

summary(mod_test)
coef(mod_test)$img



(mod_test <- lmer(vals ~ DL * errs + (DL | img), tmp_dat))

summary(mod_test)
coef(mod_test)$img


(mod_test <- lmer(vals ~ DL * errs + (errs | img), tmp_dat))

summary(mod_test)
coef(mod_test)$img
# Makes more sense without singularity????





