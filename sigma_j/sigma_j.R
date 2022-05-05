library(reticulate)
nib = import("nibabel")

img = nib$load(paste0('../../data/noise-5-INU-20/brainweb_', 0, '.mnc.gz'))
data = img$get_fdata()
dim(data)

data_reshaped = aperm(data, c(3, 2, 1))
image(data_reshaped[,,18])

mask_all = nib$load('../../data/mask/subject47_crisp_v.mnc.gz')
mask_all = mask_all$get_fdata()
dim(mask_all)
mask = (mask_all == 0)

mask_reshaped_mid = aperm(mask, c(3, 2, 1))
mask_reshaped = mask_reshaped_mid[(1:181)*2, (1:217)*2, (1:36)*10]
dim(mask_reshaped)




## Other noises etc
library(symR)
train_ind <- c(1, 9, 10)
phantom <- array(dim = c(181 * 217 * 36, 12))
test_ind <- setdiff(1:ncol(phantom), train_ind)

n <- nrow(phantom)
mask <- mask_reshaped
mask_vec <- 1 - c(mask)

for (i in 1:3) {
    img = nib$load(paste0('../../data/noise-5-INU-20/brainweb_', i-1, '.mnc.gz'))    ## BUG, this was 0, not i
    data = img$get_fdata()
    data_reshaped = aperm(data, c(3, 2, 1))
    phantom[, train_ind[i]] <- c(data_reshaped)
}

for (i in 1:9) {
    img = nib$load(paste0('../../data/test-noise-0-check/brainweb_', i-1, '.mnc.gz'))
    data = img$get_fdata()
    data_reshaped = aperm(data, c(3, 2, 1))
    phantom[, test_ind[i]] <- c(data_reshaped)
}

colMeans(phantom)
apply(phantom, 2, max)

max(phantom, na.rm = T)
phantom <- phantom * 400 / max(phantom, na.rm = T)
# phantom[phantom == 0.0] <- 0.5 ## Pre-processing to remove the -Inf issue in likelihood.


# sigma_values <- array(dim=3)
# for (i in 1:3) {
#     sigma_values[i] <- symR::estimate.sigma.j(phantom[,i])
# }
# write.table(sigma_values, "BW-noise-5-INU-20.csv", sep = ",",
#     row.names = F, col.names = F)

