library(symR)

TE_vals = c(0.5, 0.45, 0.4, 0.35)
TR_vals = c(0.3, 0.25, 0.2, 0.24)
W_i = c(120.4, 0.2, 0.5)
train_im = c(200, 150.4, 175, 220)


bloch(W_i, 0.5, 0.3)
bloch(W_i, 0.4, 0.35)
bloch(W_i, TE_vals, TR_vals)
# bloch(W_i, TE_vals, TR_vals, 40)

# sum( (bloch(W_i, TE_vals, TR_vals, 20) - train_im)^2)
# bloch(W_i, TE_vals, TR_vals) - train_im
sum( (bloch(W_i, TE_vals, TR_vals) - train_im)^2)



## LS:
train_im <- t(as.matrix(train_im))
tmp <- symr(NULL, method = "LS", c(3,1,1,1), TE_vals, TR_vals, c(0,0,0), train_im, 1, 1, 1, mask)

print(round(tmp, 6))




## Performance:

cat('\n\n\n Many voxels\n\n')
W <- rbind(c(50, 0.01, 0.003), c(36, 0.02, 0.04), c(56, 0.02, 0.04), c(106, 0.2, 0.04))
# W <- rbind(c(50, 0.01, 0.003), c(36, 0.02, 0.04))
TE <- c(0.01, 0.03, 0.04, 0.01)
TR <- c(0.6, 0.6, 1, 0.8)
(train_new <- bloch.image(W, TE, TR))
print(W)


## Now check more voxels:
sigma_val <- c(0,0)
mask_vec <- c(0,0,0,0)
est_W <- symr(NULL, method = "LS", c(3,2,1,1), TE, TR, sigma_val, train_new, 1, 1, 1, mask_vec, maxiter.LS = 25L)

print(round(est_W, 6))


## Predict again from estimated:
final <- bloch.image(est_W, TE, TR)

print(round(final, 6))

cat('\n\n\n Measure\n\n')
print(   rowMeans( abs( bloch.image(est_W, TE, TR) - train_new))  )
