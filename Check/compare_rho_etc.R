n_x <- 181; n_y <- 217; n_z <- 36

TE_values = c(0.01, 0.015,  0.02, 0.01, 0.03,  0.04, 0.01, 0.04, 0.08, 0.01, 0.06, 0.1)
TR_values = c( 0.6,   0.6,   0.6,    1,     1,    1,    2,    2,    2,    3,    3,   3)
min_TE = min(TE_values)
min_TR = min(TR_values)
TE_scale = 2.01 / min_TE
TR_scale = 2.01 / min_TR

mask_vec <- as.logical( as.matrix( read.csv("intermed/mask_vec.csv", header=F) )[,1] )



## Gold stand
W_LS_par_GS = as.matrix( read.csv("intermed/W_LS_par-noise-0-INU-00-GS-scale-1.csv", header=F) )

# W_LS_par_3D = W_LS_par.reshape(n_x, n_y, n_z, 3)  # Don't try to reshape in R - will make another problem
#pd.DataFrame(W_LS_par_3D[:,:,18, 0]).to_csv("figs/new/DL-LS-W-noise-5-INU-10-img-0-axis-1.csv", header=None, index=None)
#pd.DataFrame(W_LS_par_3D[:,:,18, 1]).to_csv("figs/new/DL-LS-W-noise-5-INU-10-img-1-axis-1.csv", header=None, index=None)
#pd.DataFrame(W_LS_par_3D[:,:,18, 2]).to_csv("figs/new/DL-LS-W-noise-5-INU-10-img-2-axis-1.csv", header=None, index=None)

# T_1 = -1/(c_1 * log(W_1))  # TR-new = c_1 TR
# T_2 = -1/(c_2 * log(W_2))  # TE-new = c_2 TE

W_LS_par_GS[, 2] = -1/(TR_scale * log(W_LS_par_GS[, 2]))
W_LS_par_GS[, 3] = -1/(TE_scale * log(W_LS_par_GS[, 3]))

W_LS_par_GS[mask_vec,] = 0


final <- c()



W_LS_par_LS = as.matrix( read.csv("intermed/W_LS_par-noise-1-INU-00.csv.gz", header=F) )
W_LS_par_DL = as.matrix( read.csv("intermed/DL-W_LS_par-noise-1-INU-00.csv.gz", header=F) )

W_LS_par_LS[, 2] = -1/(TR_scale * log(W_LS_par_LS[, 2]))
W_LS_par_LS[, 3] = -1/(TE_scale * log(W_LS_par_LS[, 3]))
W_LS_par_LS[mask_vec,] = 0

W_LS_par_DL[, 2] = -1/(TR_scale * log(W_LS_par_DL[, 2]))
W_LS_par_DL[, 3] = -1/(TE_scale * log(W_LS_par_DL[, 3]))
W_LS_par_DL[mask_vec,] = 0

final <- rbind(final,  cbind( colMeans( abs(W_LS_par_GS - W_LS_par_LS) ), colMeans( abs(W_LS_par_GS - W_LS_par_DL) ) ) )



W_LS_par_LS = as.matrix( read.csv("intermed/W_LS_par-noise-2.5-INU-00.csv.gz", header=F) )
W_LS_par_DL = as.matrix( read.csv("intermed/DL-W_LS_par-noise-2.5-INU-00.csv.gz", header=F) )

W_LS_par_LS[, 2] = -1/(TR_scale * log(W_LS_par_LS[, 2]))
W_LS_par_LS[, 3] = -1/(TE_scale * log(W_LS_par_LS[, 3]))
W_LS_par_LS[mask_vec,] = 0

W_LS_par_DL[, 2] = -1/(TR_scale * log(W_LS_par_DL[, 2]))
W_LS_par_DL[, 3] = -1/(TE_scale * log(W_LS_par_DL[, 3]))
W_LS_par_DL[mask_vec,] = 0

final <- rbind(final,  cbind( colMeans( abs(W_LS_par_GS - W_LS_par_LS) ), colMeans( abs(W_LS_par_GS - W_LS_par_DL) ) ) )



W_LS_par_LS = as.matrix( read.csv("intermed/W_LS_par-noise-5-INU-00.csv.gz", header=F) )
W_LS_par_DL = as.matrix( read.csv("intermed/DL-W_LS_par-noise-5-INU-00.csv.gz", header=F) )

W_LS_par_LS[, 2] = -1/(TR_scale * log(W_LS_par_LS[, 2]))
W_LS_par_LS[, 3] = -1/(TE_scale * log(W_LS_par_LS[, 3]))
W_LS_par_LS[mask_vec,] = 0

W_LS_par_DL[, 2] = -1/(TR_scale * log(W_LS_par_DL[, 2]))
W_LS_par_DL[, 3] = -1/(TE_scale * log(W_LS_par_DL[, 3]))
W_LS_par_DL[mask_vec,] = 0

final <- rbind(final,  cbind( colMeans( abs(W_LS_par_GS - W_LS_par_LS) ), colMeans( abs(W_LS_par_GS - W_LS_par_DL) ) ) )




W_LS_par_LS = as.matrix( read.csv("intermed/W_LS_par-noise-7.5-INU-00.csv.gz", header=F) )
W_LS_par_DL = as.matrix( read.csv("intermed/DL-W_LS_par-noise-7.5-INU-00.csv.gz", header=F) )

W_LS_par_LS[, 2] = -1/(TR_scale * log(W_LS_par_LS[, 2]))
W_LS_par_LS[, 3] = -1/(TE_scale * log(W_LS_par_LS[, 3]))
W_LS_par_LS[mask_vec,] = 0

W_LS_par_DL[, 2] = -1/(TR_scale * log(W_LS_par_DL[, 2]))
W_LS_par_DL[, 3] = -1/(TE_scale * log(W_LS_par_DL[, 3]))
W_LS_par_DL[mask_vec,] = 0

final <- rbind(final,  cbind( colMeans( abs(W_LS_par_GS - W_LS_par_LS) ), colMeans( abs(W_LS_par_GS - W_LS_par_DL) ) ) )




W_LS_par_LS = as.matrix( read.csv("intermed/W_LS_par-noise-10-INU-00.csv.gz", header=F) )
W_LS_par_DL = as.matrix( read.csv("intermed/DL-W_LS_par-noise-10-INU-00.csv.gz", header=F) )

W_LS_par_LS[, 2] = -1/(TR_scale * log(W_LS_par_LS[, 2]))
W_LS_par_LS[, 3] = -1/(TE_scale * log(W_LS_par_LS[, 3]))
W_LS_par_LS[mask_vec,] = 0

W_LS_par_DL[, 2] = -1/(TR_scale * log(W_LS_par_DL[, 2]))
W_LS_par_DL[, 3] = -1/(TE_scale * log(W_LS_par_DL[, 3]))
W_LS_par_DL[mask_vec,] = 0

final <- rbind(final,  cbind( colMeans( abs(W_LS_par_GS - W_LS_par_LS) ), colMeans( abs(W_LS_par_GS - W_LS_par_DL) ) ) )

print(final)
saveRDS(final, "final.rds")


print(xtable::xtable(final))
