# Truncation on plotting

1. For all seeds, the dps of Nash-MTL behaved abnoramlly as it continued to increase. Hence, plot of Nash-MTL are clipped at the 15000th iteration, where dps is around 0.15

2. For seed 64, UPGrad started osciliating at around the 27000th iteration. It was truncated at the 29000th iteration. Similarly, UPGrad in seed 100 started osciliating at around the 30000th iteration. It was truncated at the 320000th iteration.

3. The drastic drop to zero in dps for MGDA and Nash-MTL* are due to DEO2 reaching zero at the point. This mean the one column of $J^T$ is zero and MGDA distributed all the weights to the zero column, resulting at $||d_{PS}|| = 0$.

