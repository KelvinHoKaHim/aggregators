# Observations:

1. In general, the dps of all alogirthms fall below 0.01 in the end. 

2. For UPGrad, MGDA, and Nash-MTL*, the third loss function (DEO2) drops to zero. Simultaneously, for all of these aggregators, just before the third loss function reached zero, the norm of dps and d started oscilliating. Then when the said function actually reached zero, dps took a drastic drop to zero. The same phenomenon occured for all the said aggregators.  

3. UPGrad showed drastic oscilliation well before convergence. If we run the current benchmark using Torch JD's official implementation, the same issue persisted. This mean the issue is inherent to UPGrad. By switching to UPGrad*, the issue disappeared. 
