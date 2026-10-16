"""Standard surjective monotone DP with a beta-binomial warmup prior.
This is an implementation of established alignment ideas, not a novelty claim.
"""
import numpy as np
from numba import njit
from functools import lru_cache
from scipy.special import gammaln,betaln
@njit(cache=True)
def maximum_path(scores):
 frames,tokens=scores.shape
 if tokens<1 or frames<tokens:raise ValueError('Impossible surjective path')
 dp=np.full((frames,tokens),-np.inf,np.float32);advance=np.zeros((frames,tokens),np.bool_)
 dp[0,0]=scores[0,0]
 for frame in range(1,frames):
  first=max(0,tokens-(frames-frame));last=min(tokens,frame+1)
  for token in range(first,last):
   stay=dp[frame-1,token];previous=dp[frame-1,token-1] if token else -np.inf
   take=previous>stay;dp[frame,token]=scores[frame,token]+(previous if take else stay);advance[frame,token]=take
 if not np.isfinite(dp[-1,-1]):raise ValueError('No finite path')
 path=np.zeros(frames,np.int64);token=tokens-1
 for frame in range(frames-1,-1,-1):
  path[frame]=token
  if advance[frame,token]:token-=1
 return path
@lru_cache(maxsize=2048)
def beta_binomial_prior(frames,tokens):
 k=np.arange(tokens)[None,:];n=tokens-1
 a=np.arange(1,frames+1)[:,None];b=frames+1-a
 prior=gammaln(n+1)-gammaln(k+1)-gammaln(n-k+1)+betaln(k+a,n-k+b)-betaln(a,b)
 prior=prior.astype('float32');assert np.isfinite(prior).all()
 return prior
