import sys, time
import numpy as np
from numpy.random import normal, multivariate_normal
import os 
try:
    import nestle
except(ImportError):
    pass


try:
    import dynesty
    from dynesty.utils import *
    from dynesty.dynamicsampler import _kld_error
except(ImportError):
    pass


__all__ = ["run_nestle_sampler", "run_dynesty_sampler"]


def run_nestle_sampler(lnprobfn, model, verbose=True,
                       callback=None,
                       nestle_method='multi', nestle_npoints=200,
                       nestle_maxcall=int(1e6), nestle_update_interval=None,
                       **kwargs):

    result = nestle.sample(lnprobfn, model.prior_transform, model.ndim,
                           method=nestle_method, npoints=nestle_npoints,
                           callback=callback, maxcall=nestle_maxcall,
                           update_interval=nestle_update_interval)
    return result


def run_dynesty_sampler(lnprobfn, prior_transform, ndim,
                        verbose=True,
                        # sampler kwargs
                        hfile = 'prospector_dynesty_checkpoint.save',
                        nested_bound='multi',
                        nested_sample='unif',
                        nested_method='rwalk',
                        nested_rstate=42, #None,
                        nested_walks=25,
                        nested_update_interval=0.6,
                        nested_bootstrap=0,
                        pool=None,
                        use_pool={},
                        queue_size=1,
                        # init sampling kwargs
                        nested_nlive_init=100,
                        nested_dlogz_init=0.02,
                        nested_maxiter_init=None,
                        nested_maxcall_init=None,
                        nested_live_points=None,
                        # batch sampling kwargs
                        nested_maxbatch=None,
                        nested_nlive_batch=100,
                        nested_maxiter_batch=None,
                        nested_maxcall_batch=None,
                        nested_use_stop=True,
                        # overall kwargs
                        nested_maxcall=None,
                        nested_maxiter=None,
                        nested_first_update={},
                        stop_function=None,
                        wt_function=None,
                        nested_weight_kwargs={'pfrac': 1.0},
                        nested_stop_kwargs={},
                        nested_save_bounds=True,
                        print_progress=True,
                        **extras):
    
    nested_checkpoint_file = hfile

    if os.path.isfile(nested_checkpoint_file):
        print('\n____checkpoint file found, restoring state____\n')
        resume_sampler  =   True
        dsampler        =   dynesty.DynamicNestedSampler.restore(nested_checkpoint_file, pool=pool)
        
    else:
        print('\n____no checkpoint file found, initializing new sampler____\n')
        resume_sampler  =   False
        dsampler        =   dynesty.DynamicNestedSampler(lnprobfn, prior_transform, ndim,
                                                bound = nested_bound,
                                                #method = nested_method,
                                                update_interval = nested_update_interval,
                                                first_update = nested_first_update,
                                                #state = nested_rstate,
                                                sample = nested_sample,
                                                walks=nested_walks,
                                                bootstrap=nested_bootstrap,
                                                pool=pool, queue_size=queue_size, use_pool=use_pool
                                                )
    

    dsampler.run_nested(nlive_init      =   nested_nlive_init,  
                         maxiter_init   =   nested_maxiter_init,
                         maxcall_init   =   nested_maxcall_init,
                         dlogz_init     =   nested_dlogz_init,
                         logl_max_init  =   np.inf,
                         n_effective_init=  np.inf,
                         nlive_batch    =   nested_nlive_batch,
                         wt_function    =   wt_function,
                         wt_kwargs      =   nested_weight_kwargs, 
                         maxiter_batch  =   nested_maxiter_batch,
                         maxcall_batch  =   nested_maxcall_batch,
                         maxiter        =   nested_maxiter,
                         maxcall        =   nested_maxcall,
                         maxbatch       =   nested_maxbatch,
                         n_effective    =   None,
                         stop_function  =   stop_function,
                         stop_kwargs    =   nested_stop_kwargs,
                         use_stop       =   False,
                         save_bounds    =   nested_save_bounds,
                         print_progress =   print_progress,
                         print_func     =   None,
                         live_points    =   nested_live_points,
                         resume         =   resume_sampler,
                         checkpoint_file=   nested_checkpoint_file,
                         checkpoint_every=  60
                        )    
    
    return dsampler.results