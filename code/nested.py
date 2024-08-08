import sys, time
import numpy as np
import os
from numpy.random import normal, multivariate_normal

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
                        hfile = 'noname',
                        checkpoint_name='_dynesty_checkpoint.save',
                        checkpoint_interval=60,
                        # sampler kwargs
                        nested_bound='multi',
                        nested_sample='unif',
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
                        nested_stop_kwargs={'n_effective': 1000},
                        nested_save_bounds=False,
                        print_progress=True,
                        **extras):
    
    checkpoint_file = hfile + checkpoint_name
    print(checkpoint_file)

    # Check if checkpoint file exists and load state or initialize sampler
    if os.path.isfile(checkpoint_file):
        print('\n____checkpoint file found, restoring state____\n')
        dsampler = dynesty.DynamicNestedSampler.restore(checkpoint_file, pool=pool)
    else:
        print('\n____no checkpoint file found, initializing new sampler____\n')
        dsampler = dynesty.DynamicNestedSampler(lnprobfn, prior_transform, ndim,
                                               bound=nested_bound,
                                               sample=nested_sample,
                                               walks=nested_walks,
                                               bootstrap=nested_bootstrap,
                                               first_update=nested_first_update,
                                               update_interval=nested_update_interval,
                                               pool=pool, queue_size=queue_size, use_pool=use_pool
                                               )
    
    # Generator for nested sampling____________________________________________________
    '''
    attributes = dir(dsampler)

    # Filter out built-in attributes (those starting and ending with '__')
    custom_attributes = [attr for attr in attributes if not attr.startswith('__')]

    # Print custom attributes
    for attr in custom_attributes:
        value = getattr(dsampler, attr, None)
        print(f"{attr}: {value}")
    '''



    if os.path.isfile(checkpoint_file):
        dsampler.run_nested(# initial sampler parameters
                            n_effective = 1000,
                            dlogz_init=nested_dlogz_init,
                            nlive_init=nested_nlive_init,
                            maxiter_init = nested_maxiter_init,
                            maxcall_init = nested_maxcall_init,
                            live_points=nested_live_points,
                            # batched sampling parameters
                            nlive_batch = nested_nlive_batch,
                            maxbatch = nested_maxbatch,
                            maxiter_batch = nested_maxiter_batch,
                            maxcall_batch = nested_maxcall_batch,
                            use_stop = nested_use_stop,
                            wt_function = wt_function,
                            wt_kwargs = nested_weight_kwargs,
                            save_bounds=nested_save_bounds,
                            # checkpointing and output
                            print_progress=print_progress,
                            checkpoint_file=checkpoint_file,
                            checkpoint_every = checkpoint_interval,
                            # RESUME
                            resume=True)

    else:
        dsampler.run_nested(# initial sampler parameters
                            n_effective = 1000,
                            dlogz_init=nested_dlogz_init,
                            nlive_init=nested_nlive_init,
                            maxiter_init = nested_maxiter_init,
                            maxcall_init = nested_maxcall_init,
                            live_points=nested_live_points,
                            # batched sampling parameters
                            nlive_batch = nested_nlive_batch,
                            maxbatch = nested_maxbatch,
                            maxiter_batch = nested_maxiter_batch,
                            maxcall_batch = nested_maxcall_batch,
                            use_stop = nested_use_stop,
                            wt_function = wt_function,
                            wt_kwargs = nested_weight_kwargs,
                            save_bounds=nested_save_bounds,
                            # checkpointing and output
                            print_progress=print_progress,
                            checkpoint_file=checkpoint_file,
                            checkpoint_every = checkpoint_interval,
                            )
    print('\ndone!')
        
    print(dsampler.results.summary())

    return dsampler.results

    


def run_dynesty_sampler_umsteadnlich(lnprobfn, prior_transform, ndim,
                        verbose=True,
                        hfile = 'test',
                        init_checkpoint_name='dynesty_init_checkpoint.pkl',
                        dynamic_checkpoint_name='dynesty_dynamic_checkpoint.pkl',
                        checkpoint_interval=60,
                        # sampler kwargs
                        nested_bound='multi',
                        nested_sample='unif',
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
                        nested_save_bounds=False,
                        print_progress=True,
                        **extras):#JC: I'm guessing you took this function from the original code or have a good reason for doing it this way but just in case, I really reccommend in the future to use the run_nested command. It replaces almost everything from the first for loop onwards with one command and is much quicker to run. Not worth changing for this code but just for your information.
    
    init_checkpoint_file = hfile + init_checkpoint_name
    dynamic_checkpoint_file = hfile + dynamic_checkpoint_name

    print(init_checkpoint_file, dynamic_checkpoint_file)


    # Check if dynamic checkpoint file exists and load state
    if os.path.isfile(dynamic_checkpoint_file): #JC: use os.path.isfile instead of exists, this is more for safety. If *_checkpoint_name is blank (e.g. dynamic_checkpoint_file='') then exists returns true as long as hfile exists but isfile will not.
        print('\n____initial sampling complete, directly proceeding with dynamic sampling____\n')
        dsampler = dynesty.DynamicNestedSampler.restore(dynamic_checkpoint_file, pool=pool)#JC: You should put in pool=pool, I think the restore doesn't save the pool.
        resume_init = False
        resume_dyn = True
        
    else:
        # Check if initial checkpoint file exists and load state
        if os.path.isfile(init_checkpoint_file):
            print('\n____checkpoint file for initial sampling found, restoring state____\n')
            dsampler = dynesty.DynamicNestedSampler.restore(init_checkpoint_file, pool=pool)#JC: You should put in pool=pool, I think the restore doesn't save the pool.
            resume_init = True
            resume_dyn = False
        else:
            # Instantiate sampler
            dsampler = dynesty.DynamicNestedSampler(lnprobfn, prior_transform, ndim,
                                                    bound=nested_bound,
                                                    sample=nested_sample,
                                                    walks=nested_walks,
                                                    bootstrap=nested_bootstrap,
                                                    update_interval=nested_update_interval,
                                                    pool=pool, queue_size=queue_size, use_pool=use_pool
                                                    )
            resume_init = False
            resume_dyn = False
#______________________________________________INITIAL SAMPLING__________________________________________________________________________

       # Generator for initial nested sampling
    if not os.path.isfile(dynamic_checkpoint_file):#JC: You need to put in a check to see if you completed the init before. Even if the dynamic file doesn't exist, this may have finished.
        ncall = dsampler.ncall
        niter = dsampler.it - 1
        tstart = time.time()
        last_checkpoint_time = tstart

        sample_params = {'dlogz': nested_dlogz_init,
                         'resume': resume_init,
                         'nlive': nested_nlive_init,  
                         'maxcall': nested_maxcall_init,
                         'maxiter': nested_maxiter_init,}

        if not resume_init:
            sample_params.update({
                'live_points': nested_live_points
            })

        # Sample initial points
        for results in dsampler.sample_initial(**sample_params):#JC: You need to put in resume=resume_init

            try:
                # dynesty >= 2.0
                (worst, ustar, vstar, loglstar, logvol,
                logwt, logz, logzvar, h, nc, worst_it,
                propidx, propiter, eff, delta_logz, blob) = results
            except ValueError:
                # dynesty < 2.0
                (worst, ustar, vstar, loglstar, logvol,
                logwt, logz, logzvar, h, nc, worst_it,
                propidx, propiter, eff, delta_logz) = results

            if delta_logz > 1e6:
                delta_logz = np.inf
            ncall += nc
            niter += 1

            if print_progress:
                with np.errstate(invalid='ignore'):
                    logzerr = np.sqrt(logzvar)
                sys.stderr.write("\riter: {:d} | batch: {:d} | nc: {:d} | "
                                "ncall: {:d} | eff(%): {:6.3f} | "
                                "logz: {:6.3f} +/- {:6.3f} | "
                                "dlogz: {:6.3f} > {:6.3f}    "
                                .format(niter, 0, nc, ncall, eff, logz,
                                        logzerr, delta_logz, nested_dlogz_init))
                sys.stderr.flush()

            # Checkpointing
            current_time = time.time()
            if current_time - last_checkpoint_time > checkpoint_interval:
                dsampler.save(init_checkpoint_file)
                print(f'\ncheckpoint saved at {init_checkpoint_file}')
                last_checkpoint_time = current_time

        ndur = time.time() - tstart
        if verbose:
            print('\ndone dynesty (initial) in {0}s'.format(ndur))

        # Save initial checkpoint as dynamic checkpoint to start dynamic sampling from there
        dsampler.save(init_checkpoint_file)#JC: I recommend you put in another line here with dsampler.save(dynamic_checkpoint_file). This ensures the dynamic_checkpoint_file exists when init is completed.
        dsampler.save(dynamic_checkpoint_file)
        print(f'\nlast initial checkpoint saved at {init_checkpoint_file} and additionally saved as dynamic checkpoint at {dynamic_checkpoint_file}')

#______________________________________________DYNAMIC SAMPLING__________________________________________________________________________

    if nested_maxcall is None:
        nested_maxcall = sys.maxsize
    if nested_maxbatch is None:
        nested_maxbatch = sys.maxsize
    if nested_maxcall_batch is None:
        nested_maxcall_batch = sys.maxsize
    if nested_maxiter is None:
        nested_maxiter = sys.maxsize
    if nested_maxiter_batch is None:
        nested_maxiter_batch = sys.maxsize
    #print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAA NCALL', ncall, dsampler.ncall)
    #print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAA NITER', niter, dsampler.it, dsampler.it - 1)
    # Generator for dynamic sampling
    ncall = dsampler.ncall
    niter = dsampler.it - 1
    tstart = time.time()
    last_checkpoint_time = tstart

    for n in range(dsampler.batch, nested_maxbatch):
        # Update stopping criteria.
        dsampler.sampler.save_bounds = False
        res = dsampler.results
        mcall = min(nested_maxcall - ncall, nested_maxcall_batch)
        miter = min(nested_maxiter - niter, nested_maxiter_batch)
        if nested_use_stop:
            if dsampler.use_pool_stopfn:
                M = dsampler.M
            else:
                M = map
            stop, stop_vals = stop_function(res, nested_stop_kwargs,
                                            rstate=dsampler.rstate, M=M,
                                            return_vals=True)
            stop_val = stop_vals[2]
        else:
            stop = False
            stop_val = np.NaN

        # If we have either likelihood calls or iterations remaining,
        # run our batch.
        if mcall > 0 and miter > 0 and not stop:
            # Compute our sampling bounds using the provided
            # weight function.
            logl_bounds = wt_function(res, nested_weight_kwargs)
            lnz, lnzerr = res.logz[-1], res.logzerr[-1]

            sample_params_dyn = {'resume': resume_dyn}

            if not resume_dyn:
                sample_params_dyn.update({
                    'nlive_new': nested_nlive_batch,
                    'logl_bounds': logl_bounds,
                    'maxiter': miter,
                    'maxcall': mcall,
                    'save_bounds': nested_save_bounds,
                })

            for results in dsampler.sample_batch(**sample_params_dyn):#JC: You need to put in resume=resume_dyn in the call according to the docs

                try:
                    # dynesty >= 2.0
                    (worst, ustar, vstar, loglstar, nc,
                     worst_it, propidx, propiter, eff, blob) = results
                except(ValueError):
                    # dynesty < 2.0
                    (worst, ustar, vstar, loglstar, nc,
                     worst_it, propidx, propiter, eff) = results
                ncall += nc
                niter += 1
                if print_progress:
                    sys.stderr.write("\riter: {:d} | batch: {:d} | "
                                     "nc: {:d} | ncall: {:d} | "
                                     "eff(%): {:6.3f} | "
                                     "loglstar: {:6.3f} < {:6.3f} "
                                     "< {:6.3f} | "
                                     "logz: {:6.3f} +/- {:6.3f} | "
                                     "stop: {:6.3f}    "
                                     .format(niter, n+1, nc, ncall,
                                             eff, logl_bounds[0], loglstar,
                                             logl_bounds[1], lnz, lnzerr,
                                             stop_val))
                    sys.stderr.flush()

                # Checkpointing
                current_time = time.time()
                if current_time - last_checkpoint_time > checkpoint_interval:
                    dsampler.save(dynamic_checkpoint_file)
                    print(f'\ncheckpoint saved at {dynamic_checkpoint_file}')
                    last_checkpoint_time = current_time

            dsampler.combine_runs()
        else:
            # We're done!
            break

    ndur = time.time() - tstart
    if verbose:
        print('done dynesty (dynamic) in {0}s'.format(ndur))

    # Final checkpoint save
    dsampler.save(dynamic_checkpoint_file)

    return dsampler.results
