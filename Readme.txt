# A mindmap of functions

expectation_maximization
    forward_back_threaded!
        forward_back_chunk!
            solve!
            update! #<- updates EM object
            forward_back!

    mstep_blocks
        mstep_major_block #<- maximizes over shared parameters
            log_likelihood_threaded
        mstep_k_block #<- maximizes over type specific parameters
            log_likelihood_threaded (with extra argument)
    mstep_types
    mstep_\pi\eta
    mstep_\sigma
    log_likelihood
    savepars_vec
    basic_model_fit

# ** double check but it looks like there's a whole extra likelihood.jl script we don't use
# ** also look to be many duplicate functions comparoing lowmem to lowmem_k
# ** log_likelihood_threaded_full! #<- this uses the update routine that uses all parameters by default. Do we use this at all?