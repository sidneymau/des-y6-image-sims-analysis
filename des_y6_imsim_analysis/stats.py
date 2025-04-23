import numpy as np
import numpyro
import scipy.stats
from jax import random
from numpyro.infer import (
    MCMC,
    NUTS,
    SVI,
    Trace_ELBO,
    autoguide,
    init_to_value,
)


def measure_map(
    *,
    model_module,
    model_data,
    seed,
    num_steps=50_000,
    learning_rate=1e-3,
    progress_bar=False,
):
    """Find the MAP estimate of the model parameters using Adam.

    Parameters
    ----------
    model_module : module
        The module containing the model.
    model_data : dict
        The data to be used in the model. Built by calling `model_module.make_model_data`.
    seed : int
        The random seed.
    num_steps : int, optional
        The number of optimization steps. Default is 50,000.
    learning_rate : float, optional
        The learning rate for the Adam optimizer. Default is 1e-3.
    progress_bar : bool, optional
        Whether to show a progress bar during optimization. Default is False.

    Returns
    -------
    map_params : dict
        The MAP estimate of the model parameters. The keys are the parameter names
        defined in the numpyro model and the values are the parameter values.
    """
    guide = autoguide.AutoDelta(model_module.model)
    optimizer = numpyro.optim.Adam(learning_rate)
    svi = SVI(model_module.model, guide, optimizer, Trace_ELBO())
    svi_results = svi.run(
        random.PRNGKey(seed),
        num_steps,
        progress_bar=progress_bar,
        **model_data,
    )
    map_params = svi_results.params

    for key in list(map_params.keys()):
        if key.endswith("_auto_loc"):
            new_key = key[: -len("_auto_loc")]
            map_params[new_key] = map_params[key]
            del map_params[key]

    for k, v in (model_data["fixed_param_values"] or {}).items():
        map_params[k] = v

    return map_params


def compute_model_chi2_info(*, model_module, model_data, data, params):
    """Compute the chi2, DOF, and p-value for the model at the parameter values.

    Parameters
    ----------
    model_module : module
        The module containing the model.
    model_data : dict
        The data to be used in the model. Built by calling `model_module.make_model_data`.
    data : dict
        The raw data. Built by calling `des_y6_image_sims_analysis.utils.read_data` on
        the HDF5 file.
    params : dict
        The parameter values. The keys are the parameter names defined in the numpyro
        model and the values are the parameter values.

    Returns
    -------
    chi2_info : dict
        A dictionary containing the keys `chi2`, `dof`, and `p_value` with the corresponding
        values.
    """

    fixed_pars = set((model_data["fixed_param_values"] or {}).keys())
    all_pars = set(params.keys())
    free_pars = all_pars - fixed_pars

    model_mn = model_module.model_mean(
        params=params,
        **model_data,
    )
    dmn = model_mn - data.mn
    chi2 = np.dot(dmn, np.dot(np.linalg.inv(model_data["cov"]), dmn.T))
    dof = data.mn.shape[0] - len(free_pars)

    return {
        "chi2": chi2,
        "dof": dof,
        "p_value": scipy.stats.chi2.sf(chi2, dof),
        "model_mn": model_mn,
    }


def run_mcmc(*, model_module, model_data, init_params, seed, **mcmc_kwargs):
    """Run the No U-turn Sampler on the model + data.

    Parameters
    ----------
    model_module : module
        The module containing the model.
    model_data : dict
        The data to be used in the model. Built by calling `model_module.make_model_data`.
    data : dict
        The raw data. Built by calling `des_y6_image_sims_analysis.utils.read_data` on
        the HDF5 file.
    init_params : dict
        The initial parameter values. The keys are the parameter names defined in the numpyro
        model and the values are the parameter values.
    seed : int
        The random seed.
    mcmc_kwargs : dict
       Additional keyword arguments to pass to either the NUTS or MCMC constructor.

    Return
    ------
    mcmc : MCMC
        The MCMC results.
    """
    max_tree_depth = mcmc_kwargs.pop("max_tree_depth", 10)
    dense_mass = mcmc_kwargs.pop("dense_mass", False)
    mcmc_kwargs["num_warmup"] = mcmc_kwargs.get("num_warmpup", 500)
    mcmc_kwargs["num_samples"] = mcmc_kwargs.get("num_samples", 1000)
    mcmc_kwargs["num_chains"] = mcmc_kwargs.get("num_chains", 4)

    if "progress_bar" not in mcmc_kwargs:
        mcmc_kwargs["progress_bar"] = False

    rng_key = random.PRNGKey(seed)
    rng_key, rng_key_ = random.split(rng_key)

    kernel = NUTS(
        model_module.model,
        init_strategy=init_to_value(values=init_params),
        max_tree_depth=max_tree_depth,
        dense_mass=dense_mass
    )
    mcmc = MCMC(
        kernel,
        **mcmc_kwargs,
    )
    mcmc.run(
        rng_key_,
        **model_data,
    )

    # jax is async so we have to wait until the chains are done
    test_key = list(init_params.keys())[0]
    mcmc.get_samples()[test_key].block_until_ready()

    return mcmc
