"""Interface to all algorithm files."""
from gpi.algs.geppo import GePPO
from gpi.algs.getrpo import GeTRPO
from gpi.algs.gevmpo import GeVMPO

gen_algs = ['geppo','getrpo','gevmpo']

def init_alg(sim_seed,env,actor,critic,runner,ac_kwargs,alg_name,
    idx,save_path,save_freq,checkpoint_file,keep_checkpoints):
    """Initializes algorithm."""

    if alg_name in ['ppo','geppo']:    
        alg = GePPO(sim_seed,env,actor,critic,runner,ac_kwargs,
            idx,save_path,save_freq,checkpoint_file,keep_checkpoints)
    elif alg_name in ['trpo','getrpo']:
        alg = GeTRPO(sim_seed,env,actor,critic,runner,ac_kwargs,
            idx,save_path,save_freq,checkpoint_file,keep_checkpoints)
    elif alg_name in ['vmpo','gevmpo']:
        alg = GeVMPO(sim_seed,env,actor,critic,runner,ac_kwargs,
            idx,save_path,save_freq,checkpoint_file,keep_checkpoints)
    else:
        raise ValueError('invalid alg_name')

    return alg