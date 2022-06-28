import numpy as np
import tensorflow as tf

from gpi.common.initializers import init_seeds
from gpi.common.logger import Logger

class BaseAlg:
    """Base algorithm class for training."""

    def __init__(self,seed,env,actor,critic,runner,ac_kwargs,
        idx,save_path,save_freq,checkpoint_file,keep_checkpoints):
        """Initializes BaseAlg class.

        Args:
            seed (int): random seed
            env (NormEnv): normalized environment
            actor (Actor): policy class
            critic (Critic): value function class
            runner (Runner): runner class to generate samples
            ac_kwargs (dict): dictionary containing actor and critic kwargs
            idx (int): index to associate with checkpoint files
            save_path (str): path where checkpoint files are saved
            save_freq (float): number of steps between checkpoints
            checkpoint_file (str): name of checkpoint file for saving
            keep_checkpoints (bool): keep final dict of all checkpoints
        """

        self.seed = seed
        self.env = env
        self.actor = actor
        self.critic = critic
        self.runner = runner

        self.ac_kwargs = ac_kwargs

        self.save_path = save_path
        self.checkpoint_name = '%s_%d'%(checkpoint_file,idx)
        self.save_freq = save_freq
        self.keep_checkpoints = keep_checkpoints

        init_seeds(self.seed,env)
        self.logger = Logger()

        self._ac_setup()

    def _ac_setup(self):
        """Sets up actor and critic kwargs as class attributes."""
        self.critic_lr = self.ac_kwargs['critic_lr']        
        self.critic_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.critic_lr)

        self.update_it = self.ac_kwargs['update_it']
        self.nminibatch = self.ac_kwargs['nminibatch']

        self.adv_center = self.ac_kwargs['adv_center']
        self.adv_scale = self.ac_kwargs['adv_scale']
        self.adv_clip = self.ac_kwargs['adv_clip']

        self.eps_ppo = self.ac_kwargs['eps_ppo']
        self.eps = self.eps_ppo * self.ac_kwargs['eps_mult']
        self.eps_vary = self.ac_kwargs['eps_vary']

    def _update(self):
        """Updates actor and critic."""
        raise NotImplementedError # Algorithm specific

    def _critic_update(self,s_all,rtg_all,weights_all):
        """Updates critic.

        Args:
            s_all (np.ndarray): states
            rtg_all (np.ndarray): rewards-to-go
            weights_all (np.ndarray): policy weights
        """
        n_samples = s_all.shape[0]
        n_batch = int(n_samples / self.nminibatch)

        v_loss_all = 0
        vg_norm_all = 0

        # Minibatch update loop for critic
        for sweep_it in range(self.update_it):
            idx = np.arange(n_samples)
            np.random.shuffle(idx)
            sections = np.arange(0,n_samples,n_batch)[1:]

            batches = np.array_split(idx,sections)
            if (n_samples % n_batch != 0):
                batches = batches[:-1]

            for batch_idx in batches:
                # Active data
                s_active = s_all[batch_idx]
                rtg_active = rtg_all[batch_idx]
                weights_active = weights_all[batch_idx]

                # Critic update
                with tf.GradientTape() as tape:
                    V = self.critic.value(s_active)
                    v_loss = 0.5 * tf.reduce_mean(
                        weights_active * tf.square(rtg_active - V))
                
                vg = tape.gradient(v_loss,self.critic.trainable)
                self.critic_optimizer.apply_gradients(
                    zip(vg,self.critic.trainable))

                v_loss_all += v_loss
                vg_norm_all += tf.linalg.global_norm(vg)

        v_loss_ave = v_loss_all.numpy() / ((sweep_it+1)*len(batches))
        vg_norm_ave = vg_norm_all.numpy() / ((sweep_it+1)*len(batches))

        log_critic = {
            'critic_loss':      v_loss_ave,
            'critic_grad_norm': vg_norm_ave
        }
        self.logger.log_train(log_critic)

    def learn(self,sim_size,no_op_batches,params):
        """Main training loop.

        Args:
            sim_size (float): number of steps to run training loop
            no_op_batches (int): number of no-op batches at start of training
                to initialize running normalization stats
            params (dict): dictionary of input parameters to pass to logger
        
        Returns:
            Name of checkpoint file.
        """

        checkpt_idx = 0
        if self.save_freq is None:
            checkpoints = np.array([sim_size])
        else:
            checkpoints = np.concatenate(
                (np.arange(0,sim_size,self.save_freq)[1:],[sim_size]))

        # No-op batches to initialize running normalization stats
        for _ in range(no_op_batches):
            self.runner.generate_batch(self.env,self.actor)
        s_raw, rtg_raw = self.runner.get_env_info()
        self.env.update_rms(s_raw,rtg_raw)
        self.runner.reset()

        # Main training loop
        sim_total = 0
        while sim_total < sim_size:
            self.runner.generate_batch(self.env,self.actor)

            self._update()

            s_raw, rtg_raw = self.runner.get_env_info()
            self.env.update_rms(s_raw,rtg_raw)

            log_info = self.runner.get_log_info()
            self.logger.log_train(log_info)

            sim_total += self.runner.steps_total

            self.runner.update()

            # Save training data to checkpoint file
            if sim_total >= checkpoints[checkpt_idx]:
                self.dump_and_save(params,sim_total)
                checkpt_idx += 1
        
        return self.checkpoint_name
        
    def dump_and_save(self,params,steps):
        """Saves training data to checkpoint file and resets logger."""
        self.logger.log_params(params)

        final = {
            'actor_weights':    self.actor.get_weights(),
            'critic_weights':   self.critic.get_weights(),

            's_t':              self.env.s_rms.t_last,
            's_mean':           self.env.s_rms.mean,
            's_var':            self.env.s_rms.var,

            'r_t':              self.env.r_rms.t_last,
            'r_mean':           self.env.r_rms.mean,
            'r_var':            self.env.r_rms.var,

            'steps':            steps
        }
        self.logger.log_final(final)

        self.logger.dump_and_save(self.save_path,self.checkpoint_name,
            self.keep_checkpoints)
        self.logger.reset()