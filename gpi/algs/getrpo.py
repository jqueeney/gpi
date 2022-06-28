import numpy as np
import tensorflow as tf

from gpi.algs.base_alg import BaseAlg
from gpi.common.ac_utils import list_to_flat
from gpi.common.update_utils import cg, make_F

class GeTRPO(BaseAlg):
    """Algorithm class for GeTRPO. TRPO is a special case."""

    def __init__(self,seed,env,actor,critic,runner,ac_kwargs,
        idx,save_path,save_freq,checkpoint_file,keep_checkpoints):
        """Initializes GeTRPO class. See BaseAlg for details."""
        super(GeTRPO,self).__init__(seed,env,actor,critic,runner,ac_kwargs,
            idx,save_path,save_freq,checkpoint_file,keep_checkpoints)
    
    def _ac_setup(self):
        """Sets up actor and critic kwargs as class attributes."""
        super(GeTRPO,self)._ac_setup()
       
        self.delta = np.square(self.eps) / 2

        self.cg_it = self.ac_kwargs['cg_it']
        self.trust_sub = self.ac_kwargs['trust_sub']
        self.kl_maxfactor = self.ac_kwargs['kl_maxfactor']
        self.trust_damp = self.ac_kwargs['trust_damp']

    def _update(self):
        """Updates actor and critic."""
        data_all = self.runner.get_update_info(self.actor,self.critic)
        (s_all, a_all, adv_all, rtg_all, neglogp_old_all, kl_info_all, 
            weights_all) = data_all
        self._critic_update(s_all,rtg_all,weights_all)
        self._actor_update(s_all,a_all,adv_all,neglogp_old_all,kl_info_all,
            weights_all)

    def _actor_update(self,s_all,a_all,adv_all,neglogp_old_all,kl_info_all,
        weights_all):
        """Updates actor.
        
        Args:
            s_all (np.ndarray): states
            a_all (np.ndarray): actions
            adv_all (np.ndarray): advantages
            neglogp_old_all (np.ndarray): negative log probabilities
            kl_info_all (np.ndarray): info needed to calculate KL divergence
            weights_all (np.ndarray): policy weights
        """
        if self.eps_vary:
            neglogp_pik_all = self.actor.neglogp_pik(s_all,a_all)
            offpol_ratio = tf.exp(neglogp_old_all - neglogp_pik_all)            
            eps_old = tf.reduce_mean(weights_all * tf.abs(offpol_ratio-1.))
            self.eps = np.maximum(self.eps_ppo - eps_old,0.0)
            self.delta = np.square(self.eps) / 2
        
        pg_vec = self._get_neg_pg(s_all,a_all,adv_all,neglogp_old_all,
            weights_all,flat=True) * -1

        if np.allclose(pg_vec,0) or self.delta==0.0:
            eta_v_flat = np.zeros_like(pg_vec)
        else:
            F = make_F(self.actor,s_all,weights_all,self.trust_sub,
                self.trust_damp)
            v_flat = cg(F,pg_vec,cg_iters=self.cg_it)

            vFv = np.dot(v_flat,F(v_flat))
            eta = np.sqrt(2*self.delta/vFv)
            eta_v_flat = eta * v_flat

        self._backtrack(eta_v_flat,s_all,a_all,adv_all,neglogp_old_all,
            kl_info_all,weights_all)

    def _get_neg_pg(self,s_all,a_all,adv_all,neglogp_old_all,
        weights_all,flat=True):
        """Calculates negative gradient of surrogate objective.

        Args:
            s_all (np.ndarray): states
            a_all (np.ndarray): actions
            adv_all (np.ndarray): advantages
            neglogp_old_all (np.ndarray): negative log probabilities
            weights_all (np.ndarray): policy weights
            flat (bool): if True, flattens gradient
        
        Returns:
            Negative gradient of surrogate objective w.r.t. policy parameters.
        """

        neglogp_pik_all = self.actor.neglogp_pik(s_all,a_all)
        offpol_ratio = tf.exp(neglogp_old_all - neglogp_pik_all)

        adv_mean = (np.mean(offpol_ratio * weights_all * adv_all) / 
            np.mean(offpol_ratio * weights_all))
        adv_std = np.std(offpol_ratio * weights_all * adv_all) + 1e-8

        if self.adv_center:
            adv_all = adv_all - adv_mean 
        if self.adv_scale:
            adv_all = adv_all / adv_std
        if self.adv_clip:
            adv_all = np.clip(adv_all,-self.adv_clip,self.adv_clip)

        with tf.GradientTape() as tape:
            neglogp_cur_all = self.actor.neglogp(s_all,a_all)
            ratio = tf.exp(neglogp_old_all - neglogp_cur_all)
            
            pg_loss_surr = ratio * adv_all * -1
            pg_loss = tf.reduce_mean(pg_loss_surr*weights_all)
        
        neg_pg = tape.gradient(pg_loss,self.actor.trainable)

        if flat:
            neg_pg = list_to_flat(neg_pg)

        return neg_pg

    def _backtrack(self,eta_v_flat,s_all,a_all,adv_all,neglogp_old_all,
        kl_info_all,weights_all):
        """Performs backtracking line search and updates policy.

        Args:
            eta_v_flat (np.ndarray): pre backtrack flattened policy update
            s_all (np.ndarray): states
            a_all (np.ndarray): actions
            adv_all (np.ndarray): advantages
            neglogp_old_all (np.ndarray): negative log probabilities
            kl_info_all (np.ndarray): info needed to calculate KL divergence
            weights_all (np.ndarray): policy weights
        """
        # Current policy info
        ent = tf.reduce_mean(weights_all * self.actor.entropy(s_all))
        kl_info_ref = self.actor.get_kl_info(s_all)
        actor_weights_pik = self.actor.get_weights()

        neglogp_pik_all = self.actor.neglogp_pik(s_all,a_all)
        offpol_ratio = tf.exp(neglogp_old_all - neglogp_pik_all)

        adv_mean = (np.mean(offpol_ratio * weights_all * adv_all) / 
            np.mean(offpol_ratio * weights_all))
        adv_std = np.std(offpol_ratio * weights_all * adv_all) + 1e-8

        if self.adv_center:
            adv_all = adv_all - adv_mean 
        if self.adv_scale:
            adv_all = adv_all / adv_std
        if self.adv_clip:
            adv_all = np.clip(adv_all,-self.adv_clip,self.adv_clip)

        neglogp_cur_all = self.actor.neglogp(s_all,a_all)
        ratio = tf.exp(neglogp_old_all - neglogp_cur_all)
        surr_before = tf.reduce_mean(ratio * adv_all * weights_all)

        # Update
        self.actor.set_weights(eta_v_flat,from_flat=True,increment=True)
                
        neglogp_cur_all = self.actor.neglogp(s_all,a_all)
        ratio = tf.exp(neglogp_old_all - neglogp_cur_all)
        surr = tf.reduce_mean(ratio * adv_all * weights_all)
        improve = surr - surr_before

        kl = tf.reduce_mean(weights_all*self.actor.kl(s_all,kl_info_ref))
        pen_kl = tf.reduce_mean(weights_all*self.actor.kl(s_all,kl_info_all))
        kl_pre = kl.numpy()
        pen_kl_pre = pen_kl.numpy()

        ratio_diff = tf.abs(ratio - offpol_ratio)
        tv = 0.5 * tf.reduce_mean(weights_all * ratio_diff)
        pen = 0.5 * tf.reduce_mean(weights_all * tf.abs(ratio-1.))
        tv_pre = tv.numpy()
        pen_pre = pen.numpy()

        kl_reverse = tf.reduce_mean(weights_all * self.actor.kl(
            s_all,kl_info_ref,direction='reverse'))
        pen_kl_reverse = tf.reduce_mean(weights_all * self.actor.kl(
            s_all,kl_info_all,direction='reverse'))
        kl_reverse_pre = kl_reverse.numpy()
        pen_kl_reverse_pre = pen_kl_reverse.numpy()

        adj = 1
        for _ in range(10):
            if kl > (self.kl_maxfactor * self.delta):
                pass
            elif improve < 0:
                pass
            else:
                break
            
            # Scale policy update
            factor = np.sqrt(2)
            adj = adj / factor
            eta_v_flat = eta_v_flat / factor

            self.actor.set_weights(actor_weights_pik)
            self.actor.set_weights(eta_v_flat,from_flat=True,increment=True)
            
            neglogp_cur_all = self.actor.neglogp(s_all,a_all)
            ratio = tf.exp(neglogp_old_all - neglogp_cur_all)
            surr = tf.reduce_mean(ratio * adv_all * weights_all)
            improve = surr - surr_before

            kl = tf.reduce_mean(weights_all*self.actor.kl(s_all,kl_info_ref))
        else:
            # No policy update
            adj = 0
            self.actor.set_weights(actor_weights_pik)

            neglogp_cur_all = self.actor.neglogp(s_all,a_all)
            ratio = tf.exp(neglogp_old_all - neglogp_cur_all)
            surr = tf.reduce_mean(ratio * adv_all * weights_all)
            improve = surr - surr_before

            kl = tf.reduce_mean(weights_all*self.actor.kl(s_all,kl_info_ref))
        
        pen_kl = tf.reduce_mean(weights_all*self.actor.kl(s_all,kl_info_all))
            
        ratio_diff = tf.abs(ratio - offpol_ratio)
        tv = 0.5 * tf.reduce_mean(weights_all * ratio_diff)
        pen = 0.5 * tf.reduce_mean(weights_all * tf.abs(ratio-1.))
        
        kl_reverse = tf.reduce_mean(weights_all * self.actor.kl(
            s_all,kl_info_ref,direction='reverse'))
        pen_kl_reverse = tf.reduce_mean(weights_all * self.actor.kl(
            s_all,kl_info_all,direction='reverse'))            
        
        log_actor = {
            'ent':                  ent.numpy(),
            'tv_pre':               tv_pre,
            'kl_pre':               kl_pre,
            'kl_reverse_pre':       kl_reverse_pre,
            'pen_pre':              pen_pre,
            'pen_kl_pre':           pen_kl_pre,
            'pen_kl_reverse_pre':   pen_kl_reverse_pre,
            'tv':                   tv.numpy(),
            'kl':                   kl.numpy(),
            'kl_reverse':           kl_reverse.numpy(),
            'penalty':              pen.numpy(),
            'penalty_kl':           pen_kl.numpy(),
            'penalty_kl_reverse':   pen_kl_reverse.numpy(),
            'adj':                  adj,
            'improve':              improve.numpy(),
            'eps':                  self.eps,
            'delta':                self.delta
        }
        self.logger.log_train(log_actor)

        self.actor.update_pik_weights()