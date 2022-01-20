# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Tuple, Union

import numpy as np
import torch
from torch.distributions import Categorical
from torch.nn import Softmax

from maro.rl.model import SimpleMultiHeadModel
from maro.rl.utils import get_truncated_cumulative_reward, get_log_prob

import time
from .abs_agent import AbsAgent


class PolicyGradient_ADF(AbsAgent):
    """The vanilla Policy Gradient (VPG) algorithm with action-dependent features, a.k.a., REINFORCE.
    This algorithm is set-up for action dependent features, and the offline setting (handling distribution mismatch via importance weights)
    and assumes that the 'reward' passed in the learn step takes into account the k_step returns.

    Reference: https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch.

    Args:
        model (SimpleMultiHeadModel): Model that computes action distributions.
        reward_discount (float): Reward decay as defined in standard RL terminology.
        lam (float): Parameter for entropy regularization, default is zero.
    """
    def __init__(self, model: SimpleMultiHeadModel, reward_discount: float, lam: float = 0.0):
        self.lam = lam
        self.eps = 1e-8
        super().__init__(model, reward_discount)

    def choose_action(self, state) -> Union[int, np.ndarray]:
        """Use the actor (policy) model to generate stochastic actions.

        Args:
            state: Input to the actor model.

        Returns:
            Actions and corresponding probabilities.  NOTE: Different from MARO PG algorithm as we return the probability
            instead of the log probability 
        """

        num_actions = len(state)
        if num_actions == 1: # Then select the first action guaranteed with probability 1
            action, prob = (0, 1)
            return state[action][0], prob


        model_estimates = self._apply_model(state, training=False) # applies the model to get the probabilities
        num_actions = len(state)
        action_prob = Categorical(model_estimates) # set up the categorical distribution with those probabilities


        action = action_prob.sample() # sample from the probability
        prob = model_estimates[action] # get the probability for action selected
        action, prob = action.cpu().numpy(), prob.cpu().numpy()
        return state[action][0], prob

    # Exploration parameters currently not included.
    def set_exploration_params(self, epsilon):
        pass



    def learn(self, states, orig_actions: np.ndarray, rewards: np.ndarray, next_states):
        """
            Performs a learn step using the REINFORCE algorithm.
            Note: Assumes that rewards instead contains some version of k-step returns
            modeling the return of the trajectory needed in the updates.
            Orig_actions is a 2-D numpy matrix where first component is action, second component 
            is probability that action was selected
        """

        batch_size = len(rewards)

        # Note: Different from MARO implementation as we are not using cumulative reward
        # but assuming that is done in the vm_trajectory
        # returns = get_truncated_cumulative_reward(rewards, self.config)


        returns = torch.from_numpy(rewards).to(self.device)
        actions = torch.from_numpy(orig_actions[:,0].astype(np.int64)).to(self.device) # extracts actions taken
        old_probabilities = torch.from_numpy(orig_actions[:,1]).to(self.device).detach()
                # from the tuple containing (action, log_probability) pairs
        
        
        # initiating tensor and updating subcomponent
        mod = torch.tensor(np.ones(batch_size))
        entropy = torch.tensor(np.ones(batch_size))
        index = 0
        
        for state in states:
            if len(state) == 1:
                mod[index] = 1 # then the log probability and entropy calculations are off, so set it to be 1 so log is zero
                entropy[index] = 0
            else:
                action_probs = self._apply_model(state, training=True) # get current action probs
                mod[index] = action_probs[actions[index]] # fill in probability of selected action
                entropy[index] = Categorical(probs = action_probs).entropy() # calculate entropy
            index += 1
        
        
        loss = -(torch.div(mod, old_probabilities)*torch.log(mod+self.eps)*returns + self.lam*entropy) # calculating final loss
        # add an entropy regularizer of the softmax (could be good for exploration)
        
        self.model.step(loss.mean()) # taking gradient step
        return loss.detach().cpu().numpy()


    def _apply_model(self, state, training = False): # Applies model to action dependent features
        # and takes final layer as softmax over the model
        num_actions = len(state)
        batched_state = np.stack([state_features for _, state_features in state])
        tensor_state = torch.from_numpy(batched_state.astype(np.float32)).to(self.device)
        model_estimates = self.model(tensor_state, training = training)
        model_estimates_softmax = torch.transpose(torch.softmax(model_estimates, dim=0),0,1).squeeze()
        return model_estimates_softmax