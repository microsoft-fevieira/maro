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

    Reference: https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch.

    Args:
        model (SimpleMultiHeadModel): Model that computes action distributions.
        reward_discount (float): Reward decay as defined in standard RL terminology.
    """
    def __init__(self, model: SimpleMultiHeadModel, reward_discount: float, lam):
        self.lam = lam
        self.eps = 1e-6
        super().__init__(model, reward_discount)

    def choose_action(self, state) -> Union[int, np.ndarray]:
        """Use the actor (policy) model to generate stochastic actions.

        Args:
            state: Input to the actor model.

        Returns:
            Actions and corresponding log probabilities.
        """
        self.learn_count = 0
        num_actions = len(state)

        if num_actions == 1:
            action, prob = (0, 1)
            return state[action][0], prob


        model_estimates = self._apply_model(state, training=False)
        num_actions = len(state)
        action_prob = Categorical(model_estimates)


        action = action_prob.sample()
        prob = model_estimates[action]
        action, prob = action.cpu().numpy(), prob.cpu().numpy()
        return state[action][0], prob

    def set_exploration_params(self, epsilon):
        pass



    def learn(self, states, orig_actions: np.ndarray, rewards: np.ndarray, next_states):
        batch_size = len(rewards)
        rewards = torch.from_numpy(rewards).to(self.device)
        actions = torch.from_numpy(orig_actions[:,0].astype(np.int64)).to(self.device) # extracts actions taken
        old_probabilities = torch.from_numpy(orig_actions[:,1]).to(self.device).detach()
                # from the tuple containing (action, log_probability) pairs
        
        # initiating tensor w/ -infinity for padding, then updating subcomponents
        mod = torch.tensor(np.ones(batch_size))
        entropy = torch.tensor(np.ones(batch_size))
        index = 0
        for state in states:
            if len(state) == 1:
                mod[index] = 1
                entropy[index] = 0
            else:
                action_probs = self._apply_model(state, training=True)
                mod[index] = action_probs[actions[index]]
                entropy[index] = Categorical(probs = action_probs).entropy()
            index += 1
        loss = -(torch.div(mod, old_probabilities+self.eps)*torch.log(mod+self.eps)*rewards + self.lam*entropy) # calculating final loss
        loss[torch.isnan(loss)] = 0
            # add an entropy regularizer of the softmax (could be good for exploration)
        self.model.step(loss.mean()) # taking gradient step
        self.learn_count += 1
        return loss.detach().numpy()


    def _apply_model(self, state, training = False):
        num_actions = len(state)
        batched_state = np.stack([state_features for _, state_features in state])
        tensor_state = torch.from_numpy(batched_state.astype(np.float32)).to(self.device)
        model_estimates = self.model(tensor_state, training = training)
        model_estimates_softmax = torch.transpose(torch.softmax(model_estimates, dim=0),0,1).squeeze()
        return model_estimates_softmax

    def _batched_apply_model(self, states, training = False):
        batched_state = np.stack([state_features for state in states for _, state_features in state])
        tensor_state = torch.from_numpy(batched_state.astype(np.float32)).to(self.device)
        model_estimates = self.model(tensor_state, training=training)
        return model_estimates.squeeze(1)