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
    def __init__(self, model: SimpleMultiHeadModel, reward_discount: float):
        super().__init__(model, reward_discount)

    def choose_action(self, state) -> Union[int, np.ndarray]:
        """Use the actor (policy) model to generate stochastic actions.

        Args:
            state: Input to the actor model.

        Returns:
            Actions and corresponding log probabilities.
        """


        model_estimates = self._apply_model(state, training=False)
        num_actions = len(state)
        action_prob = Categorical(model_estimates)


        action = action_prob.sample()
        log_p = action_prob.log_prob(action)
        if num_actions > 1:
            action, log_p = action.cpu().numpy()[0], log_p.cpu().numpy()[0]
        else:
            action, log_p = action.cpu().numpy(), log_p.cpu().numpy()
        return (action, log_p)

    def set_exploration_params(self, epsilon):
        pass


    def learn(self, states, actions: np.ndarray, rewards: np.ndarray, next_states):
        batch_size = len(rewards)
        rewards = torch.from_numpy(rewards).to(self.device)
        actions = torch.from_numpy(actions[:,0].astype(np.int64)).to(self.device) # extracts actions taken
                # from the tuple containing (action, log_probability) pairs
        model_estimates = self._batched_apply_model(states, training=True)
            # apply the model to get the last-layer for all (s,a) pairs in the batch
        
        # calculating max number of actions for padding
        max_action = 0
        for state in states:
            if len(state) > max_action:
                max_action = len(state)
        
        # initiating tensor w/ -infinity for padding, then updating subcomponents
        mod = torch.tensor((-1)*np.ones((batch_size, max_action)) * np.inf)
        start_idx = 0
        index = 0
        for state in states:
            end_idx = start_idx + len(state)
            mod[index, 0:len(state)] = model_estimates[start_idx:end_idx]
            index += 1
            start_idx = end_idx
        action_probs = torch.softmax(mod, dim=1) # taking softmax across each state
        action_probs = action_probs.gather(1, actions.unsqueeze(1)).squeeze() # evaluating at sampled actions
        loss = -(torch.log(action_probs)*rewards) # calculating final loss
        self.model.step(loss.mean()) # taking gradient step
        return loss.detach().numpy()

    def _apply_model(self, state, training = False):
        num_actions = len(state)
        batched_state = np.stack([state_features for _, state_features in state])
        tensor_state = torch.from_numpy(batched_state.astype(np.float32)).to(self.device)
        model_estimates = self.model(tensor_state, training = training)
        model_estimates = torch.transpose(torch.div(torch.exp(model_estimates), torch.sum(torch.exp(model_estimates))),0,1)
        return model_estimates.squeeze(1)

    def _batched_apply_model(self, states, training = False):
        batched_state = np.stack([state_features for state in states for _, state_features in state])
        tensor_state = torch.from_numpy(batched_state.astype(np.float32)).to(self.device)
        model_estimates = self.model(tensor_state, training=training)
        return model_estimates.squeeze(1)

