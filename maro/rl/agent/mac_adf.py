# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Callable, Tuple, Union

import numpy as np
import torch
from torch.distributions import Categorical
from torch.nn import MSELoss

from maro.rl.model import SimpleMultiHeadModel
from maro.rl.utils import get_lambda_returns, get_log_prob, get_truncated_cumulative_reward
from maro.utils.exception.rl_toolkit_exception import UnrecognizedTask
from maro.rl.agent.ac import ActorCriticConfig

from maro.rl.agent.abs_agent import AbsAgent


class MeanActorCritic_ADF(AbsAgent):
    """Mean Actor Critic algorithm with separate policy and value models, and action dependent features.
    This algorithm assumes that the 'reward' passed in the learn step takes into account the k_step returns.

    References:
    https://arxiv.org/pdf/1709.00503.pdf

    Args:
        model (SimpleMultiHeadModel): Multi-task model that computes action distributions and state values.
            It may or may not have a shared bottom stack.
        config: Configuration for the AC algorithm
        lam: parameter for entropy regularization  
    """
    def __init__(self, model: SimpleMultiHeadModel, config: ActorCriticConfig, lam: float = 0.0):
        self.lam = lam
        self.eps = 1e-6
        if model.task_names is None or set(model.task_names) != {"actor", "critic"}:
            raise UnrecognizedTask(f"Expected model task names 'actor' and 'critic', but got {model.task_names}")
        super().__init__(model, config)


    # No exploration used by default
    def set_exploration_params(self, epsilon):
        pass


    def choose_action(self, state) -> Union[int, np.ndarray]:
        """Use the actor (policy) model to generate stochastic actions.  NOTE: Different from MARO in returns probabilities
        instead of log probabilities.

        Args:
            state: Input to the actor model.

        Returns:
            Actions and corresponding probabilities.
        """

        num_actions = len(state)

        if num_actions == 1: # If only a single action then just output that action with probability one.
            action, prob = (0, 1)
            return state[action][0], prob

        model_estimates = self._apply_model(state, task_name = "actor", training=False) # apply model to get probabilities
        action_prob = Categorical(model_estimates) # set up distribution and sample
        action = action_prob.sample()
        prob = model_estimates[action]
        action, prob = action.cpu().numpy(), prob.cpu().numpy()
        return state[action][0], prob # return selected action and its probability

    def learn(self, states, orig_actions: np.ndarray, rewards: np.ndarray, next_states):

        batch_size = len(rewards)
        
        return_est = torch.from_numpy(rewards).to(self.device)
            # Note assumes the rewards passed from the trajectory already take into account cumulative discounted
            # reward vs one step

        actions = torch.from_numpy(orig_actions[:,0].astype(np.int64)).to(self.device) # extracts actions taken
        old_probabilities = torch.from_numpy(orig_actions[:,1]).to(self.device)


        # Calculating critic loss
        q_estimates = self._get_selected_q_values(states, actions, training=True)
        critic_loss = self.config.critic_loss_func(q_estimates, return_est)


        # Calculating actor loss
        mod = torch.empty(batch_size)
        index = 0

        for state in states:
            if len(state) > 1: # Takes dot product between probability and q values for each state
                mod[index] = torch.dot(self._apply_model(state, task_name="actor", training=True), 
                            self._get_q_values(state, task_name="critic", training=False).detach())
            else:
                mod[index] = self._apply_model(state, task_name="actor", training=True) * self._get_q_values(state, task_name="critic", training=False).detach()
            index += 1
        actor_loss = mod.mean()
        batch_loss = torch.abs(q_estimates - return_est) + self.config.actor_loss_coefficient * mod
        loss = critic_loss + self.config.actor_loss_coefficient * actor_loss
        self.model.step(loss) # takes a step w/r/t the loss

        return batch_loss.detach()



    def _apply_model(self, state, task_name = "actor", training = False): # applies model to get probabilities via softmax
        num_actions = len(state)
        batched_state = np.stack([state_features for _, state_features in state])
        tensor_state = torch.from_numpy(batched_state.astype(np.float32)).to(self.device)
        model_estimates = self.model(tensor_state, task_name = task_name, training = training)
        model_estimates = torch.transpose(torch.softmax(model_estimates, dim=0),0,1)
        return model_estimates.squeeze()

    def _get_q_values(self, state, task_name = 'critic', training=False): # gets the q values from the critic
        num_actions = len(state)
        batched_state = np.stack([state_features for _, state_features in state])
        tensor_state = torch.from_numpy(batched_state.astype(np.float32)).to(self.device)
        return self.model(tensor_state, task_name = task_name, training=training).squeeze()

    def _batched_apply_model(self, states, task_name = 'critic', training = False): # gets the batched q value from the critic
        batched_state = np.stack([state_features for state in states for _, state_features in state])
        tensor_state = torch.from_numpy(batched_state.astype(np.float32)).to(self.device)
        model_estimates = self.model(tensor_state, task_name = task_name, training=training)
        return model_estimates.squeeze(1)

    def _get_selected_q_values(self, states, actions, training=False):
        batch_size = len(states)
        state_action_estimates = self._batched_apply_model(states, task_name="critic", training=training)
        # apply the model to get the last-layer for all (s,a) pairs in the batch

        state_offsets = np.cumsum([0] + [len(state) for state in states[:-1]])
        # calculates offset actions to deal with stacking, and return estimates of selected actions

        return state_action_estimates[torch.from_numpy(state_offsets).to(self.device) + actions]       

       