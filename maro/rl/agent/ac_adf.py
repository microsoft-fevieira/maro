# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Callable, Tuple, Union

import numpy as np
import torch
from torch.distributions import Categorical
from torch.nn import MSELoss

from maro.rl.model import SimpleMultiHeadModel
from maro.rl.utils import get_lambda_returns, get_log_prob
from maro.utils.exception.rl_toolkit_exception import UnrecognizedTask
from .ac import ActorCriticConfig

from .abs_agent import AbsAgent


class ActorCritic_ADF(AbsAgent):
    """Actor Critic algorithm with separate policy and value models.
    This algorithm is set-up for action dependent features, and the offline setting (handling distribution mismatch via importance weights)
    and assumes that the 'reward' passed in the learn step takes into account the k_step returns.

    References:
    https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch.
    https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f

    Args:
        model (SimpleMultiHeadModel): Multi-task model that computes action distributions and state values.
            It may or may not have a shared bottom stack.
        config: Configuration for the AC algorithm.
    """
    def __init__(self, model: SimpleMultiHeadModel, config: ActorCriticConfig, lam: float = 0.0):
        self.lam = lam
        self.eps = 1e-6
        if model.task_names is None or set(model.task_names) != {"actor", "critic"}:
            raise UnrecognizedTask(f"Expected model task names 'actor' and 'critic', but got {model.task_names}")
        super().__init__(model, config)


    def set_exploration_params(self, epsilon):
        pass


    def choose_action(self, state) -> Union[int, np.ndarray]:
        """Use the actor (policy) model to generate stochastic actions.  Note, different from MARO set-up as returns probablity
        instead of log probability.

        Args:
            state: Input to the actor model.

        Returns:
            Actions and corresponding probabilities.
        """

        num_actions = len(state)

        if num_actions == 1:
            action, prob = (0, 1)
            return state[action][0], prob


        model_estimates = self._apply_model(state, task_name = "actor", training=False)
        num_actions = len(state)
        action_prob = Categorical(model_estimates)
        action = action_prob.sample()
        prob = model_estimates[action]
        action, prob = action.cpu().numpy(), prob.cpu().numpy()
        return state[action][0], prob

    def learn(self, states, orig_actions: np.ndarray, rewards: np.ndarray, next_states):
        batch_size = len(rewards)

        truncate_return_est = torch.from_numpy(rewards[:, 0].astype(np.float32)).to(self.device)

        discount_ticks = torch.from_numpy(rewards[:, 1].astype(np.float32)).to(self.device)
        actual_discount = self.config.reward_discount**(1 + discount_ticks)
        final_state = rewards[:, 2]


        return_est = (truncate_return_est +  actual_discount * self._get_state_values(final_state)).detach()


        actions = torch.from_numpy(orig_actions[:,0].astype(np.int64)).to(self.device) # extracts actions taken
        old_probabilities = torch.from_numpy(orig_actions[:,1]).to(self.device)

        # Calculating critic loss
        state_values = self._get_state_values(states, training=True)
        critic_loss = self.config.critic_loss_func(state_values, return_est)

        

        # Calculates advantages
        advantages = return_est - state_values


        log_p = torch.log(old_probabilities + self.eps)
        # actor loss
        new_prob, entropy = self._get_softmax_prob(states, actions, training=True) # Gets the new
                # probabilities for the actions taken for each state in the batch
        log_p_new = torch.log(new_prob+self.eps) # gets the new log probability
        if self.config.clip_ratio is not None:
            ratio = torch.exp(log_p_new - log_p)
            clip_ratio = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio)
            actor_loss = -(torch.div(new_prob, old_probabilities)*torch.min(ratio * advantages, clip_ratio * advantages) + self.lam * entropy)
        else:
            actor_loss = -(torch.div(new_prob, old_probabilities)*log_p_new * advantages + self.lam * entropy)

        loss = critic_loss # + self.config.actor_loss_coefficient * actor_loss
        self.model.step(loss.mean())
        return loss.detach().cpu().numpy()


    # Softmax probabilities used for the pi^\theta(a | s) values
    # single state
    def _apply_model(self, state, task_name = "actor", training = False): # applies model to get probabilities via softmax
        num_actions = len(state)
        batched_state = np.stack([state_features for _, state_features in state])
        tensor_state = torch.from_numpy(batched_state.astype(np.float32)).to(self.device)
        model_estimates = self.model(tensor_state, task_name = task_name, training = training)
        model_estimates = torch.transpose(torch.softmax(model_estimates, dim=0),0,1) # apply final softmax layer to get probabilities
        return model_estimates.squeeze()

    # Gets Q(s,a) values for each available action in a single state s from the critic network
    def _get_q_values(self, state, task_name = 'critic', training=False): # gets the q values from the critic
        num_actions = len(state)
        batched_state = np.stack([state_features for _, state_features in state])
        tensor_state = torch.from_numpy(batched_state.astype(np.float32)).to(self.device)
        return self.model(tensor_state, task_name = task_name, training=training).squeeze()

    # Q(s,a) for multiple states, unfolded into a very long vector
    def _batched_apply_model(self, states, task_name = 'critic', training = False): # gets the batched q value from the critic
        batched_state = np.stack([state_features for state in states for _, state_features in state])
        tensor_state = torch.from_numpy(batched_state.astype(np.float32)).to(self.device)
        model_estimates = self.model(tensor_state, task_name = task_name, training=training)
        return model_estimates.squeeze(1)

    # Gets Q(s, a_selected) for multiple states in the batch
    def _get_selected_q_values(self, states, actions, training=False):
        batch_size = len(states)
        state_action_estimates = self._batched_apply_model(states, task_name="critic", training=training)
        # apply the model to get the last-layer for all (s,a) pairs in the batch

        state_offsets = np.cumsum([0] + [len(state) for state in states[:-1]])
        # calculates offset actions to deal with stacking, and return estimates of selected actions

        return state_action_estimates[torch.from_numpy(state_offsets).to(self.device) + actions]       

    # Returns a vector of V^\pi(s) for each state in states via <Q^\pi(s,a), \pi(s,a)>
    def _get_state_values(self, states, training=False):
        # Uses the critic network to get Q^\pi(s,a) values and uses the actor to get \pi(a | s) values
        # and uses these in order to get an estimate of V^\pi(s).
        batch_size = len(states)
        mod = torch.empty(batch_size)
        index = 0
        for state in states:
            if len(state) > 1:
                mod[index] = torch.dot(self._apply_model(state, task_name="actor", training=False).detach(), self._get_q_values(state, task_name='critic', training=training))
            else:
                mod[index] = self._apply_model(state, task_name="actor", training=False).detach() * self._get_q_values(state, task_name='critic', training=training)
            index += 1
        return mod

    def _get_softmax_prob(self, states, actions, training=False): # Gets the softmax probabilites for all (s,a) pairs in the batch
        batch_size = len(states)

        # initiating tensor w/ -infinity for padding, then updating subcomponents
        mod = torch.empty(batch_size)
        entropy = torch.empty(batch_size)

        index = 0
        for state in states:
            if len(state) == 1:
                mod[index] = 1 # then the log probability and entropy calculations are off, so set it to be 1 so log is zero
                entropy[index] = 0
            else:
                action_probs = self._apply_model(state, task_name="actor", training=True) # get current action probs
                mod[index] = action_probs[actions[index]] # fill in probability of selected action
                entropy[index] = Categorical(probs = action_probs).entropy() # calculate entropy
            index += 1
        return mod, entropy
