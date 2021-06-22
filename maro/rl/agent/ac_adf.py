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

    References:
    https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch.
    https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f

    Args:
        model (SimpleMultiHeadModel): Multi-task model that computes action distributions and state values.
            It may or may not have a shared bottom stack.
        config: Configuration for the AC algorithm.
    """
    def __init__(self, model: SimpleMultiHeadModel, config: ActorCriticConfig):
        if model.task_names is None or set(model.task_names) != {"actor", "critic"}:
            raise UnrecognizedTask(f"Expected model task names 'actor' and 'critic', but got {model.task_names}")
        super().__init__(model, config)
    def set_exploration_params(self, epsilon):
        pass


    def choose_action(self, state) -> Union[int, np.ndarray]:
        """Use the actor (policy) model to generate stochastic actions.

        Args:
            state: Input to the actor model.

        Returns:
            Actions and corresponding log probabilities.
        """
        model_estimates = self._apply_model(state, task_name = "actor", training=False)
        num_actions = len(state)
        action_prob = Categorical(model_estimates)


        action = action_prob.sample()
        log_p = action_prob.log_prob(action)
        if num_actions > 1:
            action, log_p = action.cpu().numpy()[0], log_p.cpu().numpy()[0]
        else:
            action, log_p = action.cpu().numpy(), log_p.cpu().numpy()
        return (action, log_p)



    def learn(self, states, actions: np.ndarray, rewards: np.ndarray, next_states):
        batch_size = len(rewards)
        rewards = torch.from_numpy(rewards).to(self.device)
        actions = torch.from_numpy(actions[:,0].astype(np.int64)).to(self.device) # extracts actions taken
                # from the tuple containing (action, log_probability) pairs
        # model_estimates = self._batched_apply_model(states, training=True)

        # model_estimates = self._batched_apply_mode
        state_values = self._get_state_values(states, training=False).detach()
        return_est = get_lambda_returns(
            rewards, state_values, self.config.reward_discount, self.config.lam, k=self.config.k
        )
        advantages = return_est - state_values

        log_p = torch.log(self._get_softmax_prob(states, training=True).gather(1, actions.unsqueeze(1)).squeeze())

        for i in range(self.config.train_iters):
            # actor loss
            log_p_new = torch.log(self._get_softmax_prob(states, training=True).gather(1, actions.unsqueeze(1)).squeeze())
            if self.config.clip_ratio is not None:
                ratio = torch.exp(log_p_new - log_p)
                clip_ratio = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio)
                actor_loss = -(torch.min(ratio * advantages, clip_ratio * advantages)).mean()
            else:
                actor_loss = -(log_p_new * advantages).mean()

            # critic_loss
            state_values = self._get_state_values(states, training=True)
            critic_loss = self.config.critic_loss_func(state_values, return_est)
            loss = critic_loss + self.config.actor_loss_coefficient * actor_loss
            self.model.step(loss)

    def _apply_model(self, state, task_name = "actor", training = False):
        num_actions = len(state)
        batched_state = np.stack([state_features for _, state_features in state])
        tensor_state = torch.from_numpy(batched_state.astype(np.float32)).to(self.device)
        model_estimates = self.model(tensor_state, task_name = task_name, training = training)
        model_estimates = torch.transpose(torch.div(torch.exp(model_estimates), torch.sum(torch.exp(model_estimates))),0,1)
        return model_estimates.squeeze(1)

    def _batched_apply_model(self, states, task_name = "actor", training = False):
        batched_state = np.stack([state_features for state in states for _, state_features in state])
        tensor_state = torch.from_numpy(batched_state.astype(np.float32)).to(self.device)
        model_estimates = self.model(tensor_state, task_name = task_name, training=training)
        return model_estimates.squeeze(1)


    def _get_softmax_prob(self, states, training=False):
        batch_size = len(states)
        model_estimates = self._batched_apply_model(states, task_name = "actor", training=training)
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
        return action_probs

    def _get_state_values(self, states, training=False):
        batch_size = len(states)
        model_estimates = self._batched_apply_model(states, task_name = "actor", training=training).detach()
        state_action_estimates = self._batched_apply_model(states, task_name="critic", training=training)
        # apply the model to get the last-layer for all (s,a) pairs in the batch
        
        mod = torch.empty(batch_size)
        start_idx = 0
        index = 0
        for state in states:
            end_idx = start_idx + len(state)
            mod[index] = torch.dot(torch.softmax(model_estimates[start_idx:end_idx], dim=0), state_action_estimates[start_idx:end_idx])
            index += 1
            start_idx = end_idx
        return mod

       