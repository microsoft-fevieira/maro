# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Union

import numpy as np
import torch

from maro.rl.models.learning_model import LearningModel

from .abs_algorithm import AbsAlgorithm
from .dqn import DQNConfig


class DQN_ADF(AbsAlgorithm):
    """DQN with action-dependent features
    
    Args:
        model (LearningModel): Q-value model that takes a state-action pair and returns a number.
        config: Configuration for DQN algorithm.
    """
    def __init__(self, model: LearningModel, config: DQNConfig):
        self.validate_task_names(model.task_names, {"state_value", "advantage"})
        super().__init__(model, config)
        self._training_counter = 0
        self._target_model = model.copy() if model.is_trainable else None

    def _apply_model(self, model, state):
        num_actions = len(state)
        q_estimates = np.zeros(num_actions)
        for i in range(num_actions):
            action_id, state_features = state[i]
            state_feat = torch.from_numpy(state_features.astype(np.float32)).to(self._device)
            q_estimates[i] = self._get_q_values(model, state_feat, is_training=False)
        return q_estimates
    
    def choose_action(self, state) -> Union[int, np.ndarray]:
        q_estimates = self._apply_model(self._model, state)
        num_actions = np.shape(q_estimates)[0]
        greedy_action = q_estimates.argmax()
        # No exploration
        if self._config.epsilon == .0:
            return state[greedy_action][0]

        return state[greedy_action][0] if np.random.random() > self._config.epsilon else state[np.random.choice(num_actions)][0]

    def _get_q_values(self, model, state, is_training: bool = True):
        if self._config.advantage_mode is not None:
            output = model(state, is_training=is_training)
            state_values = output["state_value"]
            advantages = output["advantage"]
            # Use mean or max correction to address the identifiability issue
            corrections = advantages.mean(1) if self._config.advantage_mode == "mean" else advantages.max(1)[0]
            q_values = state_values + advantages - corrections.unsqueeze(1)
            return q_values
        else:
            return model(state, is_training=is_training)

    def _get_next_q_values(self, current_q_values_for_all_actions, next_states):
        next_q_values_for_all_actions = self._apply_model(self._target_model, next_states)
        if self._config.is_double:
            actions = current_q_values_for_all_actions.argmax()
            return next_q_values_for_all_actions[actions]
        else:
            return next_q_values_for_all_actions.max()

    def _compute_td_errors(self, states, actions, rewards, next_states):
        current_q_values_for_all_actions = self._apply_model(self._model, states)
        current_q_values = current_q_values_for_all_actions[actions]
        next_q_values = self._get_next_q_values(current_q_values_for_all_actions, next_states)  # (N,)
        target_q_values = (rewards + self._config.reward_discount * next_q_values).detach()  # (N,)
        return self._config.loss_func(current_q_values, target_q_values)

    def train(self, states, actions: np.ndarray, rewards: np.ndarray, next_states):
        #states = torch.from_numpy(states).to(self._device)
        #actions = torch.from_numpy(actions).to(self._device)
        #rewards = torch.from_numpy(rewards).to(self._device)
        #next_states = torch.from_numpy(next_states).to(self._device)
        loss = self._compute_td_errors(states, actions, rewards, next_states)
        self._model.learn(loss.mean() if self._config.per_sample_td_error_enabled else loss)
        self._training_counter += 1
        if self._training_counter % self._config.target_update_frequency == 0:
            self._target_model.soft_update(self._model, self._config.tau)

        return loss.detach().numpy()

    def set_exploration_params(self, epsilon):
        self._config.epsilon = epsilon
