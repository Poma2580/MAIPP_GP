import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SequentialSampler
import os


# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain: float = 1.0):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=float(gain))  # type: ignore[arg-type]


class Actor_RNN(nn.Module):
    def __init__(self, args, actor_input_dim):
        super(Actor_RNN, self).__init__()
        self.rnn_hidden = None
        self.hidden_dim = args.rnn_hidden_dim

        self.fc1 = nn.Linear(actor_input_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.action_dim)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.rnn)
            orthogonal_init(self.fc2, gain=0.01)

    def forward(self, actor_input):
        # When 'choose_action': actor_input.shape=(N, actor_input_dim), prob.shape=(N, action_dim)
        # When 'train':         actor_input.shape=(mini_batch_size*N, actor_input_dim),prob.shape=(mini_batch_size*N, action_dim)
        x = self.activate_func(self.fc1(actor_input))
        # initialize hidden on correct device/shape if needed
        if self.rnn_hidden is None or self.rnn_hidden.size(0) != x.size(0) or self.rnn_hidden.device != x.device:
            self.rnn_hidden = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
        self.rnn_hidden = self.rnn(x, self.rnn_hidden)
        prob = torch.softmax(self.fc2(self.rnn_hidden), dim=-1)
        return prob


class Critic_RNN(nn.Module):
    def __init__(self, args, critic_input_dim):
        super(Critic_RNN, self).__init__()
        self.rnn_hidden = None
        self.hidden_dim = args.rnn_hidden_dim

        self.fc1 = nn.Linear(critic_input_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, 1)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.rnn)
            orthogonal_init(self.fc2)

    def forward(self, critic_input):
        # When 'get_value': critic_input.shape=(N, critic_input_dim), value.shape=(N, 1)
        # When 'train':     critic_input.shape=(mini_batch_size*N, critic_input_dim), value.shape=(mini_batch_size*N, 1)
        x = self.activate_func(self.fc1(critic_input))
        # initialize hidden on correct device/shape if needed
        if self.rnn_hidden is None or self.rnn_hidden.size(0) != x.size(0) or self.rnn_hidden.device != x.device:
            self.rnn_hidden = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
        self.rnn_hidden = self.rnn(x, self.rnn_hidden)
        value = self.fc2(self.rnn_hidden)
        return value


class Actor_MLP(nn.Module):
    def __init__(self, args, actor_input_dim):
        super(Actor_MLP, self).__init__()
        self.fc1 = nn.Linear(actor_input_dim, args.mlp_hidden_dim)
        self.fc2 = nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim)
        self.fc3 = nn.Linear(args.mlp_hidden_dim, args.action_dim)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3, gain=0.01)

    def forward(self, actor_input):
        # When 'choose_action': actor_input.shape=(N, actor_input_dim), prob.shape=(N, action_dim)
        # When 'train':         actor_input.shape=(mini_batch_size, episode_limit, N, actor_input_dim), prob.shape(mini_batch_size, episode_limit, N, action_dim)
        x = self.activate_func(self.fc1(actor_input))
        x = self.activate_func(self.fc2(x))
        prob = torch.softmax(self.fc3(x), dim=-1)
        return prob


class Critic_MLP(nn.Module):
    def __init__(self, args, critic_input_dim):
        super(Critic_MLP, self).__init__()
        self.fc1 = nn.Linear(critic_input_dim, args.mlp_hidden_dim)
        self.fc2 = nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim)
        self.fc3 = nn.Linear(args.mlp_hidden_dim, 1)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, critic_input):
        # When 'get_value': critic_input.shape=(N, critic_input_dim), value.shape=(N, 1)
        # When 'train':     critic_input.shape=(mini_batch_size, episode_limit, N, critic_input_dim), value.shape=(mini_batch_size, episode_limit, N, 1)
        x = self.activate_func(self.fc1(critic_input))
        x = self.activate_func(self.fc2(x))
        value = self.fc3(x)
        return value
    
class QCritic_MLP(nn.Module):
    """
    Q Critic for COMA counterfactual baseline
    Input: [global_state, joint_action_onehot_flat]
    Output: Q(s, joint_a) - per-agent Q values
    """
    def __init__(self, args, state_dim, joint_action_dim):
        super(QCritic_MLP, self).__init__()
        input_dim = state_dim + joint_action_dim  # s + flattened joint action onehot
        hidden_dim = getattr(args, 'coma_q_hidden_dim', 128)
        self.N = args.N
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, self.N)  # Output per-agent Q values
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]
        
        if args.use_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)
    
    def forward(self, state, joint_action_onehot):
        """
        Args:
            state: (B, state_dim) or (B, T, state_dim)
            joint_action_onehot: (B, N*action_dim) or (B, T, N*action_dim)
        Returns:
            q_values: (B, N) or (B, T, N) per-agent Q values
        """
        x = torch.cat([state, joint_action_onehot], dim=-1)
        x = self.activate_func(self.fc1(x))
        x = self.activate_func(self.fc2(x))
        q_values = self.fc3(x)
        return q_values

class MAPPO_MPE:
    def __init__(self, args):
        self.N = args.N
        self.action_dim = args.action_dim
        self.obs_dim = args.obs_dim
        self.state_dim = args.state_dim
        self.episode_limit = args.episode_limit
        self.rnn_hidden_dim = args.rnn_hidden_dim
        # device
        self.device = args.device if isinstance(args.device, torch.device) else torch.device(str(args.device))

        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr = args.lr
        self.gamma = args.gamma
        self.lamda = args.lamda
        self.epsilon = args.epsilon
        self.K_epochs = args.K_epochs
        self.entropy_coef = args.entropy_coef
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm
        self.use_rnn = args.use_rnn
        self.add_agent_id = args.add_agent_id
        self.use_value_clip = args.use_value_clip
        # Whether each agent has its own actor network (instead of sharing one actor across agents)
        self.per_agent_actor = bool(getattr(args, "per_agent_actor", False))

        # COMA 相关参数
        self.use_coma_shaping = getattr(args, 'use_coma_shaping', False)
        self.coma_beta = getattr(args, 'coma_beta', 0.1)
        self.coma_clip = getattr(args, 'coma_clip', 5.0)

        # get the input dimension of actor and critic
        self.actor_input_dim = args.obs_dim
        self.critic_input_dim = args.state_dim
        if self.add_agent_id:
            print("------add agent id------")
            self.actor_input_dim += args.N
            self.critic_input_dim += args.N

        if self.use_rnn:
            print("------use rnn------")
            if self.per_agent_actor:
                self.actor = nn.ModuleList([Actor_RNN(args, self.actor_input_dim) for _ in range(self.N)])
            else:
                self.actor = Actor_RNN(args, self.actor_input_dim)
            self.critic = Critic_RNN(args, self.critic_input_dim)
        else:
            if self.per_agent_actor:
                self.actor = nn.ModuleList([Actor_MLP(args, self.actor_input_dim) for _ in range(self.N)])
            else:
                self.actor = Actor_MLP(args, self.actor_input_dim)
            self.critic = Critic_MLP(args, self.critic_input_dim)
        # move networks to device
        self.actor.to(self.device)
        self.critic.to(self.device)

        self.ac_parameters = list(self.actor.parameters()) + list(self.critic.parameters())
        # Optimizer (explicit import to avoid tooling complaints)
        from torch import optim as _optim
        if self.set_adam_eps:
            print("------set adam eps------")
            self.ac_optimizer = _optim.Adam(self.ac_parameters, lr=self.lr, eps=1e-5)  # type: ignore[attr-defined]
        else:
            self.ac_optimizer = _optim.Adam(self.ac_parameters, lr=self.lr)  # type: ignore[attr-defined]

        # Initialize Q Critic for COMA
        if self.use_coma_shaping:
            print("------use COMA reward shaping------")
            joint_action_dim = self.N * self.action_dim  # N agents * action_dim (onehot)
            self.q_critic = QCritic_MLP(args, self.state_dim, joint_action_dim)
            self.q_critic.to(self.device)
            
            # Separate optimizer for Q critic
            coma_q_lr = getattr(args, 'coma_q_lr', self.lr)
            if self.set_adam_eps:
                self.q_optimizer = _optim.Adam(self.q_critic.parameters(), lr=coma_q_lr, eps=1e-5)
            else:
                self.q_optimizer = _optim.Adam(self.q_critic.parameters(), lr=coma_q_lr)

    def reset_rnn_hidden(self):
        """Reset RNN hidden states (supports both shared-actor and per-agent-actor)."""
        if not self.use_rnn:
            return
        if self.per_agent_actor:
            for a in self.actor:  # type: ignore[union-attr]
                a.rnn_hidden = None  # type: ignore[assignment]
        else:
            self.actor.rnn_hidden = None  # type: ignore[assignment]
        self.critic.rnn_hidden = None  # type: ignore[assignment]

    def choose_action(self, obs_n, evaluate):
        with torch.no_grad():
            actor_inputs = []
            # obs_n 通常ে: list[np.ndarray] 会触发非常慢的构造路径；先转成单个 ndarray
            if not isinstance(obs_n, torch.Tensor):
                import numpy as _np
                obs_n = _np.asarray(obs_n, dtype=_np.float32)
                obs_n = torch.from_numpy(obs_n)
            obs_n = obs_n.to(device=self.device, dtype=torch.float32)  # obs_n.shape=(N, obs_dim)
            if self.per_agent_actor:
                # Per-agent actor: compute prob for each agent independently
                probs = []
                for i in range(self.N):
                    if self.add_agent_id:
                        eye = torch.eye(self.N, device=self.device)
                        inp = torch.cat([obs_n[i], eye[i]], dim=-1)
                    else:
                        inp = obs_n[i]
                    prob_i = self.actor[i](inp.unsqueeze(0)).squeeze(0)  # type: ignore[index]
                    probs.append(prob_i)
                prob = torch.stack(probs, dim=0)  # (N, action_dim)
            else:
                actor_inputs.append(obs_n)
                if self.add_agent_id:
                    """
                        Add an one-hot vector to represent the agent_id
                        For example, if N=3
                        [obs of agent_1]+[1,0,0]
                        [obs of agent_2]+[0,1,0]
                        [obs of agent_3]+[0,0,1]
                        So, we need to concatenate a N*N unit matrix(torch.eye(N))
                    """
                    actor_inputs.append(torch.eye(self.N, device=self.device))

                actor_inputs = torch.cat([x for x in actor_inputs], dim=-1)  # actor_input.shape=(N, actor_input_dim)
                prob = self.actor(actor_inputs)  # prob.shape=(N,action_dim)
            if evaluate:  # When evaluating the policy, we select the action with the highest probability
                a_n = prob.argmax(dim=-1)
                return a_n.detach().cpu().numpy(), None
            else:
                dist = Categorical(probs=prob)
                a_n = dist.sample()
                a_logprob_n = dist.log_prob(a_n)
                return a_n.detach().cpu().numpy(), a_logprob_n.detach().cpu().numpy()

    def get_value(self, s):
        with torch.no_grad():
            critic_inputs = []
            # Because each agent has the same global state, we need to repeat the global state 'N' times.
            s = torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(self.N, 1)  # (state_dim,)-->(N,state_dim)
            critic_inputs.append(s)
            if self.add_agent_id:  # Add an one-hot vector to represent the agent_id
                critic_inputs.append(torch.eye(self.N, device=self.device))
            critic_inputs = torch.cat([x for x in critic_inputs], dim=-1)  # critic_input.shape=(N, critic_input_dim)
            v_n = self.critic(critic_inputs)  # v_n.shape(N,1)
            return v_n.detach().cpu().numpy().flatten()

    def train(self, replay_buffer, total_steps):
        batch = replay_buffer.get_training_data()  # get training data
        # move batch to device
        for k in batch:
            batch[k] = batch[k].to(self.device)

        # === COMA 奖励重塑 ===
        coma_stats = None
        if self.use_coma_shaping:
            batch['r_n'], coma_stats = self.reshape_rewards_with_coma(batch)

        # Calculate the advantage using GAE
        adv = []
        gae = 0
        with torch.no_grad():  # adv and td_target have no gradient
            deltas = batch['r_n'] + self.gamma * batch['v_n'][:, 1:] * (1 - batch['done_n']) - batch['v_n'][:, :-1]  # deltas.shape=(batch_size,episode_limit,N)
            for t in reversed(range(self.episode_limit)):
                gae = deltas[:, t] + self.gamma * self.lamda * gae
                adv.insert(0, gae)
            adv = torch.stack(adv, dim=1)  # adv.shape(batch_size,episode_limit,N)
            v_target = adv + batch['v_n'][:, :-1]  # v_target.shape(batch_size,episode_limit,N)
            if self.use_adv_norm:  # Trick 1: advantage normalization
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        """
            Get actor_inputs and critic_inputs
            actor_inputs.shape=(batch_size, max_episode_len, N, actor_input_dim)
            critic_inputs.shape=(batch_size, max_episode_len, N, critic_input_dim)
        """
        actor_inputs, critic_inputs = self.get_inputs(batch)

        # Collect batch-level loss stats (one value per train() call)
        actor_loss_sum = 0.0
        critic_loss_sum = 0.0
        update_count = 0
        q_loss_sum = 0.0
        q_update_count = 0

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            for index in BatchSampler(SequentialSampler(range(self.batch_size)), self.mini_batch_size, False):
                """
                    get probs_now and values_now
                    probs_now.shape=(mini_batch_size, episode_limit, N, action_dim)
                    values_now.shape=(mini_batch_size, episode_limit, N)
                """
                if self.use_rnn:
                    # If use RNN, we need to reset the rnn_hidden of the actor and critic.
                    self.reset_rnn_hidden()
                    probs_now, values_now = [], []
                    for t in range(self.episode_limit):
                        if self.per_agent_actor:
                            probs_t = []
                            for i in range(self.N):
                                inp_i = actor_inputs[index, t, i]  # (mini_batch_size, actor_input_dim)
                                prob_i = self.actor[i](inp_i)  # type: ignore[index]
                                probs_t.append(prob_i)
                            probs_t = torch.stack(probs_t, dim=1)  # (mini_batch_size, N, action_dim)
                            probs_now.append(probs_t)
                        else:
                            prob = self.actor(actor_inputs[index, t].reshape(self.mini_batch_size * self.N, -1)) # prob.shape=(mini_batch_size*N, action_dim)
                            probs_now.append(prob.reshape(self.mini_batch_size, self.N, -1))  # prob.shape=(mini_batch_size,N,action_dim）
                        v = self.critic(critic_inputs[index, t].reshape(self.mini_batch_size * self.N, -1))  # v.shape=(mini_batch_size*N,1)
                        values_now.append(v.reshape(self.mini_batch_size, self.N))  # v.shape=(mini_batch_size,N)
                    # Stack them according to the time (dim=1)
                    probs_now = torch.stack(probs_now, dim=1)
                    values_now = torch.stack(values_now, dim=1)
                else:
                    if self.per_agent_actor:
                        probs_agents = []
                        for i in range(self.N):
                            prob_i = self.actor[i](actor_inputs[index, :, i])  # type: ignore[index]
                            # prob_i: (mini_batch_size, episode_limit, action_dim)
                            probs_agents.append(prob_i)
                        probs_now = torch.stack(probs_agents, dim=2)  # (mini_batch_size, episode_limit, N, action_dim)
                    else:
                        probs_now = self.actor(actor_inputs[index])
                    values_now = self.critic(critic_inputs[index]).squeeze(-1)

                dist_now = Categorical(probs_now)
                dist_entropy = dist_now.entropy()  # dist_entropy.shape=(mini_batch_size, episode_limit, N)
                # batch['a_n'][index].shape=(mini_batch_size, episode_limit, N)
                a_logprob_n_now = dist_now.log_prob(batch['a_n'][index])  # a_logprob_n_now.shape=(mini_batch_size, episode_limit, N)
                # a/b=exp(log(a)-log(b))
                ratios = torch.exp(a_logprob_n_now - batch['a_logprob_n'][index].detach())  # ratios.shape=(mini_batch_size, episode_limit, N)
                surr1 = ratios * adv[index]
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy

                if self.use_value_clip:
                    values_old = batch["v_n"][index, :-1].detach()
                    values_error_clip = torch.clamp(values_now - values_old, -self.epsilon, self.epsilon) + values_old - v_target[index]
                    values_error_original = values_now - v_target[index]
                    critic_loss = torch.max(values_error_clip ** 2, values_error_original ** 2)
                else:
                    critic_loss = (values_now - v_target[index]) ** 2

                # batch-level statistics (mean over all update steps in this train() call)
                actor_loss_sum += float(actor_loss.mean().item())
                critic_loss_sum += float(critic_loss.mean().item())
                update_count += 1

                self.ac_optimizer.zero_grad()
                ac_loss = actor_loss.mean() + critic_loss.mean()
                ac_loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.ac_parameters, 10.0)
                self.ac_optimizer.step()

                # === 训练 Q Critic ===
                if self.use_coma_shaping:
                    q_loss_val = self.train_q_critic(batch, index)
                    if q_loss_val is not None:
                        q_loss_sum += float(q_loss_val)
                        q_update_count += 1

        if self.use_lr_decay:
            self.lr_decay(total_steps)
        
        # Return stats for logging (COMA + losses)
        train_stats = {} if coma_stats is None else dict(coma_stats)
        if update_count > 0:
            train_stats['actor_loss'] = actor_loss_sum / update_count
            train_stats['critic_loss'] = critic_loss_sum / update_count
        if q_update_count > 0:
            train_stats['q_loss'] = q_loss_sum / q_update_count
        return train_stats

    def lr_decay(self, total_steps):  # Trick 6: learning rate Decay
        lr_now = self.lr * (1 - total_steps / self.max_train_steps)
        for p in self.ac_optimizer.param_groups:
            p['lr'] = lr_now

    def get_inputs(self, batch):
        actor_inputs, critic_inputs = [], []
        actor_inputs.append(batch['obs_n'])
        critic_inputs.append(batch['s'].unsqueeze(2).repeat(1, 1, self.N, 1))
        if self.add_agent_id:
            # agent_id_one_hot.shape=(mini_batch_size, max_episode_len, N, N)
            agent_id_one_hot = torch.eye(self.N, device=self.device).unsqueeze(0).unsqueeze(0).repeat(self.batch_size, self.episode_limit, 1, 1)
            actor_inputs.append(agent_id_one_hot)
            critic_inputs.append(agent_id_one_hot)

        actor_inputs = torch.cat([x for x in actor_inputs], dim=-1)  # actor_inputs.shape=(batch_size, episode_limit, N, actor_input_dim)
        critic_inputs = torch.cat([x for x in critic_inputs], dim=-1)  # critic_inputs.shape=(batch_size, episode_limit, N, critic_input_dim)
        return actor_inputs, critic_inputs

    def save_model(self,total_steps, save_dir):
        """
        Save actor weights.
        - If per_agent_actor=False: save a single file: MAPPO_actor_step_{k}k.pth
        - If per_agent_actor=True:  save N files:      MAPPO_actor_agent_{i}_step_{k}k.pth
        """
        step_k = int(total_steps / 1000)
        if self.per_agent_actor:
            for i in range(self.N):
                save_path = os.path.join(save_dir, f"MAPPO_actor_agent_{i}_step_{step_k}k.pth")
                torch.save(self.actor[i].state_dict(), save_path)  # type: ignore[index]
        else:
            save_path = os.path.join(save_dir, f"MAPPO_actor_step_{step_k}k.pth")
            torch.save(self.actor.state_dict(), save_path)

    def load_model(self,total_steps, load_dir):
        """
        Load actor weights from directory.
        - If per_agent_actor=False: expects MAPPO_actor_step_{k}k.pth
        - If per_agent_actor=True:  expects MAPPO_actor_agent_{i}_step_{k}k.pth (for all i)
        """
        step_k = int(total_steps / 1000)
        if self.per_agent_actor:
            for i in range(self.N):
                load_path = os.path.join(load_dir, f"MAPPO_actor_agent_{i}_step_{step_k}k.pth")
                if not os.path.isfile(load_path):
                    raise FileNotFoundError(f"[Error] actor checkpoint not found: {load_path}")
                try:
                    state_dict = torch.load(load_path, map_location=self.device, weights_only=True)
                except TypeError:
                    state_dict = torch.load(load_path, map_location=self.device)
                self.actor[i].load_state_dict(state_dict)  # type: ignore[index]
        else:
            load_path = os.path.join(load_dir, f"MAPPO_actor_step_{step_k}k.pth")
            if not os.path.isfile(load_path):
                raise FileNotFoundError(f"[Error] actor checkpoint not found: {load_path}")
            try:
                state_dict = torch.load(load_path, map_location=self.device, weights_only=True)
            except TypeError:
                state_dict = torch.load(load_path, map_location=self.device)
            self.actor.load_state_dict(state_dict)

    def compute_counterfactual_advantages(self, batch, step_indices=None):
        """
        计算每个 agent 的反事实优势 A_i^cf
        
        A_i^cf(s, a) = Q_i(s, a) - sum_{a_i'} pi_i(a_i'|o_i) * Q_i(s, (a_i', a_{-i}))
        """
        with torch.no_grad():
            if step_indices is not None:
                s = batch['s'][step_indices]  # (mini_batch, T, state_dim)
                obs_n = batch['obs_n'][step_indices]  # (mini_batch, T, N, obs_dim)
                actions_onehot = batch['actions_onehot_n'][step_indices]  # (mini_batch, T, N, A)
            else:
                s = batch['s']  # (batch_size, T, state_dim)
                obs_n = batch['obs_n']
                actions_onehot = batch['actions_onehot_n']
            
            B, T, N, A = actions_onehot.shape
            
            # 确保 s 的维度正确 (B, T, state_dim)
            if s.dim() == 2:
                # 如果 s 是 (B, state_dim)，扩展到 (B, T, state_dim)
                s = s.unsqueeze(1).expand(B, self.episode_limit, -1)
            
            # 1. 计算当前 joint action 的 Q 值：Q_i(s, a)
            joint_action_flat = actions_onehot.reshape(B, T, N * A)  # (B, T, N*A)
            Q_current = self.q_critic(s, joint_action_flat)  # (B, T, N)
            
            # 2. 为每个 agent 计算反事实基线
            advantages_cf = torch.zeros(B, T, N, device=self.device)
            
            for i in range(N):
                # 获取 agent i 的策略分布 pi_i(a_i'|o_i)
                obs_i = obs_n[:, :, i, :]  # (B, T, obs_dim)
                
                if self.add_agent_id:
                    agent_id_onehot = torch.zeros(B, T, N, device=self.device)
                    agent_id_onehot[:, :, i] = 1.0
                    actor_input_i = torch.cat([obs_i, agent_id_onehot], dim=-1)
                else:
                    actor_input_i = obs_i
                
                # 获取策略概率分布
                if self.per_agent_actor:
                    actor_input_i_flat = actor_input_i.reshape(B * T, -1)
                    probs_i = self.actor[i](actor_input_i_flat)  # (B*T, A)
                    probs_i = probs_i.reshape(B, T, A)  # (B, T, A)
                else:
                    probs_i = self.actor(actor_input_i.reshape(B * T, -1))
                    probs_i = probs_i.reshape(B, T, A)
                
                # 3. 计算反事实基线：sum_{a_i'} pi_i(a_i') * Q_i(s, (a_i', a_{-i}))
                baseline = torch.zeros(B, T, device=self.device)
                
                # 批量化计算所有可能的 a_i'
                for a_prime in range(A):
                    # 构造反事实 joint action
                    counterfactual_actions_onehot = actions_onehot.clone()
                    counterfactual_actions_onehot[:, :, i, :] = 0.0
                    counterfactual_actions_onehot[:, :, i, a_prime] = 1.0
                    
                    # Flatten joint action
                    cf_joint_flat = counterfactual_actions_onehot.reshape(B, T, N * A)
                    
                    # 计算 Q_i(s, (a_i', a_{-i}))
                    Q_cf = self.q_critic(s, cf_joint_flat)[:, :, i]  # (B, T)
                    
                    # 加权求和
                    baseline += probs_i[:, :, a_prime] * Q_cf
                
                # 4. 计算优势：A_i^cf = Q_i(s, a) - baseline
                advantages_cf[:, :, i] = Q_current[:, :, i] - baseline
            
            return advantages_cf

    def reshape_rewards_with_coma(self, batch):
        """
        使用 COMA 反事实优势重塑奖励
        
        r_shaped_{i,t} = r_{i,t} + beta * clip(A_{i,t}^cf, -c, c)
        """
        if not self.use_coma_shaping:
            return batch['r_n'], None
        
        # 计算反事实优势
        advantages_cf = self.compute_counterfactual_advantages(batch)
        
        # 统计信息（用于日志记录）
        adv_cf_mean = advantages_cf.mean().item()
        adv_cf_std = advantages_cf.std().item()
        coma_stats = {'adv_cf_mean': adv_cf_mean, 'adv_cf_std': adv_cf_std}
        
        # Clip 优势
        advantages_clipped = torch.clamp(advantages_cf, -self.coma_clip, self.coma_clip)
        
        # 重塑奖励：原奖励 + beta * clipped_advantage
        r_shaped_n = batch['r_n'] + self.coma_beta * advantages_clipped
        
        return r_shaped_n, coma_stats

    def train_q_critic(self, batch, index):
        """
        训练 Q Critic 以拟合 TD target
        
        TD target: y_i = r_i + gamma * V_i(s')
        Loss: MSE(Q_i(s, a), y_i)
        """
        s = batch['s'][index]  # (mini_batch, T, state_dim)
        actions_onehot = batch['actions_onehot_n'][index]  # (mini_batch, T, N, A)
        B, T, N, A = actions_onehot.shape
        
        # Flatten joint action
        joint_action_flat = actions_onehot.reshape(B, T, N * A)
        
        # 计算 Q 值
        q_values = self.q_critic(s, joint_action_flat)  # (B, T, N)
        
        # TD target：使用个体奖励 r_i + gamma * V_i(s')
        q_target = batch['r_n'][index] + self.gamma * batch['v_n'][index, 1:] * (1 - batch['done_n'][index])
        
        # Q loss
        q_loss = F.mse_loss(q_values, q_target.detach())
        
        # 更新 Q critic
        self.q_optimizer.zero_grad()
        q_loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.q_critic.parameters(), 10.0)
        self.q_optimizer.step()

        return float(q_loss.detach().item())


