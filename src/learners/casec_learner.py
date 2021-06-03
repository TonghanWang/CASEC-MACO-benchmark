import copy

import torch as th
from components.episode_buffer import EpisodeBatch
from torch.optim import RMSprop


class CASECLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1
        self.p_lr = args.p_lr

        # action encoder
        self.action_encoder_params = list(self.mac.action_encoder_params())
        self.action_encoder_optimiser = RMSprop(params=self.action_encoder_params, lr=args.lr,
                                                alpha=args.optim_alpha, eps=args.optim_eps)

        self.action_repr_updating = True
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        self.l1_loss_weight = args.l1_loss_weight
        self.q_var_loss_weight = args.q_var_loss_weight
        self.delta_var_loss_weight = args.delta_var_loss_weight
        self.l1_loss = args.l1_loss
        self.q_var_loss = args.q_var_loss
        self.delta_var_loss = args.delta_var_loss
        self.sparse_graph = not args.full_graph

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]  # (bs,t,1)
        actions = batch["actions"][:, :-1]  # (bs,t,n,1)
        terminated = batch["terminated"][:, :-1].float()  # (bs,t,1)
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]  # (bs,t+1,n,|A|)

        # Calculate estimated Q-Values
        mac_out = []
        f_i_left, delta_ij_left, q_ij_left, atten_ij_left = [], [], [], []
        self.mac.init_hidden(batch.batch_size)  # (bs,n,hidden_size)
        for t in range(batch.max_seq_length):  # t+1
            agent_outs, f_i, delta_ij, q_ij, atten_ij = self.mac.forward(batch,
                                                                         t=t)  # (bs,1), (bs,n,|A|), (bs,n,n,|A|,|A|)
            mac_out.append(agent_outs)  # [t+1,(bs,1)]
            f_i_left.append(f_i)
            delta_ij_left.append(delta_ij)
            q_ij_left.append(q_ij)
            atten_ij_left.append(atten_ij)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time  # (bs,t+1,1)

        f_i_left = th.stack(f_i_left, dim=1)
        delta_ij_left = th.stack(delta_ij_left, dim=1)
        q_ij_left_all = th.stack(q_ij_left, dim=1)
        q_ij_left = th.stack(q_ij_left, dim=1)[:, :-1]
        atten_ij_left = th.stack(atten_ij_left, dim=1)

        chosen_action_qvals = mac_out[:, :-1]  # (bs,t,1)

        # Calculate the estimated Q-Values for target Q
        target_f_i, target_delta_ij, target_q_ij, target_his_cos_sim, target_atten_ij = [], [], [], [], []
        self.target_mac.init_hidden(batch.batch_size)  # (bs,n,hidden_size)
        for t in range(batch.max_seq_length):
            f_i, delta_ij, q_ij, his_cos_similarity, atten_ij = self.target_mac.caller_ip_q(batch,
                                                                                            t=t)  # (bs,n,|A|), (bs,n,n,|A|,|A|)
            target_f_i.append(f_i)  # [t+1,(bs,n,|A|)]
            target_delta_ij.append(delta_ij)  # [t+1,(bs,n,n,|A|,|A|)]
            target_q_ij.append(q_ij)  # [t+1,(bs,n,n,|A|,|A|)]
            target_his_cos_sim.append(his_cos_similarity)
            target_atten_ij.append(atten_ij)

        target_f_i = th.stack(target_f_i[1:], dim=1)  # (bs,t,n,|A|)
        target_delta_ij_all = th.stack(target_delta_ij, dim=1)  # (bs,t,n,n,|A|,|A|)
        target_q_ij_all = th.stack(target_q_ij, dim=1)  # (bs,t,n,n,|A|,|A|)
        target_his_cos_sim_all = th.stack(target_his_cos_sim, dim=1)  # (bs,t,n,n,|A|,|A|)
        target_q_ij = th.stack(target_q_ij[1:], dim=1)  # (bs,t,n,n,|A|,|A|)
        target_atten_ij = th.stack(target_atten_ij, dim=1)  # (bs,t,n,n)

        # Calculate the Q-Values necessary for the target
        mac_out_right = []
        self.mac.init_hidden(batch.batch_size)  # (bs,n,hidden_size)
        for t in range(batch.max_seq_length):
            f_i = f_i_left[:, t].detach()
            delta_ij = delta_ij_left[:, t].detach()
            q_ij = q_ij_left_all[:, t].detach()
            atten_ij = atten_ij_left[:, t].detach()
            target_agent_outs = self.mac.max_sum(batch, t=t, f_i=f_i, delta_ij=delta_ij, q_ij=q_ij, atten_ij=atten_ij,
                                                 target_delta_ij=target_delta_ij_all[:, t].detach(),
                                                 target_q_ij=target_q_ij_all[:, t].detach(),
                                                 target_his_cos_sim=target_his_cos_sim_all[:, t],
                                                 target_atten_ij=target_atten_ij[:, t])  # (bs,n,|A|)
            mac_out_right.append(target_agent_outs)  # [t,(bs,n,|A|)]

        # We don't need the first timesteps Q-Value estimate for calculating targets
        mac_out_right = th.stack(mac_out_right[1:], dim=1)  # Concat across time
        # (bs,t,n,n_actions)

        # Mask out unavailable actions
        mac_out_right[avail_actions[:, 1:] == 0] = -9999999  # Q values

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out_right.clone().detach()  # return a new Tensor, detached from the current graph
            # (bs,t,n,|A|), discard t=0
            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]  # indices instead of values
            # (bs,t,n,1)
            target_f_i_gather = th.gather(target_f_i, index=cur_max_actions, dim=-1)  # (bs,t,n,1)

            agent_actions_gather_i = cur_max_actions.unsqueeze(dim=3).unsqueeze(dim=-1).repeat(1, 1, 1,
                                                                                               self.args.n_agents, 1,
                                                                                               self.args.n_actions)
            agent_actions_gather_j = cur_max_actions.unsqueeze(dim=2).unsqueeze(dim=-2).repeat(1, 1, self.args.n_agents,
                                                                                               1, 1, 1)
            target_q_ij_gather = th.gather(target_q_ij, index=agent_actions_gather_i, dim=-2)
            target_q_ij_gather = th.gather(target_q_ij_gather, index=agent_actions_gather_j, dim=-1)
            target_q_ij_gather = target_q_ij_gather.squeeze()  # * self.mac.adj  # (bs,t,n,n)
            if self.args.construction_attention:
                target_max_qvals = target_f_i_gather.squeeze(dim=-1).mean(dim=-1) + (target_atten_ij[:, 1:] * target_q_ij_gather).sum(dim=-1).sum(dim=-1)
            else:
                target_max_qvals = target_f_i_gather.squeeze(dim=-1).mean(dim=-1) + target_q_ij_gather.sum(dim=-1).sum(dim=-1) / self.n_agents / (self.n_agents - 1)
            target_max_qvals.unsqueeze_(dim=-1)  # (bs,t,1)
        else:
            pass

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals.detach()

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())
        # (bs,t,1)

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

        if self.delta_var_loss:
            delta_ij_left_temp = delta_ij_left[:, :-1]
            var_loss = (delta_ij_left_temp.view(-1, batch.max_seq_length - 1,
                                                self.n_agents * self.n_agents * self.n_actions,
                                                self.n_actions).var(-1)).sum(-1).unsqueeze(-1) / self.n_agents / (self.n_agents - 1) / self.n_actions
            masked_var_loss = var_loss * mask
            loss = loss + self.delta_var_loss_weight * (masked_var_loss.sum() / mask.sum())
        elif self.q_var_loss:
            var_loss = (q_ij_left.view(-1, batch.max_seq_length - 1,
                                       self.n_agents * self.n_agents * self.n_actions,
                                       self.n_actions).var(-1)).sum(-1).unsqueeze(-1) / self.n_agents / (self.n_agents - 1) / self.n_actions
            masked_var_loss = var_loss * mask
            loss = loss + self.q_var_loss_weight * (masked_var_loss.sum() / mask.sum())
        elif self.l1_loss:
            delta_ij_left = delta_ij_left[:, :-1]
            l1_loss = th.norm(delta_ij_left.view(-1, batch.max_seq_length - 1, self.n_agents, self.n_agents,
                                                 self.n_actions * self.n_actions), p=1, dim=-1).sum(-1).sum(-1).unsqueeze(-1) / self.n_agents / (self.n_agents - 1)
            masked_l1_loss = l1_loss * mask
            loss = loss + self.l1_loss_weight * (masked_l1_loss.sum() / mask.sum())
        else:
            pass

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)  # max_norm
        self.optimiser.step()

        pred_obs_loss = None
        pred_r_loss = None
        pred_grad_norm = None
        if self.action_repr_updating:
            # train action encoder
            no_pred = []
            r_pred = []
            for t in range(batch.max_seq_length):
                no_preds, r_preds = self.mac.action_repr_forward(batch, t=t)
                no_pred.append(no_preds)
                r_pred.append(r_preds)
            no_pred = th.stack(no_pred, dim=1)[:, :-1]  # Concat over time
            r_pred = th.stack(r_pred, dim=1)[:, :-1]
            no = batch["obs"][:, 1:].detach().clone()
            repeated_rewards = batch["reward"][:, :-1].detach().clone().unsqueeze(2).repeat(1, 1, self.n_agents, 1)

            pred_obs_loss = th.sqrt(((no_pred - no) ** 2).sum(dim=-1)).mean()
            pred_r_loss = ((r_pred - repeated_rewards) ** 2).mean()

            pred_loss = pred_obs_loss + 10 * pred_r_loss
            self.action_encoder_optimiser.zero_grad()
            pred_loss.backward()
            pred_grad_norm = th.nn.utils.clip_grad_norm_(self.action_encoder_params, self.args.grad_norm_clip)
            self.action_encoder_optimiser.step()

            if t_env > self.args.action_repr_learning_phase:
                self.mac.update_action_repr()
                self.action_repr_updating = False
                self._update_targets()
                self.last_target_update_episode = episode_num

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            if self.sparse_graph:
                if not self.l1_loss:
                    delta_ij_left = delta_ij_left[:, :-1]
                nonzero = (delta_ij_left.detach() > self.mac.edge_threshold).float()
                self.logger.log_stat("sparseness",
                                     (nonzero.mean(-1).mean(-1).mean(
                                         -1).mean(-1).unsqueeze(-1) * mask).sum().item() / (
                                             mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            if pred_obs_loss is not None:
                self.logger.log_stat("pred_obs_loss", pred_obs_loss.item(), t_env)
                self.logger.log_stat("pred_r_loss", pred_r_loss.item(), t_env)
                self.logger.log_stat("action_encoder_grad_norm", pred_grad_norm, t_env)
            self.logger.log_stat("q_taken_mean",
                                 (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                                 t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")
        self.target_mac.action_repr_updating = self.action_repr_updating

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
