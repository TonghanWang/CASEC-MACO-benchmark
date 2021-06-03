import copy

import torch as th
import torch.nn as nn
import torch_scatter
from components.action_selectors import REGISTRY as action_REGISTRY
from modules.action_encoders import REGISTRY as action_encoder_REGISTRY
from modules.agents import REGISTRY as agent_REGISTRY


# from torch_geometric.data import DataLoader
# from utils.gnn_utils import *


class CASECMAC(object):
    def __init__(self, scheme, groups, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.bs = None
        self.independent_p_q = args.independent_p_q

        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type
        self.hidden_states = None
        self.p_hidden_states = None

        self.action_selector = action_REGISTRY[args.action_selector](args)
        self.use_action_repr = args.use_action_repr

        delta_input_length = (
                2 * self.args.pair_rnn_hidden_dim) if self.independent_p_q else 2 * self.args.rnn_hidden_dim

        if self.use_action_repr:
            self.delta = nn.Linear(delta_input_length, 2 * self.args.action_latent_dim)
        else:
            self.delta = nn.Linear(delta_input_length, self.n_actions ** 2)

        self.p_lr = args.p_lr
        self.zeros = th.zeros([1, self.n_agents, self.n_agents]).to(args.device)
        wo_diag = th.ones(self.n_agents, self.n_agents)
        diag = th.ones(self.n_agents)
        diag = th.diag(diag, 0)
        self.wo_diag = (wo_diag - diag).to(args.device).unsqueeze(0)

        self.eye2 = th.eye(self.n_agents).bool()
        self.eye2 = self.eye2.to(args.device)
        self.edges_from = th.tensor([[i] * self.n_agents for i in range(self.n_agents)]).view(-1).unsqueeze(0).to(args.device)
        self.edges_to = th.tensor([[i for i in range(self.n_agents)] * self.n_agents]).view(-1).unsqueeze(0).to(args.device)
        self.message = th.zeros(2, 1, self.n_actions).to(args.device)

        # adjacency matrix
        self.full_graph = args.full_graph
        if self.full_graph:
            adj = th.ones(self.n_agents, self.n_agents)
            self.adj = adj.to(self.args.device).unsqueeze(0)

        self.action_encoder = action_encoder_REGISTRY[args.action_encoder](args)
        self.action_repr = th.ones(self.n_actions, self.args.action_latent_dim).to(args.device)
        input_i = self.action_repr.unsqueeze(1).repeat(1, self.n_actions, 1)
        input_j = self.action_repr.unsqueeze(0).repeat(self.n_actions, 1, 1)
        self.p_action_repr = th.cat([input_i, input_j], dim=-1).view(self.n_actions * self.n_actions,
                                                                     -1).t().unsqueeze(0)
        self.temp_tensor = th.zeros(self.n_agents, self.n_actions).to(args.device)
        self.edge_threshold = args.edge_threshold_start

        self.construction_delta_var = args.construction_delta_var
        self.construction_q_var = args.construction_q_var
        self.construction_delta_abs = args.construction_delta_abs
        self.construction_history_similarity = args.construction_history_similarity
        self.construction_attention = args.construction_attention
        self.random_graph = args.random_graph

        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

        self.atten_key = nn.Linear(args.rnn_hidden_dim, args.atten_dim)
        self.atten_query = nn.Linear(args.rnn_hidden_dim, args.atten_dim)
        self.atten_sofmax = nn.Softmax(dim=1)

    def MaxSum_faster(self, x, adj, q_ij, atten_ij=None, available_actions=None, k=5):
        # (bs,n,|A|), (bs,n,n), (bs,n,n,|A|,|A|), (bs,n,|A|) -> (bs,n,|A|)
        adj[:, self.eye2] = 0.
        num_edges = int(adj[0].sum(-1).sum(-1))  # Samples in the batch should have the same number of edges
        edges_from = self.edges_from.repeat(x.shape[0], 1)[adj.view(-1, self.n_agents ** 2) == 1].view(-1, num_edges)
        edges_to = self.edges_to.repeat(x.shape[0], 1)[adj.view(-1, self.n_agents ** 2) == 1].view(-1, num_edges)
        nodes = th.cat([edges_from, edges_to], dim=1)  # (bs,2|E|)

        x = x / self.n_agents
        if self.construction_attention:
            q_ij = q_ij * atten_ij.unsqueeze(-1).unsqueeze(-1)
            q_ij /= (atten_ij * adj).sum(-1).sum(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        else:
            q_ij = q_ij / num_edges

        q_ij_new = q_ij[adj == 1].view(-1, num_edges, self.n_actions, self.n_actions)
        # q_left_down = self.message.clone().unsqueeze(0).repeat(self.bs, 1, num_edges, 1)
        r_down_left = self.message.clone().unsqueeze(0).repeat(self.bs, 1, num_edges, 1)
        # (bs,2,|E|,|A|,|A|),(bs,2,|E|,|A|), (bs,2,|E|,|A|)

        # Unavailable actions have a utility of -inf, which propagates throughout message passing
        if available_actions is not None:
            x = x.masked_fill(available_actions == 0, -99999999)
            available_actions_new = th.gather(available_actions, dim=1,\
                    index=nodes.unsqueeze(-1).repeat(1, 1, self.n_actions)).view(-1, 2, num_edges, self.n_actions)

        for _ in range(k):
            # Message from variable node i to function node g:
            q_left_down_sum = torch_scatter.scatter_add(src=r_down_left.view(-1, 2 * num_edges, self.n_actions), \
                    index=nodes, dim=1, dim_size=self.n_agents)
            q_left_down_sum += x
            q_left_down = th.gather(q_left_down_sum, dim=1,\
                    index=nodes.unsqueeze(-1).repeat(1, 1, self.n_actions)).view(-1, 2, num_edges, self.n_actions)
            q_left_down -= r_down_left
            # Normalize
            if available_actions is not None:
                q_left_down -= (q_left_down * available_actions_new).sum(-1, keepdim=True) / available_actions_new.sum(-1, keepdim=True)
            else:
                q_left_down -= q_left_down.mean(dim=-1, keepdim=True)

            # Message from function node g to variable node i:
            r_down_left[:, 0] = (q_ij_new + q_left_down[:, 1].unsqueeze(-2)).max(dim=-1)[0]
            r_down_left[:, 1] = (q_ij_new + q_left_down[:, 0].unsqueeze(-1)).max(dim=-2)[0]

        # Calculate the z value
        z = torch_scatter.scatter_add(src=r_down_left.view(-1, 2 * num_edges, self.n_actions), \
                                      index=nodes, dim=1, dim_size=self.n_agents)
        z += x
        return z

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        f_i, delta_ij, q_ij, his_cos_sim, atten_ij = self.calculate(ep_batch, t_ep)
        agent_outputs = self.max_sum(ep_batch, t_ep, f_i=f_i, delta_ij=delta_ij, q_ij=q_ij,
                                     his_cos_sim=his_cos_sim.detach(), atten_ij=atten_ij)  # (bs,n,|A|)
        # select optim actions, so we should use forward right
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env,
                                                            test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t):
        # (bs,n,|A|), (bs,n,n,|A|,|A|) = (b,T,n,|A|)
        f_i, delta_ij, q_ij, _, atten_ij = self.calculate(ep_batch, t)

        # Gather individual Qs
        agent_actions = ep_batch['actions'][:, t]  # (bs,n,1)
        f_i_gather = th.gather(f_i, index=agent_actions, dim=-1)  # (bs,n,1)

        # Gather pairwise Qs
        agent_actions_gather_i = agent_actions.unsqueeze(dim=2).unsqueeze(dim=-1).repeat(1, 1, self.n_agents, 1,
                                                                                         self.n_actions)
        agent_actions_gather_j = agent_actions.unsqueeze(dim=1).unsqueeze(dim=-2).repeat(1, self.n_agents, 1, 1, 1)
        edge_attr = th.gather(q_ij, index=agent_actions_gather_i, dim=-2)
        edge_attr = th.gather(edge_attr, index=agent_actions_gather_j, dim=-1)
        edge_attr = edge_attr.squeeze()  # * self.adj  # (bs,n,n)

        if self.construction_attention:
            agent_outs = f_i_gather.squeeze(dim=-1).mean(dim=-1) + (edge_attr * atten_ij).sum(dim=-1).sum(
                dim=-1) * self.p_lr
        else:
            agent_outs = f_i_gather.squeeze(dim=-1).mean(dim=-1) + edge_attr.sum(dim=-1).sum(dim=-1) / self.n_agents / (self.n_agents - 1) * self.p_lr

        agent_outs.unsqueeze_(dim=-1)
        return agent_outs, f_i, delta_ij, q_ij, atten_ij
        # (bs, 1), (bs,n,|A|), (bs,n,n,|A|,|A|)

    def max_sum(self, ep_batch, t, f_i=None, delta_ij=None, q_ij=None, his_cos_sim=None, atten_ij=None,
                target_delta_ij=None, target_q_ij=None, target_his_cos_sim=None, target_atten_ij=None):
        # Calculate the utilities of each agent i and the incremental matrix delta for each agent pair (i&j).
        x, adj, edge_attr, q_ij = self.construction(f_i, delta_ij, q_ij, his_cos_sim, atten_ij, ep_batch, t,
                                                    target_delta_ij, target_q_ij, target_his_cos_sim, target_atten_ij,
                                                    available_actions=ep_batch['avail_actions'][:, t])

        # (bs,n,|A|) = (b,n,|A|), (b,n,n), (b,E,|A|,|A|)
        x_out = self.MaxSum_faster(x.detach(), adj.detach(), q_ij.detach(), atten_ij.detach(), available_actions=ep_batch['avail_actions'][:, t])
        return x_out

    def caller_ip_q(self, ep_batch, t):
        # Calculate the utilities of each agent i and the incremental matrix delta for each agent pair (i&j).
        # (bs,n,|A|), (bs,n,n,|A|,|A|) = (b,T,n,|A|)
        f_i, delta_ij, q_ij, his_cos_similarity, atten_ij = self.calculate(ep_batch, t)

        # return individual and pair-wise q function
        return f_i, delta_ij, q_ij * self.p_lr, his_cos_similarity, atten_ij

    def calculate(self, ep_batch, t):
        agent_inputs = self._build_inputs(ep_batch, t)  # (bs*n, 3n) i.e. (bs*n, (obs+act+id))
        self.bs = ep_batch.batch_size

        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states, self.action_repr)
        agent_outs = agent_outs.view(self.bs, self.n_agents, self.n_actions)
        f_i = agent_outs.clone()

        if self.independent_p_q:
            self.p_hidden_states = self.p_agent.h_forward(agent_inputs, self.p_hidden_states).view(self.bs,
                                                                                                   self.n_agents, -1)
        else:
            self.p_hidden_states = self.hidden_states.clone().view(self.bs, self.n_agents, -1)

        delta_ij, his_cos_similarity = self._calculate_delta(self.p_hidden_states)
        # (bs,n,rnn_hidden_dim) -> (bs,n,n,|A|,A|)

        f_i_expand_j = f_i.unsqueeze(dim=1).unsqueeze(dim=-2).repeat(1, self.n_agents, 1, self.n_actions, 1)
        f_i_expand_i = f_i.unsqueeze(dim=2).unsqueeze(dim=-1).repeat(1, 1, self.n_agents, 1, self.n_actions)
        q_ij = f_i_expand_i.detach() + f_i_expand_j.detach() + delta_ij

        atten_ij = self._calculate_attention(self.p_hidden_states)

        delta_ij[:, self.eye2] = 0
        q_ij[:, self.eye2] = 0
        return f_i, delta_ij, q_ij, his_cos_similarity, atten_ij

    def _calculate_attention(self, hidden_states):
        # print(hidden_states.shape)
        atten_i = self.atten_key(hidden_states.view(-1, self.args.rnn_hidden_dim)).view(self.bs, self.n_agents,
                                                                                        -1).unsqueeze(2).repeat(1, 1,
                                                                                                                self.n_agents,
                                                                                                                1)  # (bs, n, n, atten_dim)
        atten_j = self.atten_query(hidden_states.view(-1, self.args.rnn_hidden_dim)).view(self.bs, self.n_agents,
                                                                                          -1).unsqueeze(1).repeat(1,
                                                                                                                  self.n_agents,
                                                                                                                  1,
                                                                                                                  1)  # (bs, n, n, atten_dim)

        atten_ij = (atten_i * atten_j).sum(-1).view(self.bs, -1)  # (bs, n*n)
        atten_ij[:, self.eye2.view(-1)] = -999999
        atten_ij = self.atten_sofmax(atten_ij)

        return atten_ij.view(self.bs, self.n_agents, self.n_agents)

    def _calculate_delta(self, hidden_states):
        # (bs,n,rnn_hidden_dim) -> (bs,n,n,|A|,|A|)
        # (bs,n,rnn_hidden_dim) -> (bs,n,n,2*rnn_hidden_dim)
        input_i = hidden_states.unsqueeze(2).repeat(1, 1, self.n_agents, 1)
        input_j = hidden_states.unsqueeze(1).repeat(1, self.n_agents, 1, 1)

        if self.independent_p_q:
            inputs = th.cat([input_i, input_j], dim=-1).view(-1, 2 * self.args.pair_rnn_hidden_dim)
        else:
            inputs = th.cat([input_i, input_j], dim=-1).view(-1, 2 * self.args.rnn_hidden_dim)

        history_cos_similarity = self.zeros.repeat(self.bs, 1, 1)
        if self.construction_history_similarity:
            history_cos_similarity = self.cosine_similarity(input_i.detach(), input_j.detach()).detach()

        # (bs,n,n,2*rnn_hidden_dim) -> (bs,n,n,|A|x|A|)

        if self.use_action_repr:
            key = self.delta(inputs).view(self.bs, self.n_agents * self.n_agents, -1)
            # (bs, n*n, 2al)  # p_action_repr (bs, 2al, |A|*|A|)
            f_ij = th.bmm(key, self.p_action_repr.repeat(self.bs, 1, 1)) / self.args.action_latent_dim / 2
        else:
            f_ij = self.delta(inputs)

        f_ij = f_ij.view(self.bs, self.n_agents, self.n_agents, self.n_actions, self.n_actions)
        f_ij = (f_ij + f_ij.permute(0, 2, 1, 4, 3).detach()) / 2.
        return f_ij, history_cos_similarity

    def _variance(self, a, mask):
        mean = (a * mask).sum(-1) / mask.sum(-1)
        var = ((a - mean.unsqueeze(-1)) ** 2 * mask).sum(-1) / mask.sum(-1)
        return var

    def construction(self, f_i, delta_ij, q_ij, his_cos_sim, atten_ij, ep_batch, t, target_delta_ij=None,
                     target_q_ij=None, target_his_cos_sim=None, target_atten_ij=None, available_actions=None):
        # available_actions: (bs,n,|A|)
        available_actions_i = available_actions.detach().unsqueeze(dim=2).repeat(1, 1, self.n_agents, 1).view(-1, self.n_agents * self.n_agents, self.n_actions)
        available_actions_j = available_actions.detach().unsqueeze(dim=1).unsqueeze(dim=-2)\
            .repeat(1, self.n_agents, 1, self.n_actions, 1).view(-1, self.n_agents * self.n_agents, self.n_actions, self.n_actions)
        available_actions_ij = (available_actions_i.unsqueeze(dim=-1).repeat(1, 1, 1, self.n_actions) * available_actions_j)\
            .view(-1, self.n_agents * self.n_agents, self.n_actions * self.n_actions)
        x = f_i.clone()

        if self.full_graph:
            adj = self.adj.repeat(self.bs, 1, 1)
        else:
            if self.random_graph:
                indicator = th.rand(self.bs, self.n_agents * self.n_agents).to(self.args.device)
            elif self.construction_q_var:
                if target_q_ij is not None:
                    indicator = self._variance(target_q_ij.detach().view(-1, self.n_agents * self.n_agents, self.n_actions, self.n_actions), available_actions_j)
                    indicator = (indicator * available_actions_i).max(-1)[0]
                else:
                    indicator = self._variance(q_ij.detach().view(-1, self.n_agents * self.n_agents, self.n_actions, self.n_actions), available_actions_j)
                    indicator = (indicator * available_actions_i).max(-1)[0]

            elif self.construction_delta_var:
                if target_delta_ij is not None:
                    indicator = self._variance(target_delta_ij.detach().view(-1, self.n_agents * self.n_agents, self.n_actions, self.n_actions), available_actions_j)
                    indicator = (indicator * available_actions_i).max(-1)[0]
                else:
                    indicator = self._variance(delta_ij.detach().view(-1, self.n_agents * self.n_agents, self.n_actions, self.n_actions), available_actions_j)
                    indicator = (indicator * available_actions_i).max(-1)[0]
            elif self.construction_delta_abs:
                if target_delta_ij is not None:
                    indicator = (target_delta_ij.detach().view(-1, self.n_agents * self.n_agents, self.n_actions * self.n_actions)
                                 * available_actions_ij).abs().max(-1)[0]
                else:
                    indicator = (delta_ij.detach().view(-1, self.n_agents * self.n_agents, self.n_actions * self.n_actions)
                                 * available_actions_ij).abs().max(-1)[0]
            elif self.construction_history_similarity:
                if target_his_cos_sim is not None:
                    indicator = target_his_cos_sim.view(-1, self.n_agents * self.n_agents)
                else:
                    indicator = his_cos_sim.view(-1, self.n_agents * self.n_agents)
            elif self.construction_attention:
                if target_atten_ij is not None:
                    indicator = target_atten_ij.detach().view(-1, self.n_agents * self.n_agents)[0]
                else:
                    indicator = atten_ij.detach().view(-1, self.n_agents * self.n_agents)[0]
            else:
                indicator = None
                raise NotImplementedError

            adj_tensor = indicator.masked_fill(self.wo_diag.repeat(self.bs, 1, 1).view(-1, self.n_agents * self.n_agents) == 0, -99999999)
            adj_tensor_topk = \
                th.topk(adj_tensor, int(self.n_agents * self.n_agents * self.args.threshold // 2 * 2), dim=-1)[1]
            adj = self.zeros.repeat(self.bs, 1, 1).view(-1, self.n_agents * self.n_agents)
            adj.scatter_(1, adj_tensor_topk, 1)
            adj = adj.view(-1, self.n_agents, self.n_agents).detach()
            adj[:, self.eye2] = 1.

        return x, adj, None, q_ij * self.p_lr

    def update_action_repr(self):
        action_repr = self.action_encoder()

        self.action_repr = action_repr.detach().clone()

        print('>>> Action Representation', self.action_repr)

        # Pairwise Q (|A|, al) -> (|A|, |A|, 2*al)
        input_i = self.action_repr.unsqueeze(1).repeat(1, self.n_actions, 1)
        input_j = self.action_repr.unsqueeze(0).repeat(self.n_actions, 1, 1)
        self.p_action_repr = th.cat([input_i, input_j], dim=-1).view(self.n_actions * self.n_actions,
                                                                     -1).t().unsqueeze(0)

    def post_processing(self, x_out):
        agent_outs = th.max(x_out, dim=-1, keepdim=True)[1]
        return agent_outs

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents,
                                                                          -1)  # (b, n, actions)
        if self.independent_p_q:
            self.p_hidden_states = self.p_agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents,
                                                                                  -1)  # (b, n, actions)
        else:
            self.p_hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents,
                                                                                -1)  # (b, n, actions)

    def parameters(self):
        """ Returns a generator for all parameters of the controller. """
        if self.construction_attention:
            param = list(self.agent.parameters()) + list(self.delta.parameters()) + list(
                self.atten_query.parameters()) + list(self.atten_key.parameters())
        elif self.independent_p_q:
            param = list(self.agent.parameters()) + list(self.p_agent.parameters()) + list(self.delta.parameters())
        else:
            param = list(self.agent.parameters()) + list(self.delta.parameters())

        return param

    def load_state(self, other_mac):
        """ Overwrites the parameters with those from other_mac. """
        self.agent.load_state_dict(other_mac.agent.state_dict())
        if self.independent_p_q:
            self.p_agent.load_state_dict(other_mac.p_agent.state_dict())
        if self.construction_attention:
            self.atten_query.load_state_dict(other_mac.atten_query.state_dict())
            self.atten_key.load_state_dict(other_mac.atten_key.state_dict())
        self.delta.load_state_dict(other_mac.delta.state_dict())
        self.action_encoder.load_state_dict(other_mac.action_encoder.state_dict())
        self.action_repr = copy.deepcopy(other_mac.action_repr)
        self.p_action_repr = copy.deepcopy(other_mac.p_action_repr)

    def cuda(self):
        """ Moves this controller to the GPU, if one exists. """
        self.agent.cuda()
        if self.independent_p_q:
            self.p_agent.cuda()
        self.delta.cuda()
        self.atten_sofmax.cuda()
        self.atten_query.cuda()
        self.atten_key.cuda()
        self.action_encoder.cuda()

    def save_models(self, path):
        """ Saves parameters to the disc. """
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))
        if self.independent_p_q:
            th.save(self.p_agent.state_dict(), "{}/p_agent.th".format(path))
        if self.construction_attention:
            th.save(self.atten_query.state_dict(), "{}/atten_query.th".format(path))
            th.save(self.atten_key.state_dict(), "{}/atten_key.th".format(path))
        th.save(self.delta.state_dict(), "{}/delta.th".format(path))
        th.save(self.action_encoder.state_dict(), "{}/action_encoder.th".format(path))
        th.save(self.action_repr, "{}/action_repr.pt".format(path))
        th.save(self.p_action_repr, "{}/p_action_repr.pt".format(path))

    def load_models(self, path):
        """ Loads parameters from the disc. """
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
        # gpu -> cpu
        if self.independent_p_q:
            self.p_agent.load_state_dict(
                th.load("{}/p_agent.th".format(path), map_location=lambda storage, loc: storage))
        if self.construction_attention:
            self.atten_query.load_state_dict(
                th.load("{}/atten_query.th".format(path), map_location=lambda storage, loc: storage))
            self.atten_key.load_state_dict(
                th.load("{}/atten_key.th".format(path), map_location=lambda storage, loc: storage))
        self.delta.load_state_dict(th.load("{}/delta.th".format(path), map_location=lambda storage, loc: storage))
        self.action_encoder.load_state_dict(th.load("{}/action_encoder.th".format(path),
                                                    map_location=lambda storage, loc: storage))

        self.action_repr = th.load("{}/action_repr.pt".format(path),
                                   map_location=lambda storage, loc: storage).to(self.args.device)
        self.p_action_repr = th.load("{}/p_action_repr.pt".format(path),
                                     map_location=lambda storage, loc: storage).to(self.args.device)

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)
        if self.independent_p_q:
            self.p_agent = agent_REGISTRY[self.args.pair_agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:  # True for QMix
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))  # last actions are empty
            else:
                inputs.append(batch["actions_onehot"][:, t - 1])
        if self.args.obs_agent_id:  # True for QMix
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))  # onehot agent ID

        # inputs[i]: (bs,n,n)
        inputs = th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)  # (bs*n, act+obs+id)
        # inputs[i]: (bs*n,n); ==> (bs*n,3n) i.e. (bs*n,(obs+act+id))
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape

    def action_encoder_params(self):
        return list(self.action_encoder.parameters())

    def action_repr_forward(self, ep_batch, t):
        return self.action_encoder.predict(ep_batch["obs"][:, t], ep_batch["actions_onehot"][:, t])
