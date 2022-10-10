from typing import Dict
import torch

from .problem_pdp import PDP


class PDTSPL(PDP):
    def __init__(
        self, p_size: int, init_val_met: str = 'p2d', with_assert: bool = False
    ) -> None:
        super().__init__(p_size, init_val_met, with_assert)

        self.name = 'pdtspl'  # Pickup and Delivery TSP with LIFO constriant

        print(
            f'PDTSP-LIFO with {self.size} nodes.',
            ' Do assert:',
            with_assert,
        )

    def get_initial_solutions(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:

        batch_size = batch['coordinates'].size(0)

        def get_solution(methods: str) -> torch.Tensor:

            half_size = self.size // 2
            p_size = self.size

            if methods == 'random':
                candidates = torch.ones(batch_size, self.size + 1).bool()
                candidates[:, half_size + 1 :] = 0
                rec = torch.zeros(batch_size, self.size + 1).long()
                selected_node = torch.zeros(batch_size, 1).long()
                candidates.scatter_(1, selected_node, 0)
                stacks = (
                    torch.zeros(batch_size, half_size + 1) - 0.01
                )  # fix bug: max is not stable sorting
                stacks[:, 0] = 0  # fix bug: max is not stable sorting
                for i in range(self.size):

                    index1 = (selected_node <= half_size) & (selected_node > 0)
                    if index1.any():
                        stacks[index1.view(-1), selected_node[index1]] = i + 1
                    top = stacks.max(-1)[1]

                    dists = torch.ones(batch_size, self.size + 1)
                    dists.scatter_(1, selected_node, -1e20)
                    dists[~candidates] = -1e20
                    dists[:, half_size + 1 :] = -1e20  # mask all delivery
                    dists[top > 0, top[top > 0] + half_size] = 1
                    dists = torch.softmax(dists, -1)
                    next_selected_node = dists.multinomial(1).view(-1, 1)
                    index2 = (next_selected_node > half_size) & (next_selected_node > 0)
                    if index2.any():
                        stacks[
                            index2.view(-1), next_selected_node[index2] - half_size
                        ] = -0.01

                    rec.scatter_(1, selected_node, next_selected_node)
                    candidates.scatter_(1, next_selected_node, 0)
                    selected_node = next_selected_node

                return rec

            elif methods == 'greedy':

                candidates = torch.ones(batch_size, self.size + 1).bool()
                candidates[:, half_size + 1 :] = 0
                rec = torch.zeros(batch_size, self.size + 1).long()
                selected_node = torch.zeros(batch_size, 1).long()
                candidates.scatter_(1, selected_node, 0)
                stacks = torch.zeros(batch_size, half_size + 1) - 0.01
                stacks[:, 0] = 0  # fix bug: max is not stable sorting
                for i in range(self.size):

                    index1 = (selected_node <= half_size) & (selected_node > 0)
                    if index1.any():
                        stacks[index1.view(-1), selected_node[index1]] = i + 1
                    top = stacks.max(-1)[1]

                    d1 = (
                        batch['coordinates']
                        .cpu()
                        .gather(
                            1,
                            selected_node.unsqueeze(-1).expand(
                                batch_size, self.size + 1, 2
                            ),
                        )
                    )
                    d2 = batch['coordinates'].cpu()

                    dists = (d1 - d2).norm(p=2, dim=2)
                    # dists = batch['dist'].cpu().gather(1,selected_node.view(batch_size,1,1).expand(batch_size, 1, self.size + 1)).squeeze().clone()
                    dists.scatter_(1, selected_node, 1e6)
                    dists[~candidates] = 1e6

                    dists[:, p_size // 2 + 1 :] += 1e3  # mask all delivery
                    dists[top > 0, top[top > 0] + half_size] -= 1e3

                    next_selected_node = dists.min(-1)[1].view(-1, 1)

                    index2 = (next_selected_node > half_size) & (next_selected_node > 0)
                    if index2.any():
                        stacks[
                            index2.view(-1), next_selected_node[index2] - half_size
                        ] = -0.01

                    add_index = (next_selected_node <= half_size).view(-1)
                    pairing = (
                        next_selected_node[next_selected_node <= half_size].view(-1, 1)
                        + half_size
                    )
                    candidates[add_index] = candidates[add_index].scatter_(
                        1, pairing, 1
                    )

                    rec.scatter_(1, selected_node, next_selected_node)
                    candidates.scatter_(1, next_selected_node, 0)
                    selected_node = next_selected_node

                return rec

            else:
                raise NotImplementedError()

        return get_solution(self.init_val_met).expand(batch_size, self.size + 1).clone()

    @staticmethod
    def get_swap_mask(
        selected_node: torch.Tensor,
        visited_order_map: torch.Tensor,
        top2: torch.Tensor,
    ) -> torch.Tensor:

        bs, gs, _ = visited_order_map.size()

        top = torch.where(top2[:, :, 0] == selected_node, top2[:, :, 1], top2[:, :, 0])
        mask_pd = top.view(-1, gs, 1) != top.view(-1, 1, gs)

        mask = visited_order_map.clone()
        mask[torch.arange(bs), selected_node.view(-1)] = True
        mask[torch.arange(bs), selected_node.view(-1) + gs // 2] = True
        mask[torch.arange(bs), :, selected_node.view(-1)] = True
        mask[torch.arange(bs), :, selected_node.view(-1) + gs // 2] = True

        return mask | mask_pd

    def _check_feasibility(self, rec: torch.Tensor) -> None:

        p_size = self.size + 1

        assert (
            (torch.arange(p_size, out=rec.new())).view(1, -1).expand_as(rec)
            == rec.sort(1)[0]
        ).all(), (
            (
                (torch.arange(p_size, out=rec.new())).view(1, -1).expand_as(rec)
                == rec.sort(1)[0]
            ),
            "not visiting all nodes",
            rec,
        )

        # calculate visited time
        bs = rec.size(0)
        visited_time = torch.zeros((bs, p_size), device=rec.device)
        stacks = torch.zeros((bs, p_size // 2), device=rec.device).long() - 0.01
        stacks[:, 0] = 0  # fix bug: max is not stable sorting
        pre = torch.zeros((bs), device=rec.device).long()
        arange = torch.arange(bs)
        for i in range(p_size):
            cur = rec[arange, pre]
            visited_time[arange, cur] = i + 1
            pre = cur
            index1 = (cur <= p_size // 2) & (cur > 0)
            index2 = (cur > p_size // 2) & (cur > 0)
            if index1.any():
                stacks[index1, cur[index1] - 1] = i + 1
            assert (
                stacks.max(-1)[1][index2] == (cur[index2] - 1 - p_size // 2)
            ).all(), 'pdtsp error'
            if (index2).any():
                stacks[index2, cur[index2] - 1 - p_size // 2] = -0.01

        assert (
            visited_time[:, 1 : p_size // 2 + 1] < visited_time[:, p_size // 2 + 1 :]
        ).all(), (
            visited_time[:, 1 : p_size // 2 + 1] < visited_time[:, p_size + 1 // 2 :],
            "deliverying without pick-up",
        )
