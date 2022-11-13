from typing import Dict
import torch

from .problem_pdp import PDP


class PDTSPL(PDP):
    def __init__(
        self, size: int, init_val_method: str, check_feasible: bool = False
    ) -> None:
        super().__init__(size, init_val_method, check_feasible)

        self.name = 'pdtspl'  # Pickup and Delivery TSP with LIFO constriant

        print(
            f'PDTSP-LIFO with {self.size} nodes.',
            ' Do assert:',
            check_feasible,
        )

    @staticmethod
    def get_swap_mask(
        selected_node: torch.Tensor,
        visit_index: torch.Tensor,
        top2: torch.Tensor,
    ) -> torch.Tensor:
        visited_order_map = PDP._get_visit_order_map(visit_index)
        batch_size, graph_size_plus1, _ = visited_order_map.size()
        arange = torch.arange(batch_size)
        top = torch.where(top2[:, :, 0] == selected_node, top2[:, :, 1], top2[:, :, 0])
        mask_pd = top.view(-1, graph_size_plus1, 1) != top.view(-1, 1, graph_size_plus1)

        mask = visited_order_map.clone()  # true means unavailable
        mask[arange, selected_node.view(-1)] = True
        mask[arange, selected_node.view(-1) + graph_size_plus1 // 2] = True
        mask[arange, :, selected_node.view(-1)] = True
        mask[arange, :, selected_node.view(-1) + graph_size_plus1 // 2] = True

        return mask | mask_pd

    def get_initial_solutions(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:

        batch_size = batch['coordinates'].size(0)

        def get_solution(methods: str) -> torch.Tensor:

            half_size = self.size // 2

            if methods == 'random':
                candidates = torch.ones(batch_size, self.size + 1).bool()
                candidates[:, half_size + 1 :] = 0
                solution = torch.zeros(batch_size, self.size + 1).long()
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
                    dists[~candidates] = -1e20
                    dists[top > 0, top[top > 0] + half_size] = 1
                    dists = torch.softmax(dists, -1)
                    next_selected_node = dists.multinomial(1).view(-1, 1)
                    index2 = (next_selected_node > half_size) & (next_selected_node > 0)
                    if index2.any():
                        stacks[
                            index2.view(-1), next_selected_node[index2] - half_size
                        ] = -0.01

                    solution.scatter_(1, selected_node, next_selected_node)
                    candidates.scatter_(1, next_selected_node, 0)
                    selected_node = next_selected_node

                return solution

            elif methods == 'greedy':

                candidates = torch.ones(batch_size, self.size + 1).bool()
                candidates[:, half_size + 1 :] = 0
                solution = torch.zeros(batch_size, self.size + 1).long()
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
                    dists[~candidates] = 1e6

                    dists[:, self.size // 2 + 1 :] += 1e3  # mask all delivery
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

                    solution.scatter_(1, selected_node, next_selected_node)
                    candidates.scatter_(1, next_selected_node, 0)
                    selected_node = next_selected_node

                return solution

            else:
                raise NotImplementedError()

        return (
            get_solution(self.init_val_method).expand(batch_size, self.size + 1).clone()
        )

    def _check_feasibility(self, solution: torch.Tensor) -> None:

        size_p1 = self.size + 1

        assert (
            (torch.arange(size_p1, out=solution.new())).view(1, -1).expand_as(solution)
            == solution.sort(1)[0]
        ).all(), (
            (
                (torch.arange(size_p1, out=solution.new()))
                .view(1, -1)
                .expand_as(solution)
                == solution.sort(1)[0]
            ),
            "not visiting all nodes",
            solution,
        )

        # calculate visited time
        batch_size = solution.size(0)
        visited_time = torch.zeros((batch_size, size_p1), device=solution.device)
        stacks = (
            torch.zeros((batch_size, size_p1 // 2), device=solution.device).long()
            - 0.01
        )
        pre = torch.zeros(batch_size, device=solution.device).long()
        arange = torch.arange(batch_size)
        for i in range(size_p1):
            cur = solution[arange, pre]
            visited_time[arange, cur] = i + 1
            pre = cur
            index1 = (cur <= size_p1 // 2) & (cur > 0)
            index2 = (cur > size_p1 // 2) & (cur > 0)
            if index1.any():
                stacks[index1, cur[index1] - 1] = i + 1
            assert (
                stacks.max(-1)[1][index2] == (cur[index2] - 1 - size_p1 // 2)
            ).all(), 'pdtsp error'
            if (index2).any():
                stacks[index2, cur[index2] - 1 - size_p1 // 2] = -0.01
        
        assert ((stacks == -0.01).all())
        assert (
            visited_time[:, 1 : size_p1 // 2 + 1] < visited_time[:, size_p1 // 2 + 1 :]
        ).all(), (
            visited_time[:, 1 : size_p1 // 2 + 1] < visited_time[:, size_p1 // 2 + 1 :],
            "deliverying without pick-up",
        )
