from typing import Dict
import torch

from .problem_pdp import PDP


class PDTSP(PDP):
    def __init__(
        self, size: int, init_val_method: str, check_feasible: bool = False
    ) -> None:
        super().__init__(size, init_val_method, check_feasible)

        self.name = 'pdtsp'  # Pickup and Delivery TSP

        print(
            f'PDTSP with {self.size} nodes.',
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

        mask = visited_order_map.clone()  # true means unavailable
        arange = torch.arange(batch_size)
        mask[arange, selected_node.view(-1)] = True
        mask[arange, selected_node.view(-1) + graph_size_plus1 // 2] = True
        mask[arange, :, selected_node.view(-1)] = True
        mask[arange, :, selected_node.view(-1) + graph_size_plus1 // 2] = True

        return mask

    def get_initial_solutions(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:

        batch_size = batch['coordinates'].size(0)

        def get_solution(methods: str) -> torch.Tensor:

            half_size = self.size // 2

            if methods == 'random':
                candidates = torch.ones(batch_size, self.size + 1).bool()  # all Ture
                candidates[:, half_size + 1 :] = 0  # set to False
                solution = torch.zeros(batch_size, self.size + 1).long()
                selected_node = torch.zeros(batch_size, 1).long()
                candidates.scatter_(1, selected_node, 0)  # set to False

                for _ in range(self.size):
                    dists: torch.Tensor = torch.ones(batch_size, self.size + 1)
                    dists[~candidates] = -1e20
                    dists = torch.softmax(dists, -1)
                    next_selected_node = dists.multinomial(1).view(-1, 1)

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

            elif methods == 'greedy':
                candidates = torch.ones(batch_size, self.size + 1).bool()
                candidates[:, half_size + 1 :] = 0
                solution = torch.zeros(batch_size, self.size + 1).long()
                selected_node = torch.zeros(batch_size, 1).long()
                candidates.scatter_(1, selected_node, 0)

                for _ in range(self.size):
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
                    d2 = batch['coordinates'].cpu()  # (batch_size, graph_size+1, 2)

                    dists = (d1 - d2).norm(p=2, dim=2)  # (batch_size, graph_size+1)
                    dists[~candidates] = 1e6
                    next_selected_node = dists.min(-1)[1].view(-1, 1)

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
        assert (
            (torch.arange(self.size + 1, out=solution.new()))
            .view(1, -1)
            .expand_as(solution)
            == solution.sort(1)[0]
        ).all(), (
            (
                (torch.arange(self.size + 1, out=solution.new()))
                .view(1, -1)
                .expand_as(solution)
                == solution.sort(1)[0]
            ),
            "not visiting all nodes",
            solution,
        )

        # calculate visited time
        batch_size = solution.size(0)
        visited_time = torch.zeros((batch_size, self.size), device=solution.device)
        pre = torch.zeros(batch_size, device=solution.device).long()
        for i in range(self.size):
            visited_time[
                torch.arange(batch_size), solution[torch.arange(batch_size), pre] - 1
            ] = (i + 1)
            pre = solution[torch.arange(batch_size), pre]

        assert (
            visited_time[:, 0 : self.size // 2] < visited_time[:, self.size // 2 :]
        ).all(), (
            visited_time[:, 0 : self.size // 2] < visited_time[:, self.size // 2 :],
            "deliverying without pick-up",
        )
