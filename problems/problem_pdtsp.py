from typing import Dict
import torch

from .problem_pdp import PDP


class PDTSP(PDP):
    def __init__(
        self, p_size: int, init_val_met: str = 'p2d', with_assert: bool = False
    ) -> None:
        super().__init__(p_size, init_val_met, with_assert)
        
        self.name = 'pdtsp'  # Pickup and Delivery TSP

        print(
            f'PDTSP with {self.size} nodes.',
            ' Do assert:',
            with_assert,
        )

    def get_initial_solutions(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:

        batch_size = batch['coordinates'].size(0)

        def get_solution(methods: str) -> torch.Tensor:

            half_size = self.size // 2

            if methods == 'random':
                candidates = torch.ones(batch_size, self.size + 1).bool()
                candidates[:, half_size + 1 :] = 0
                rec = torch.zeros(batch_size, self.size + 1).long()
                selected_node = torch.zeros(batch_size, 1).long()
                candidates.scatter_(1, selected_node, 0)

                for i in range(self.size):
                    dists = torch.ones(batch_size, self.size + 1)
                    dists.scatter_(1, selected_node, -1e20)
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

                for i in range(self.size):

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
                    next_selected_node = dists.min(-1)[1].view(-1, 1)

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

        mask = visited_order_map.clone()
        mask[torch.arange(bs), selected_node.view(-1)] = True
        mask[torch.arange(bs), selected_node.view(-1) + gs // 2] = True
        mask[torch.arange(bs), :, selected_node.view(-1)] = True
        mask[torch.arange(bs), :, selected_node.view(-1) + gs // 2] = True

        return mask

    def _check_feasibility(self, rec: torch.Tensor) -> None:

        p_size = self.size

        assert (
            (torch.arange(p_size + 1, out=rec.new())).view(1, -1).expand_as(rec)
            == rec.sort(1)[0]
        ).all(), (
            (
                (torch.arange(p_size + 1, out=rec.new())).view(1, -1).expand_as(rec)
                == rec.sort(1)[0]
            ),
            "not visiting all nodes",
            rec,
        )

        # calculate visited time
        bs = rec.size(0)
        visited_time = torch.zeros((bs, p_size), device=rec.device)
        pre = torch.zeros((bs), device=rec.device).long()
        for i in range(p_size):
            visited_time[torch.arange(bs), rec[torch.arange(bs), pre] - 1] = i + 1
            pre = rec[torch.arange(bs), pre]

        assert (
            visited_time[:, 0 : p_size // 2] < visited_time[:, p_size // 2 :]
        ).all(), (
            visited_time[:, 0 : p_size // 2] < visited_time[:, p_size // 2 :],
            "deliverying without pick-up",
        )
