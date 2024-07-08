import torch.nn as nn
import torch


class MemoryBank(nn.Module):

    def __init__(self, ncrops, K=65536, out_dim=256):
        # create the queue
        super().__init__()
        self.K = K
        self.ncrops = ncrops
        self.register_buffer("queue", torch.randn(out_dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    # utils
    @staticmethod
    @torch.no_grad()
    def concat_all_gather(tensor):
        """
        Performs all_gather operation on the provided tensors.
        *** Warning ***: torch.distributed.all_gather has no gradient.
        """
        tensors_gather = [
            torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

        output = torch.cat(tensors_gather, dim=0)
        return output

    @torch.no_grad()
    def update_queue(self, keys):
        # gather keys before updating queue
        keys = self.concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0, f"{self.K} % {batch_size}"  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def get_features(self):
        return self.queue.clone().detach()

    def forward(self, x, update=False):
        logits = x @ self.queue.clone().detach()
        if update:
            # assume the queue is only updated with the teacher embeddings and there are only two views
            x2 = x.detach().chunk(2)[1]
            # x_ = torch.stack(x_, dim=0).mean(dim=0)
            # x_ = nn.functional.normalize(x_, dim=-1, eps=1e-6)
            self.update_queue(x2)
        return logits
