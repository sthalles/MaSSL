import torch.nn as nn
import torch


class Criterion(nn.Module):
    def __init__(self):
        super().__init__()

    def cross_entropy(self, p, q):
        # assert inputs.shape == targets.shape
        # assert inputs.requires_grad == True
        # assert targets.requires_grad == False

        p = torch.log_softmax(p, dim=-1)
        q = torch.softmax(q, dim=-1)

        loss = torch.sum(-q * p, dim=-1).mean()
        return loss

    def forward(self, student_output, teacher_output):
        # EPS = torch.finfo(student_output[0].dtype).eps
        consistency = 0
        count = 0
        for i in range(len(student_output)):
            for j in range(len(teacher_output)):
                if i == j:
                    continue
                consistency += self.cross_entropy(student_output[i], teacher_output[j])
                count += 1

        consistency /= count
        return consistency
