# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# Copyright (c) 2022 Snap Inc.
# 
# All rights reserved.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Implements the knowledge distillation loss
"""
import torch
from torch.nn import functional as F


class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """

    def __init__(self, daily_criterion: torch.nn.Module, gender_criterion: torch.nn.Module, embel_criterion: torch.nn.Module,
                 teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float):
        super().__init__()
        
        self.daily_criterion = daily_criterion
        self.gender_criterion = gender_criterion
        self.embel_criterion = embel_criterion
        
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            outputs, outputs_kd = outputs
            
        base_loss_daily = self.daily_criterion(outputs[0], labels['daily_label'])
        base_loss_gender = self.gender_criterion(outputs[1], labels['gender_label'])
        base_loss_embel = self.embel_criterion(outputs[2], labels['embel_label'])
        
        if self.distillation_type == 'none':
            return base_loss_daily, base_loss_gender, base_loss_embel

        if outputs_kd is None:
            raise ValueError("When knowledge distillation is enabled, the model is "
                             "expected to return a Tuple[Tensor, Tensor] with the output of the "
                             "class_token and the dist_token")
        # don't backprop throught the teacher
        with torch.no_grad():
            teacher_outputs_daily, teacher_outputs_gender, teacher_outputs_embel = self.teacher_model(inputs)

        if self.distillation_type == 'soft':
            T = self.tau
            # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # with slight modifications
            distillation_loss_daily = F.kl_div(
                F.log_softmax(outputs_kd[0] / T, dim=1),
                F.log_softmax(teacher_outputs_daily / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs_kd[0].numel()
            distillation_loss_gender = F.kl_div(
                F.log_softmax(outputs_kd[1] / T, dim=1),
                F.log_softmax(teacher_outputs_gender / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs_kd[1].numel()
            distillation_loss_embel = F.kl_div(
                F.log_softmax(outputs_kd[2] / T, dim=1),
                F.log_softmax(teacher_outputs_embel / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs_kd[2].numel()
            
        elif self.distillation_type == 'hard':
            distillation_loss_daily = F.cross_entropy(
                outputs_kd[0], teacher_outputs_daily.argmax(dim=1))
            distillation_loss_gender = F.cross_entropy(
                outputs_kd[1], teacher_outputs_gender.argmax(dim=1))
            distillation_loss_embel = F.cross_entropy(
                outputs_kd[2], teacher_outputs_embel.argmax(dim=1))

        loss_daily = base_loss_daily * (1 - self.alpha) + distillation_loss_daily * self.alpha
        loss_gender = base_loss_gender * (1 - self.alpha) + distillation_loss_gender * self.alpha
        loss_embel = base_loss_embel * (1 - self.alpha) + distillation_loss_embel * self.alpha
        
        return loss_daily, loss_gender, loss_embel
