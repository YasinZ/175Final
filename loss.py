import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Loss(nn.Module):
    def __init__(self, m):
        super(Loss, self).__init__()
        self.num_anchors = 5
        self.num_classes = 1

        self.sprob = float(m['class_scale'])
        self.sconf = float(m['object_scale'])
        self.snoob = float(m['noobject_scale'])
        self.scoor = float(m['coord_scale'])
        self.H, self.W = 13, 13
        self.B, self.C = 5, 1
        self.HW = Variable(torch.FloatTensor([self.W, self.H]).view(1, 1, 1, 2))
        self.anchors = Variable(torch.FloatTensor(m['anchors']))

        print('YOLOv2-like loss hyper-parameters:'
        print('\tH       = {}'.format(self.H))
        print('\tW       = {}'.format(self.W))
        print('\tbox     = {}'.format(self.B))
        print('\tscales  = {}'.format([self.sprob, self.sconf, self.snoob, self.scoor]))

        self.cuda()
        self.HW = self.HW.cuda()
        self.anchors = self.anchors.cuda()


    def forward(self, output, target):

        _probs = target['probs']
        _confs = target['confs']
        _coord = target['coord']
        _proid = target['proid']
        _areas = target['areas']
        _upleft = target['upleft']
        _botright = target['botright']

        output = output.permute(0, 2, 3, 1).contiguous()
        output_reshape = output.view(-1, self.H, self.W, self.B, (5 + self.C))
        coords = output_reshape[:, :, :, :, :4]
        coords = coords.contiguous().view(-1, self.H * self.W, self.B, 4)
        adjusted_coords_xy = F.sigmoid(coords[:, :, :, 0:2])
        adjusted_coords_wh = torch.sqrt(torch.exp(coords[:, :, :, 2:4]) * self.anchors.view(1, 1, self.B, 2) / self.HW)
        coords = torch.cat([adjusted_coords_xy, adjusted_coords_wh], 3)

        adjusted_c = F.sigmoid(output_reshape[:, :, :, :, 4])
        adjusted_c = adjusted_c.view(-1, self.H * self.W, self.B, 1)

        adjusted_prob = F.softmax(output_reshape[:, :, :, :, 5:], -1)
        adjusted_prob = adjusted_prob.view(-1, self.H * self.W, self.B, self.C)

        adjusted_output = torch.cat([adjusted_coords_xy, adjusted_coords_wh, adjusted_c, adjusted_prob], 3)


        wh = torch.pow(coords[:, :, :, 2:4], 2) * self.HW
        area_pred = wh[:, :, :, 0] * wh[:, :, :, 1]
        centers = coords[:, :, :, 0:2]
        floor = centers - (wh * .5)
        ceil = centers + (wh * .5)

        intersect_upleft = torch.max(floor, _upleft)
        intersect_botright = torch.min(ceil, _botright)
        intersect_wh = intersect_botright - intersect_upleft
        intersect_wh = F.relu(intersect_wh)
        intersect = intersect_wh[:, :, :, 0] * intersect_wh[:, :, :, 1]

        iou = intersect / (_areas + area_pred - intersect)
        best_box = torch.eq(iou, torch.max(iou, 2, keepdim=True)[0]).float()
        confs = best_box * _confs

        conid = self.snoob * (1. - confs) + self.sconf * confs
        weight_coo = torch.cat(4 * [confs.unsqueeze(-1)], 3)
        cooid = self.scoor * weight_coo
        weight_pro = torch.cat(self.C * [confs.unsqueeze(-1)], 3)
        proid = self.sprob * weight_pro

        true = torch.cat([_coord, confs.unsqueeze(3), _probs], 3)
        wght = torch.cat([cooid, conid.unsqueeze(3), proid], 3)

        loss = torch.pow(adjusted_output - true, 2)
        loss = loss * wght
        loss = loss.view(-1, self.H * self.W * self.B * (5 + self.C))
        loss = torch.sum(loss, 1)
        loss = .5 * torch.mean(loss)
        return loss
