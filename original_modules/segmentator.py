#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import imp
import torch
import torch.nn as nn
import torch.nn.functional as F
from original_modules.postprocess.CRF import CRF


class Segmentator(nn.Module):
  def __init__(self, ARCH, nclasses, path_append="", strict=False):
    super().__init__()
    self.ARCH = ARCH
    self.nclasses = nclasses
    self.path_append = path_append
    self.strict = False
    self.input_H = self.ARCH["dataset"]["sensor"]["img_prop"]["height"]
    self.input_W = self.ARCH["dataset"]["sensor"]["img_prop"]["width"]

    # get the model
    bboneModule = imp.load_source("bboneModule","original_modules/backbone/darknet.py")
    self.backbone = bboneModule.Backbone(params=self.ARCH["backbone"])

    # do a pass of the backbone to initialize the skip connections
    stub = torch.zeros((1,self.backbone.get_input_depth(),self.input_H,self.input_W))

    if torch.cuda.is_available():
      stub = stub.cuda()
      self.backbone.cuda()
    _, stub_skips = self.backbone(stub)

    decoderModule = imp.load_source("decoderModule","original_modules/decoder/darknet.py")
    self.decoder = decoderModule.Decoder(params=self.ARCH["decoder"],
                                         stub_skips=stub_skips,
                                         OS=self.ARCH["backbone"]["OS"],
                                         feature_depth=self.backbone.get_last_depth())

    self.head = nn.Sequential(nn.Dropout2d(p=ARCH["head"]["dropout"]),
                              nn.Conv2d(self.decoder.get_last_depth(),
                                        self.nclasses, kernel_size=3,
                                        stride=1, padding=1))

    if self.ARCH["post"]["CRF"]["use"]:
      self.CRF = CRF(self.ARCH["post"]["CRF"]["params"], self.nclasses)
    else:
      self.CRF = None

    # train backbone?
    if not self.ARCH["backbone"]["train"]:
      for w in self.backbone.parameters():
        w.requires_grad = False

    # train decoder?
    if not self.ARCH["decoder"]["train"]:
      for w in self.decoder.parameters():
        w.requires_grad = False

    # train head?
    if not self.ARCH["head"]["train"]:
      for w in self.head.parameters():
        w.requires_grad = False

    # train CRF?
    if self.CRF and not self.ARCH["post"]["CRF"]["train"]:
      for w in self.CRF.parameters():
        w.requires_grad = False

    # print number of parameters and the ones requiring gradients
    # print number of parameters and the ones requiring gradients
    weights_total = sum(p.numel() for p in self.parameters())
    weights_grad = sum(p.numel() for p in self.parameters() if p.requires_grad)
    print("Total number of parameters: ", weights_total)
    print("Total number of parameters requires_grad: ", weights_grad)

    # breakdown by layer
    weights_enc = sum(p.numel() for p in self.backbone.parameters())
    weights_dec = sum(p.numel() for p in self.decoder.parameters())
    weights_head = sum(p.numel() for p in self.head.parameters())
    print("Param encoder ", weights_enc)
    print("Param decoder ", weights_dec)
    print("Param head ", weights_head)
    if self.CRF:
      weights_crf = sum(p.numel() for p in self.CRF.parameters())
      print("Param CRF ", weights_crf)


  def forward(self, x, mask=None):
    y, skips = self.backbone(x)
    y = self.decoder(y, skips)
    y = self.head(y)
    y = F.softmax(y, dim=1)
    if self.CRF:
      assert(mask is not None)
      y = self.CRF(x, y, mask)
    return y

  def save_checkpoint(self, suffix=""):
    # Save the weights
    modelname='/Range_Net_Test.pth'
    torch.save(self.backbone.state_dict(), 'original_modules/saved_models/backbone' + suffix+modelname)
    torch.save(self.decoder.state_dict(), 'original_modules/saved_models/segmentation_decoder' + suffix+modelname)
    torch.save(self.head.state_dict(), 'original_modules/saved_models/segmentation_head' + suffix+modelname)
    if self.CRF:
      torch.save(self.CRF.state_dict(), 'original_modules/saved_models/segmentation_CRF' + suffix+modelname)

