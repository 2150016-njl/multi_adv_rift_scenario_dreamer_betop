#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : convert_pretrained_ckpt.py
@Date    : 2023/10/26
"""
import torch
from rift.cbv.recognition.attention_based.attn_model import EncoderModel
from rift.util.run_util import load_config
import os.path as osp
import argparse
from collections import OrderedDict


def recog(model_path, recog_model_path, recog_config):
    checkpoint = torch.load(model_path, map_location="cuda")
    new_state_dict = OrderedDict()
    for key, value in checkpoint["state_dict"].items():
        print("old key:", key)
        cleaned_key = key[len("model."):] if key.startswith("model.") else key
        print("new key:", cleaned_key)
        print("_________________________")
        if cleaned_key.startswith("heads") or cleaned_key.startswith("wp") or cleaned_key == "model.embeddings.position_ids":
            continue
        else:
            new_state_dict[cleaned_key] = value
    checkpoint["state_dict"] = new_state_dict

    torch.save(checkpoint, recog_model_path)
    print("successfully saving the updated state encoder ckpt")
    model = EncoderModel.load_from_checkpoint(recog_model_path, config=recog_config, strict=True)
    for name, param in model.named_parameters():
        print(f"Loaded parameter: {name}, Shape: {param.shape}")


def main(args):
    ROOT_DIR = osp.abspath(osp.dirname(osp.dirname(osp.realpath(__file__))))
    # Ego PlanT model
    model_path = osp.join(ROOT_DIR, 'rift/ego/model_ckpt/PlanT_medium/checkpoints/PlanT_pretrain.ckpt')
    if args.recog:
        recog_config_path = osp.join(ROOT_DIR, 'rift/cbv/recognition/config/attention.yaml')
        recog_config = load_config(recog_config_path)
        recog_model_path = osp.join(ROOT_DIR, recog_config['pretrained_model_path'])  # for the cbv recog
        recog(model_path, recog_model_path, recog_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--recog', action='store_true', default=False)
    args = parser.parse_args()

    main(args)