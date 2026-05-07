import torch
import mindspore as ms
import mindspore.train.serialization as mss
import argparse
import os

def torch_pth_to_mindspore_ckpt(pth_path, ckpt_path):
    state_dict = torch.load(pth_path, map_location='cpu')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    ms_params = []
    for k, v in state_dict.items():
        # 只转换浮点权重
        if hasattr(v, 'numpy'):
            ms_params.append({
                'name': k,
                'data': ms.Tensor(v.cpu().numpy())
            })
    mss.save_checkpoint(ms_params, ckpt_path)
    print(f"[convert] Saved MindSpore ckpt: {ckpt_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert torch .pth to MindSpore .ckpt')
    parser.add_argument('--pth', type=str, required=True, help='Input torch pth file')
    parser.add_argument('--ckpt', type=str, required=True, help='Output mindspore ckpt file')
    args = parser.parse_args()
    torch_pth_to_mindspore_ckpt(args.pth, args.ckpt)
