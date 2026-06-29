import torch

ckpt = torch.load('data/checkpoints/drawspeech.ckpt', map_location='cpu')
new_state = {}
for k, v in ckpt.items():
    new_k = k.replace('phoneme_encoder', 'text_encoder')
    new_k = new_k.replace('detailed_curve_predictor', 'sketch_to_contour_predictor')
    new_state[new_k] = v
torch.save(new_state, 'data/checkpoints/drawspeech_fixed.ckpt')
print('Fixed checkpoint saved as data/checkpoints/drawspeech_fixed.ckpt')
