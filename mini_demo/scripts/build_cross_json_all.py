import json

with open('mini_demo/features/phoneme_level/metadata.txt') as f:
    lines = [l.strip().split('|') for l in f if l.strip()]

data = []
for parts in lines:
    basename, speaker, phonemes, raw_text = parts
    data.append({
        'wav': f'mini_demo/LJSpeech-1.1/wavs/{basename}.wav',
        'transcription': raw_text,
        'phonemes': phonemes,
        'pitch_sketch': f'mini_demo/features/cross_sketches/{basename}_cross_sketch.npy',
        'energy_sketch': f'mini_demo/features/phoneme_level/energy/LJSpeech-energy-{basename}.npy',
        'pitch': '',
        'energy': '',
        'duration': None,
        'pitch_length': None,
        'energy_length': None
    })

with open('mini_demo/inference_cross_all.json', 'w') as f:
    json.dump({'data': data}, f, indent=2)

print("Created mini_demo/inference_cross_all.json with 10 cross-utterance pairs")
