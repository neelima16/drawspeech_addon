import json

with open('mini_demo/features/phoneme_level/metadata.txt') as f:
    lines = [l.strip().split('|') for l in f if l.strip()]

# Phoneme JSON
phoneme_data = []
for parts in lines:
    basename, speaker, phonemes, raw_text = parts
    phoneme_data.append({
        'wav': f'mini_demo/LJSpeech-1.1/wavs/{basename}.wav',
        'transcription': raw_text,
        'phonemes': phonemes,
        'pitch_sketch': f'mini_demo/features/phoneme_level/pitch/LJSpeech-pitch-{basename}.npy',
        'energy_sketch': f'mini_demo/features/phoneme_level/energy/LJSpeech-energy-{basename}.npy',
        'pitch': '',          # prevent loading original pitch
        'energy': '',         # prevent loading original energy
        'duration': None,
        'pitch_length': None,
        'energy_length': None
    })

with open('mini_demo/inference_phoneme.json', 'w') as f:
    json.dump({'data': phoneme_data}, f, indent=2)

# Word JSON
word_data = []
for parts in lines:
    basename, speaker, phonemes, raw_text = parts
    word_data.append({
        'wav': f'mini_demo/LJSpeech-1.1/wavs/{basename}.wav',
        'transcription': raw_text,
        'phonemes': phonemes,
        'pitch_sketch': f'mini_demo/features/word_level/{basename}_word_pitch.npy',
        'energy_sketch': f'mini_demo/features/word_level/{basename}_word_energy.npy',
        'pitch': '',
        'energy': '',
        'duration': None,
        'pitch_length': None,
        'energy_length': None
    })

with open('mini_demo/inference_word.json', 'w') as f:
    json.dump({'data': word_data}, f, indent=2)

print("Created inference JSONs with empty pitch/energy to avoid conflict.")
