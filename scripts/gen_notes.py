import importlib
import os
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent.resolve()
os.environ['PYTHONPATH'] = str(root_dir)
sys.path.insert(0, str(root_dir))

import csv
import numpy as np
import librosa
import soundfile as sf

from utils.hparams import set_hparams, hparams
from modules.pe import initialize_pe

WAV_CANDIDATE_EXTENSIONS = ['.wav', '.flac']


def quantize_to_note(hz):
    midi = int(round(librosa.hz_to_midi(hz)))
    midi = max(0, min(127, midi))
    return librosa.midi_to_note(midi, unicode=False)


def gen_notes_for_dataset(raw_data_dir: Path, pe):
    csv_path = raw_data_dir / 'transcriptions.csv'
    if not csv_path.exists():
        print(f'| Skip (no transcriptions.csv): {raw_data_dir}')
        return

    with open(csv_path, 'r', encoding='utf8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    has_notes = 'note_seq' in fieldnames and 'note_dur' in fieldnames
    if has_notes:
        print(f'| Already has note_seq/note_dur, skipping: {csv_path}')
        return

    sr = hparams['audio_sample_rate']
    hop_size = hparams['hop_size']
    timestep = hop_size / sr
    f0_min = hparams['f0_min']
    f0_max = hparams['f0_max']

    out_rows = []
    for row in rows:
        name = row['name']
        ph_dur = [float(x) for x in row['ph_dur'].split()]
        ph_seq = row['ph_seq'].split()

        wav_fn = None
        for ext in WAV_CANDIDATE_EXTENSIONS:
            cand = raw_data_dir / 'wavs' / f'{name}{ext}'
            if cand.exists():
                wav_fn = cand
                break
        if wav_fn is None:
            raise FileNotFoundError(f'Waveform not found for item \'{name}\' in {raw_data_dir}')

        waveform, _ = librosa.load(str(wav_fn), sr=sr, mono=True)
        seconds = sum(ph_dur)
        length = round(seconds / timestep)
        f0, uv = pe.get_pitch(
            waveform, samplerate=sr, length=length,
            hop_size=hop_size, f0_min=f0_min, f0_max=f0_max, interp_uv=True
        )
        f0 = np.asarray(f0, dtype=np.float32)
        uv = np.asarray(uv, dtype=bool)

        ph_acc = np.round(np.cumsum(ph_dur) / timestep + 0.5).astype(np.int64)
        ph_acc = np.concatenate([[0], ph_acc])
        L = len(f0)

        note_seq = []
        note_dur = []
        for i in range(len(ph_seq)):
            s = int(min(ph_acc[i], L - 1))
            e = int(min(max(ph_acc[i + 1], s + 1), L))
            seg_f0 = f0[s:e]
            seg_uv = uv[s:e]
            voiced = seg_f0[~seg_uv] if (~seg_uv).any() else seg_f0
            voiced = voiced[voiced > 0]
            if voiced.size == 0:
                note_seq.append('rest')
            else:
                note_seq.append(quantize_to_note(float(np.mean(voiced))))
            note_dur.append(ph_dur[i])

        if not any(n != 'rest' for n in note_seq):
            print(f'| WARNING: all notes are rest for \'{name}\', forcing first note to C4')
            note_seq[0] = 'C4'

        row['note_seq'] = ' '.join(note_seq)
        row['note_dur'] = ' '.join(f'{d:.6f}' for d in note_dur)
        out_rows.append(row)

    new_fieldnames = list(fieldnames) + ['note_seq', 'note_dur']
    for r in out_rows:
        for fn in new_fieldnames:
            r.setdefault(fn, '')

    with open(csv_path, 'w', encoding='utf8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f'| Wrote note_seq/note_dur for {len(out_rows)} items: {csv_path}')


def main():
    set_hparams()
    pe = initialize_pe()
    for ds in hparams['datasets']:
        raw_data_dir = Path(ds['raw_data_dir'])
        gen_notes_for_dataset(raw_data_dir, pe)
    print('| Done.')


if __name__ == '__main__':
    main()
