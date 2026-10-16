"""Audit and prepare an explicitly described, single-speaker fr-CI pilot."""
import csv
import hashlib
import json
import math
from pathlib import Path
from collections import Counter, defaultdict
import unicodedata
from dataclasses import asdict
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch
from mandia_voice import Config, tokenize, to_mel


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    if str(path).startswith('/content/drive/MyDrive/') and not Path('/content/drive/MyDrive/IvoireTTS_v01/pipeline_state.json').is_file():
        raise OSError('Google Drive disconnected; metadata save stopped without local fallback')
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf8')
    temporary.replace(path)


def read_wave(path, config=Config()):
    audio, rate = sf.read(path, dtype='float32', always_2d=True)
    # Do not silently average potentially overlapping speakers in stereo.
    if audio.shape[1] != 1:
        raise ValueError('Mono required: inspect channels before conversion')
    wave = audio[:, 0]
    if rate != config.sample_rate:
        divisor = math.gcd(rate, config.sample_rate)
        wave = resample_poly(wave, config.sample_rate//divisor, rate//divisor).astype('float32')
    return wave


def audit(root, config=Config()):
    root = Path(root).resolve()
    errors, entries = [], []
    manifest, metadata = root/'dataset.json', root/'metadata.csv'
    if not manifest.exists() or not metadata.exists():
        return {'pilot_ready': False, 'errors': ['dataset.json or metadata.csv missing'], 'rows': 0}
    declaration = json.loads(manifest.read_text())
    if declaration.get('locale') != 'fr-CI' or not declaration.get('locale_evidence'):
        errors.append('Missing documented Ivorian French provenance')
    for key in ['training_allowed', 'voice_synthesis_allowed']:
        if declaration.get(key) is not True:
            errors.append('Permission not documented: '+key)
    for key in ['source', 'license_or_agreement', 'permission_evidence']:
        if not declaration.get(key):
            errors.append('Missing dataset field: '+key)
    rows = list(csv.DictReader(metadata.open(encoding='utf-8-sig')))
    reserved_file = Path(__file__).parent/'phrases-test-reservees.csv'
    reserved = set()
    if reserved_file.exists():
        for row in csv.DictReader(reserved_file.open(encoding='utf8')):
            reserved.add(unicodedata.normalize('NFC', ' '.join(row['text'].casefold().split())))
    ids, audio_splits, text_splits, sessions = set(), defaultdict(set), defaultdict(set), defaultdict(set)
    speaker = set()
    for line, row in enumerate(rows, 2):
        try:
            uid, text = row['id'], row['text'].strip()
            split = row['split']
            if split not in ['train', 'validation', 'test']:
                raise ValueError('Invalid split')
            if not uid or uid in ids:
                raise ValueError('Empty or duplicate id')
            ids.add(uid)
            if not row['speaker_id'] or not row['session_id']:
                raise ValueError('Speaker and recording session required')
            if row.get('transcript_checked') != 'yes':
                raise ValueError('Transcript must be checked against the recording')
            path = (root/row['audio']).resolve()
            if not path.is_relative_to(root):
                raise ValueError('Audio path outside corpus directory')
            info = sf.info(path)
            if info.channels != 1 or info.samplerate < 24000:
                raise ValueError('Require original mono >=24kHz (upsampling is not new detail)')
            wave = read_wave(path, config)
            seconds = len(wave)/config.sample_rate
            if not 1 <= seconds <= 20 or not np.isfinite(wave).all():
                raise ValueError('Require finite audio lasting 1–20 seconds')
            clipping = float((np.abs(wave) >= .999).mean())
            rms = float(np.sqrt(np.mean(wave**2)))
            if clipping > .005 or rms < .002:
                raise ValueError('Excess clipping or nearly silent audio')
            tokens = tokenize(text)
            if len(tokens) > 500 or len(tokens) > len(wave)//config.hop+1:
                raise ValueError('Text too long for alignment budget')
            digest = sha(path)
            normalized_text = unicodedata.normalize('NFC', ' '.join(text.casefold().split()))
            if split == 'train' and normalized_text in reserved:
                raise ValueError('Reserved evaluation prompt found in training')
            audio_splits[digest].add(split)
            text_splits[normalized_text].add(split)
            sessions[row['session_id']].add(split)
            speaker.add(row['speaker_id'])
            entries.append(dict(row, audio_sha256=digest, seconds=seconds,
                                original_rate=info.samplerate, clipping=clipping, rms=rms))
        except (ValueError, KeyError, RuntimeError, OSError) as error:
            errors.append(f'Line {line}: {error}')
    if len(speaker) != 1:
        errors.append('This first pilot requires exactly one documented speaker')
    if len(entries) != len(audio_splits):
        errors.append('Repeated identical audio must be deduplicated before counting corpus hours')
    for label, groups in [('audio', audio_splits), ('text', text_splits), ('session', sessions)]:
        if any(len(splits) > 1 for splits in groups.values()):
            errors.append('Cross-split leakage: '+label)
    counts = Counter(x['split'] for x in entries)
    seconds = {split: sum(x['seconds'] for x in entries if x['split']==split)
               for split in ['train', 'validation', 'test']}
    # Operational pilot thresholds, not a scientific guarantee of useful TTS.
    if counts['train'] < 100 or seconds['train'] < 1200:
        errors.append('Pilot needs >=100 training clips and >=20 training minutes')
    if counts['validation'] < 10 or counts['test'] < 10:
        errors.append('Reserve >=10 validation and >=10 test clips, distinct texts/sessions')
    feature_config = {key: asdict(config)[key] for key in ['mel','sample_rate','n_fft','hop']}
    payload = {'config': feature_config, 'reserved_prompts_sha256': sha(reserved_file) if reserved_file.exists() else None,
               'declaration_sha256': sha(manifest), 'metadata_sha256': sha(metadata),
               'audio': [(x['id'],x['audio_sha256']) for x in entries]}
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    result = {'pilot_ready': not errors, 'errors': errors, 'rows': len(entries), 'feature_config': feature_config,
              'counts': dict(counts), 'seconds': seconds, 'fingerprint': fingerprint,
              'thresholds_are_operational_not_quality_guarantees': True,
              'speaker_generalization_evaluated': False,
              'accent_not_inferred_from_country_or_by_a_classifier': True}
    write_json(root/'audit.json', result)
    if not errors:
        write_json(root/'accepted.json', entries)
    return result


def prepare(root, config=Config()):
    root = Path(root)
    report = audit(root, config)
    if not report['pilot_ready']:
        raise ValueError('Corpus not ready; inspect audit.json: '+ '; '.join(report['errors'][:5]))
    rows = json.loads((root/'accepted.json').read_text())
    cache = root/'features'/report['fingerprint']
    cache.mkdir(parents=True, exist_ok=True)
    total, square, frames = torch.zeros(config.mel), torch.zeros(config.mel), 0
    for index, row in enumerate(rows):
        wave = torch.from_numpy(read_wave(root/row['audio'], config))
        mel = to_mel(wave, config)
        if row['split'] == 'train':
            total += mel.sum(0); square += mel.square().sum(0); frames += len(mel)
        # Sequential ids avoid accepting paths from arbitrary clip identifiers.
        destination = cache/f'{index:06d}.pt'
        temporary = destination.with_suffix('.tmp')
        torch.save({'wave': wave, 'mel': mel, 'tokens': torch.tensor(tokenize(row['text']))}, temporary)
        temporary.replace(destination)
        row['feature'] = str(destination.relative_to(root))
        row['feature_sha256'] = sha(destination)
    mean = total/frames
    std = (square/frames-mean.square()).clamp_min(1e-4).sqrt()
    normalization = {'mean': mean.tolist(), 'std': std.tolist(), 'train_frames': frames}
    write_json(cache/'normalization.json', normalization)
    write_json(cache/'rows.json', rows)
    write_json(cache/'source.json', report)
    return cache
