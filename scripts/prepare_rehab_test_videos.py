"""Prepare only the named REHAB24-6 test clips from hash-verified local originals.

Python standard library + existing FFmpeg/FFprobe. No network, downloads, installs,
Git operations, overwrite, spatial crop, brightness edits or time interpolation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE_RECORD = 'https://zenodo.org/records/13305826'
SOURCES = {
    'PM012': {'file': 'Ex1/PM_012-Camera17-30fps.mp4', 'bytes': 20580711,
              'sha256': '0af4e251ad54c17434ab357b6d2304f24cf9cd4aba624caf69b75c50a196be0e',
              'zip_crc32': '99bec477'},
    'PM039': {'file': 'Ex1/PM_039-Camera17-30fps.mp4', 'bytes': 23698813,
              'sha256': 'becc9eae0729570cba09b4402bbd685ac688c12203fb02c75d313adc88e02f87',
              'zip_crc32': '628c4cbe'},
}
# FFmpeg decoded-frame indices: start inclusive, end exclusive.
CLIPS = (
    ('TESTE-TEC4-01-REFERENCIA-OMBRO-DIREITO-PM012-C17-4p8s.mp4', 'PM012', 891, 1035, 5, 1),
    ('TESTE-TEC4-02-OBSERVADO-OMBRO-DIREITO-PM012-C17-4p8s.mp4', 'PM012', 2071, 2215, 9, 0),
    ('TESTE-TEC4-03-PM039-C17-REP02.mp4', 'PM039', 225, 370, 2, 1),
    ('TESTE-TEC4-04-PM039-C17-REP08.mp4', 'PM039', 1515, 1660, 8, 0),
    ('TESTE-TEC4-05-PM012-C17-REP08.mp4', 'PM012', 1936, 2070, 8, 0),
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_source(path, identity):
    path = Path(path).expanduser()
    if str(path).startswith(('\\\\', '//')):
        raise ValueError('Use a local file, not a network share.')
    expected = SOURCES[identity]
    if not path.is_file() or path.stat().st_size != expected['bytes']:
        raise ValueError(f'{identity}: original file missing or unexpected size.')
    if sha256(path) != expected['sha256']:
        raise ValueError(f'{identity}: SHA-256 mismatch. Nothing will be converted.')
    return path.resolve()


def video_filter(start, end):
    pts = 'PTS-STARTPTS' if (start, end) in ((891, 1035), (2071, 2215)) else 'N/(30*TB)'
    return f'trim=start_frame={start}:end_frame={end},setpts={pts},scale=1280:-2'


def ffmpeg_command(binary, source, output, start, end):
    return [binary, '-nostdin', '-n', '-hide_banner', '-loglevel', 'error',
            '-threads', '2', '-i', str(source), '-map', '0:v:0', '-filter_threads', '1',
            '-vf', video_filter(start, end), '-frames:v', str(end-start), '-r', '30',
            '-fps_mode', 'cfr', '-c:v', 'libx264',
            '-preset', 'fast', '-crf', '18', '-threads', '2', '-pix_fmt', 'yuv420p',
            '-an', '-map_metadata', '-1', '-movflags', '+faststart', str(output)]


def probe_clip(binary, path, frames):
    process = subprocess.run([binary, '-v', 'error', '-count_frames', '-select_streams', 'v:0',
        '-show_entries', 'stream=codec_name,width,height,pix_fmt,avg_frame_rate,nb_read_frames,duration:format=duration,size',
        '-of', 'json', str(path)], check=True, capture_output=True, text=True, timeout=90)
    data = json.loads(process.stdout)
    stream = data['streams'][0]
    if (stream['codec_name'] != 'h264' or stream['width'] != 1280 or stream['height'] != 720
            or stream['pix_fmt'] != 'yuv420p' or stream['avg_frame_rate'] != '30/1'
            or int(stream['nb_read_frames']) != frames
            or abs(float(data['format']['duration']) - frames / 30) > 0.001):
        raise ValueError('Unexpected encoded media properties; do not use this output.')
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pm012', type=Path, help='Original PM_012-Camera17-30fps.mp4')
    parser.add_argument('--pm039', type=Path, help='Original PM_039-Camera17-30fps.mp4, optional')
    parser.add_argument('--output-dir', type=Path, required=True, help='NEW directory outside this repository')
    args = parser.parse_args()
    raw = {key: value for key, value in (('PM012', args.pm012), ('PM039', args.pm039)) if value}
    if not raw:
        parser.error('Provide --pm012 and/or --pm039. Obtain originals from the cited official source.')
    binaries = {name: shutil.which(name) for name in ('ffmpeg', 'ffprobe')}
    if not all(binaries.values()):
        parser.error('FFmpeg and FFprobe must already be installed and available in PATH.')
    paths = {key: validate_source(value, key) for key, value in raw.items()}
    output = args.output_dir.expanduser().resolve()
    if output == ROOT or ROOT in output.parents:
        parser.error('Output must be outside the repository, to avoid accidental media publication.')
    output.mkdir(parents=True, exist_ok=False)
    report = {'source_record': SOURCE_RECORD, 'sources': {key: SOURCES[key] for key in paths},
              'license': 'CC BY-NC 4.0; academic/nonprofit noncommercial research and citation as stated in the record.',
              'frame_index_note': 'Author frame-number base is not explicit. [first,last) is the common interior under zero/one-based interpretations.',
              'transformation': 'Temporal trim, uniform 30fps timestamps, proportion-preserving 1280x720 scale, H264 fast CRF18 yuv420p, no audio. No spatial crop, brightness change, masks or interpolation.',
              'clips': []}
    for name, identity, start, end, repetition, label in CLIPS:
        if identity not in paths:
            continue
        target = output / name
        subprocess.run(ffmpeg_command(binaries['ffmpeg'], paths[identity], target, start, end), check=True, timeout=180)
        metadata = probe_clip(binaries['ffprobe'], target, end - start)
        report['clips'].append({'name': name, 'source': identity, 'repetition': repetition,
            'author_correctness_label': label, 'frames_start_inclusive_end_exclusive': [start, end],
            'duration_seconds': (end - start) / 30, 'bytes': target.stat().st_size,
            'sha256': sha256(target), 'metadata': metadata})
    with (output / 'provenance.json').open('x', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(f'Prepared {len(report["clips"])} clips in: {output}')
    print('Encoder/build differences can change derived SHA-256; the original source hashes must match.')
    reference = output / CLIPS[0][0]
    if reference.exists():
        print('PowerShell, in the terminal that starts Streamlit:')
        print(f'$env:FIAP_VIDEO_REFERENCE_PATH = "{reference}"')


if __name__ == '__main__':
    main()
