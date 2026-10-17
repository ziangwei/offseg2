"""Download one official trained ADE-B checkpoint, not ImageNet backbone weights."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import re
from .common import ROOT, sha256, write_json

OFFICIAL_REPO = 'https://github.com/HVision-NKU/OffSeg'
FOLDER = 'https://drive.google.com/drive/folders/1RpQ1ouZJpiIPQU0nRpl9wGeOVoQ-_ZEy'
DEFAULT = ROOT/'pretrained/thesis/offseg_b_ade_official.pth'


def matches_ade_b(path):
    text = str(path).replace('\\', '/').lower()
    return ('ade' in text and 'offseg' in text and
            bool(re.search(r'(?:offseg[-_ ]?b\b|(?:^|[/_\-])b(?:[/_\-.]|$)|\bbase\b|\bs2\b)', text)) and
            text.endswith(('.pth', '.pt')) and 'parseg' not in text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--file-id', help='Explicit id from the official folder if naming is ambiguous')
    args = parser.parse_args()
    if args.output.exists() and not args.list:
        if not args.output.with_suffix('.source.json').is_file():
            parser.error('Existing file lacks source metadata; choose a new output or provide a local checkpoint to submitter')
        print('Already present:', args.output)
        return
    try:
        import gdown
    except ImportError:
        parser.error('Activate offseg_new2 and run: python -m pip install gdown==5.2.0')
    entries = gdown.download_folder(url=FOLDER, skip_download=True, quiet=True, use_cookies=False)
    if not entries:
        parser.error('Official folder could not be listed; no GPU job was submitted')
    weights = [e for e in entries if str(e.path).lower().endswith(('.pth','.pt'))]
    if args.list:
        for item in entries:
            print(item.id, item.path)
        return
    matches = [e for e in weights if e.id == args.file_id] if args.file_id else [e for e in weights if matches_ade_b(e.path)]
    if len(matches) != 1:
        for item in entries:
            print(item.id, item.path)
        parser.error('Need exactly one official ADE20K OffSeg-B trained weight; rerun with --file-id from listing. Never substitute a backbone or another dataset.')
    entry = matches[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = gdown.download(id=entry.id, output=str(args.output), quiet=False, resume=True, use_cookies=False)
    if not result:
        raise RuntimeError('Official checkpoint download failed')
    write_json(args.output.with_suffix('.source.json'), dict(
        source='official_download', repository=OFFICIAL_REPO, folder=FOLDER,
        drive_file_id=entry.id, official_path=entry.path, sha256=sha256(args.output),
        downloaded_utc=datetime.now(timezone.utc).isoformat(),
        note='Trained official ADE20K/B weight. Re-evaluate its actual mIoU; this is not the deleted local 46.01 run.'))
    print('Checkpoint:', args.output)


if __name__ == '__main__':
    main()
