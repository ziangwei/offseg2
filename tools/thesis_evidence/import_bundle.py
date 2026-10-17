"""Verify an evidence ZIP and import to a NEW MA/thesis/evidence directory."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile


def import_bundle(archive, destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError('Destination already exists; choose a new directory')
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate ZIP members')
        for item in z.infolist():
            path = PurePosixPath(item.filename)
            if ('\\' in item.filename or ':' in item.filename or path.is_absolute()
                    or '..' in path.parts or stat.S_ISLNK(item.external_attr >> 16)):
                raise ValueError('Unsafe ZIP member: '+item.filename)
        manifest = json.loads(z.read('manifest.json'))
        expected = {r['path']:r for r in manifest['files']}
        if set(names) != set(expected) | {'manifest.json'}:
            raise ValueError('ZIP files do not match manifest')
        # Verify before writing anything. Read streams, not full prediction archives in RAM.
        for name, row in expected.items():
            digest, size = hashlib.sha256(), 0
            with z.open(name) as stream:
                for chunk in iter(lambda:stream.read(1024*1024), b''):
                    digest.update(chunk)
                    size += len(chunk)
            if digest.hexdigest() != row['sha256'] or size != row['bytes']:
                raise ValueError('Corrupt or incomplete material: '+name)
        destination.mkdir(parents=True)
        z.extractall(destination)
    print('Imported:',destination)
    print('Status:',manifest['status'])
    print('Claude starts at:',destination/'handoff/README_FOR_CLAUDE.md')
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive',type=Path)
    parser.add_argument('--destination',type=Path,
                        default=Path(__file__).resolve().parents[3]/'thesis/evidence'/datetime.now().strftime('%Y%m%d_%H%M%S'))
    args=parser.parse_args()
    import_bundle(args.archive,args.destination)
