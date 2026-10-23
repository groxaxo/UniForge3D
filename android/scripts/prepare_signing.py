#!/usr/bin/env python3
"""Create a private upgrade identity once; never replace an existing identity."""
import argparse
import os
from pathlib import Path
import secrets
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--keytool', required=True)
args = parser.parse_args()
folder = Path.home() / '.local/share/uniforge3d/android'
folder.mkdir(parents=True, exist_ok=True)
folder.chmod(0o700)
store, properties = folder/'release.jks', folder/'signing.properties'
if store.exists() and properties.exists():
    print('Reusing private UniForge Android signing identity')
elif store.exists() or properties.exists():
    raise SystemExit('Incomplete signing identity: preserve the existing files and recover the missing part.')
else:
    password = secrets.token_urlsafe(36)
    result = subprocess.run([args.keytool, '-genkeypair', '-keystore', str(store),
        '-alias', 'uniforge-studio', '-keyalg', 'RSA', '-keysize', '3072', '-validity', '10000',
        '-dname', 'CN=UniForge Studio,O=Local,C=NZ',
        '-storepass:env', 'UNIFORGE_STORE_PASSWORD', '-keypass:env', 'UNIFORGE_STORE_PASSWORD'],
        env={**os.environ, 'UNIFORGE_STORE_PASSWORD': password}, capture_output=True)
    if result.returncode:
        raise SystemExit('Android signing identity creation failed; no credential is printed.')
    store.chmod(0o600)
    with properties.open('x') as handle:
        os.chmod(properties, 0o600)
        handle.write(f'storeFile={store}\nstorePassword={password}\nkeyAlias=uniforge-studio\nkeyPassword={password}\n')
    print('Created private UniForge Android signing identity')
