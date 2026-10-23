"""Mac Studio queue using only the .51 port-12434 GPUQ API."""
import json
from pathlib import Path
import shutil
from urllib.request import Request, build_opener, HTTPRedirectHandler

from .app import ROOT, Settings, create_app
from .gpuq_common import glb_metadata, sha256_file

PROXY = 'http://100.85.200.51:12434/v1/3d'


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, url):
        return None


class Cuda3090Bridge:
    engine = 'cuda3090'
    encoder = 'Original DINOv3'
    min_views = 1

    def __init__(self, settings, proxy=PROXY):
        self.settings = settings
        self.proxy = proxy.rstrip('/')
        self.http = build_opener(NoRedirect())

    def request(self, path, method='GET', data=None):
        headers = {'x-uniforge-studio': '1'}
        if isinstance(data, dict):
            data = json.dumps(data).encode()
            headers['Content-Type'] = 'application/json'
        request = Request(self.proxy + path, method=method, data=data, headers=headers)
        return self.http.open(request, timeout=45)

    def value(self, path, method='GET', data=None):
        with self.request(path, method, data) as response:
            return json.load(response)

    def health(self):
        return self.value('/health')

    def start(self, identifier, directory, seed):
        inputs = sorted((directory / 'input').glob('*.png'))
        for index, path in enumerate(inputs, 1):
            result = self.value(f'/jobs/{identifier}/images/{index}', 'PUT', path.read_bytes())
            if result['sha256'] != sha256_file(path):
                raise RuntimeError('Reference transfer hash mismatch')
        self.value(f'/jobs/{identifier}', 'POST', {'seed': seed,
            'inputs': {path.name: sha256_file(path) for path in inputs}})

    def read_status(self, identifier):
        return self.value(f'/jobs/{identifier}')

    def cancel(self, identifier):
        return self.value(f'/jobs/{identifier}/cancel', 'POST', {})

    def fetch(self, identifier, directory, metadata):
        temporary = directory / 'prediction.download'
        with self.request(f'/jobs/{identifier}/model') as response, temporary.open('wb') as stream:
            shutil.copyfileobj(response, stream)
        checked = glb_metadata(temporary)
        if checked['sha256'] != metadata['sha256'] or checked['bytes'] != metadata['bytes']:
            temporary.unlink(missing_ok=True)
            raise RuntimeError('The downloaded GLB does not match the GPUQ output')
        receipt = self.value(f'/jobs/{identifier}/receipt')
        (directory / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        temporary.replace(directory / 'prediction.glb')
        return checked


settings = Settings(data=ROOT / 'outputs/studio-3090', engine='cuda3090', port=7864)
app = create_app(settings, Cuda3090Bridge(settings))
