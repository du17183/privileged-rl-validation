"""Portable deployment paths. This module does not change task or policy settings."""
import json
import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
OPENPI_SOURCE = Path(os.environ.get('OPENPI_SOURCE', str(PROJECT_ROOT / 'third_party/openpi')))
PI05_BASE = Path(os.environ.get('PI05_BASE', str(PROJECT_ROOT / 'external_weights/pi05_base/model.safetensors')))
PI05_TOKENIZER = Path(os.environ.get('PI05_TOKENIZER', str(OPENPI_SOURCE / 'assets/paligemma_tokenizer.model')))
PI05_PYTHON = os.environ.get('PI05_PYTHON', str(PROJECT_ROOT / '.venv_pi05/bin/python'))


def project_path(original):
    for prefix in ['/home/xiaolong/privileged_rl_validation', '/DATA/disk1/home/xiaolong/privileged_rl_validation']:
        if original == prefix or original.startswith(prefix + '/'):
            return str(PROJECT_ROOT / original[len(prefix):].lstrip('/'))
    return original


def dependency_commit(source):
    metadata = Path(source) / 'export_provenance.json'
    if metadata.exists(): return json.loads(metadata.read_text())['commit']
    return subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()


def dependency_status(source):
    metadata = Path(source) / 'export_provenance.json'
    if metadata.exists(): return json.loads(metadata.read_text())['local_changes']
    return subprocess.check_output(['git', '-C', str(source), 'status', '--short'], text=True)
