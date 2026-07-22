import os
import sys
import importlib
import platform as _platform

_verbose = os.environ.get('PYFASTNEM_VERBOSE', '0') == '1'


def _log(msg: str) -> None:
    if _verbose:
        print(msg)


def _load_core() -> None:
    _machine = _platform.machine()
    _system = _platform.system()

    if _machine in ('x86_64', 'AMD64'):
        _cpu = importlib.import_module('pyfastnem._cpu')
        if _cpu.has_avx2():
            _log("use pyfastnem[avx2]")
            mod = importlib.import_module('pyfastnem._core_avx2')
        elif _cpu.has_sse2():
            _log("use pyfastnem[sse2]")
            mod = importlib.import_module('pyfastnem._core_sse2')
        else:
            _log("use pyfastnem[scalar]")
            mod = importlib.import_module('pyfastnem._core_scalar')

    elif _machine in ('aarch64', 'arm64', 'ARM64'):
        if _system == 'Darwin':
            _log("use pyfastnem[apple]")
            mod = importlib.import_module('pyfastnem._core_darwin_arm')
        else:
            _cpu = importlib.import_module('pyfastnem._cpu')
            if _cpu.has_sve():
                _log("use pyfastnem[sve]")
                mod = importlib.import_module('pyfastnem._core_arm8_sve')
            else:
                _log("use pyfastnem[arm8]")
                mod = importlib.import_module('pyfastnem._core_arm8')

    else:
        _log("use pyfastnem[scalar]")
        mod = importlib.import_module('pyfastnem._core_scalar')

    # Register as pyfastnem._core so ppanggolin.py's `from pyfastnem import _core` resolves.
    sys.modules[__name__ + '._core'] = mod


_load_core()

from pyfastnem.ppanggolin import partition_pangenome  # noqa: E402

__all__ = ["partition_pangenome"]
