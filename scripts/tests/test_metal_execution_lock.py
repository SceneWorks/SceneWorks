"""CPU-only lease tests: never create the production namespace or execute a model/GPU."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'ci/metal_execution_lock.py'
metal = None
if os.name == 'posix':
    spec = importlib.util.spec_from_file_location('metal_lock_test_module', SOURCE)
    metal = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = metal
    spec.loader.exec_module(metal)


def child(path, seconds=0.4):
    code = '''import importlib.util,sys,time
from pathlib import Path
spec=importlib.util.spec_from_file_location('lock_child',sys.argv[1]);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
with m._acquire(Path(sys.argv[2]),2):
 print('HELD',flush=True)
 time.sleep(float(sys.argv[3]))
'''
    process = subprocess.Popen([sys.executable, '-I', '-c', code, str(SOURCE), str(path), str(seconds)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if process.stdout.readline().strip() != 'HELD':
        raise AssertionError(process.communicate())
    return process


@unittest.skipUnless(os.name == 'posix', 'POSIX flock protocol; Windows uses existing UUID byte locks')
class MetalLease(unittest.TestCase):
    def test_canonical_namespace_is_fixed_and_cli_has_no_path_override(self):
        self.assertEqual(str(metal.CANONICAL_PATH), '/Users/Shared/SceneWorks/gpu-locks/apple-metal.lock')
        self.assertEqual(metal.PROTOCOL, 'sceneworks-host-metal-flock-v1')
        self.assertNotIn("add_argument('--path'", SOURCE.read_text())

    def test_cross_process_contention_waits_then_acquires_same_inode(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary).resolve() / 'metal.lock'
            process = child(path)
            try:
                inode = path.stat().st_ino
                with self.assertRaises(TimeoutError):
                    with metal._acquire(path, 0.08):
                        self.fail('a second process bypassed the physical lease')
                process.communicate(timeout=3)
                self.assertEqual(process.returncode, 0)
                with metal._acquire(path, 1) as lease:
                    self.assertEqual(lease.record()['inode'], inode)
                self.assertEqual(path.stat().st_ino, inode)  # no unlink/replacement on release
            finally:
                process.communicate(timeout=3)

    def test_exception_releases_without_deleting_or_replacing_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary).resolve() / 'metal.lock'
            with self.assertRaisesRegex(RuntimeError, 'fixture failure'):
                with metal._acquire(path, 1):
                    raise RuntimeError('fixture failure')
            inode = path.stat().st_ino
            process = child(path, 0)
            process.communicate(timeout=3)
            self.assertEqual(process.returncode, 0)
            self.assertEqual(path.stat().st_ino, inode)

    def test_reentrant_description_does_not_unlock_outer_lease(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary).resolve() / 'metal.lock'
            with metal._acquire(path, 1) as outer:
                with metal._acquire(path, 1, str(outer.fd)) as inner:
                    self.assertTrue(inner.borrowed)
                    self.assertEqual(inner.record()['inode'], outer.record()['inode'])
                separate = os.open(path, os.O_RDWR)
                try:
                    with self.assertRaises(OSError):
                        with metal._acquire(path, 0.1, str(separate)):
                            self.fail('forged separately-opened descriptor bypassed outer lease')
                finally:
                    os.close(separate)
            with metal._acquire(path, 1):
                pass

    def test_symlink_ancestor_leaf_hardlink_and_invalid_wait_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            real = root / 'real'
            real.mkdir()
            link = root / 'linked'
            link.symlink_to(real, target_is_directory=True)
            with self.assertRaises(ValueError):
                with metal._acquire(link / 'metal.lock', 1):
                    pass
            regular = root / 'regular'
            regular.write_bytes(b'lock')
            leaf = root / 'leaf'
            leaf.symlink_to(regular)
            with self.assertRaises(ValueError):
                with metal._acquire(leaf, 1):
                    pass
            hard = root / 'hard'
            os.link(regular, hard)
            with self.assertRaises(ValueError):
                with metal._acquire(hard, 1):
                    pass
            for wait in (0, -1, float('nan'), float('inf'), True):
                with self.assertRaises(ValueError):
                    with metal._acquire(root / 'valid.lock', wait):
                        pass

    def test_wrapper_waits_for_grandchild_after_root_exit_without_signalling(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            marker = root / 'grandchild-completed.txt'
            inner = "import time,pathlib;time.sleep(.25);pathlib.Path(" + repr(str(marker)) + ").write_text('completed')"
            outer = 'import subprocess,sys;subprocess.Popen([sys.executable,"-I","-c",' + repr(inner) + '])'
            started = time.monotonic()
            with metal._acquire(root / 'metal.lock', 1) as lease:
                status = metal.run_owned([sys.executable, '-I', '-c', outer], lease)
            self.assertEqual(status, 0)
            self.assertGreaterEqual(time.monotonic() - started, 0.2)
            self.assertEqual(marker.read_text(), 'completed')


if __name__ == '__main__':
    unittest.main()
