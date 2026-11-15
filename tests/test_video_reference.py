"""Referência local: validação de configuração/bytes, sem mídia humana."""
from pathlib import Path
import os,sys,tempfile,unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'app'))
import video_movimento_local as video
from test_video_movimento_local import MP4_HEADER


class ReferenceTests(unittest.TestCase):
    def test_default_is_ignored_project_relative(self):
        with patch.dict(os.environ,{'FIAP_VIDEO_REFERENCE_PATH':''}):
            self.assertEqual(video.local_reference_path().parts[-3:],('models','video-reference','referencia-ombro-direito.mp4'))
        ignore=(Path(__file__).resolve().parents[1]/'.gitignore').read_text().splitlines()
        self.assertIn('/models/video-reference/',ignore)
        self.assertNotIn('models/',ignore)

    def test_server_path_rejects_relative_unc_url_and_wrong_suffix(self):
        for raw in ('input.mp4','../input.mp4','https://host/clip.mp4',r'\\server\share\clip.mp4','//host/clip.mp4','/tmp/clip.txt'):
            with self.subTest(raw=raw),patch.dict(os.environ,{'FIAP_VIDEO_REFERENCE_PATH':raw}):
                with self.assertRaises(video.VideoError) as error:video.local_reference_path()
                self.assertNotIn(raw,str(error.exception))

    def test_missing_invalid_and_valid_read(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'reference.mp4'
            with patch.dict(os.environ,{'FIAP_VIDEO_REFERENCE_PATH':str(path)}):
                with self.assertRaisesRegex(video.VideoError,'ausente'):video.read_local_reference()
                path.write_bytes(b'invalid')
                with self.assertRaisesRegex(video.VideoError,'inválida'):video.read_local_reference()
                path.write_bytes(MP4_HEADER)
                self.assertEqual(video.read_local_reference(),MP4_HEADER)

    def test_size_cap_and_read_failure_are_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'reference.mp4';path.write_bytes(MP4_HEADER)
            with patch.dict(os.environ,{'FIAP_VIDEO_REFERENCE_PATH':str(path)}):
                with patch.object(video,'MAX_BYTES',len(MP4_HEADER)-1),patch.object(Path,'open',side_effect=AssertionError('oversize read')):
                    with self.assertRaisesRegex(video.VideoError,'excede'):video.read_local_reference()
                with patch.object(Path,'open',side_effect=PermissionError('private filename')):
                    with self.assertRaisesRegex(video.VideoError,'Não foi possível') as error:video.read_local_reference()
                    self.assertNotIn('private filename',str(error.exception))


if __name__=='__main__':unittest.main()
