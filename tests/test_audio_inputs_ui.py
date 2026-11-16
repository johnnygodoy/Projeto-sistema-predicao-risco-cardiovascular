"""MP3 UI regression with declared decoder/ASR doubles; native smoke is separate."""
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'app'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from audio_local import AudioError,Segment,Transcript,validate_wav
from test_audio_local import wav_bytes
from test_audio_inputs import mp3_frame,id3_tag
try:
    from streamlit.testing.v1 import AppTest
except ImportError:
    AppTest=None
PAGE=Path(__file__).resolve().parents[1]/'app/pages/2_Audio_Local.py'

class Upload(io.BytesIO):
    def __init__(self,payload,name):super().__init__(payload);self.name=name
    @property
    def size(self):return len(self.getbuffer())

@unittest.skipIf(AppTest is None,'Streamlit not installed')
class InputPageTests(unittest.TestCase):
    def setUp(self):
        import streamlit as st
        original=st.get_option
        self.payload=id3_tag(b'HIDDEN_MARKER')+mp3_frame()*10
        self.filename='sample.mp3'
        self.pcm=validate_wav(wav_bytes()).pcm
        guards=[patch('streamlit.get_option',side_effect=lambda key:'127.0.0.1' if key=='server.address' else original(key)),
                patch('audio_local.local_asr_setup_present',return_value=True),
                patch('streamlit.sidebar.file_uploader',side_effect=lambda *a,**kw:Upload(self.payload,self.filename) if self.payload is not None else None),
                patch('audio_inputs._decode_mp3',return_value=self.pcm)]
        for guard in guards:guard.start();self.addCleanup(guard.stop)
    def app(self):
        app=AppTest.from_file(str(PAGE),default_timeout=10).run();self.assertFalse(app.exception);return app
    def validate(self,app):
        app.button(key='_audio_validate').click().run();self.assertFalse(app.exception);return app
    def transcript(self,app,empty=False):
        sample=app.session_state['_audio_preview'][1]
        result=Transcript(() if empty else (Segment(0,.9,'Test speech.'),),'en',sample.fingerprint,vad_seconds=0 if empty else .9)
        def asr(payload,*args,**kwargs):
            self.assertTrue(payload.startswith(b'RIFF'))
            self.assertEqual(payload,sample.payload)
            self.assertNotEqual(payload,self.payload)
            return result
        with patch('audio_local.transcribe_local',side_effect=asr) as called:
            app.button(key='_audio_run').click().run();called.assert_called_once()
        self.assertFalse(app.exception)
        return app
    def test_mp3_only_normalized_audio_is_played_and_sent_to_asr(self):
        app=self.app();self.assertEqual(len(app.get('audio')),0)
        self.validate(app);self.transcript(app)
        self.assertEqual(len(app.get('audio')),1)
        self.assertTrue(any('MP3 convertido localmente' in x.value for x in app.caption))
        self.assertNotIn('HIDDEN_MARKER',' '.join(str(x.value) for x in app.caption))
    def test_invalid_mp3_and_missing_ffmpeg_never_show_player_or_result(self):
        self.payload=b'not MP3'
        app=self.validate(self.app())
        self.assertEqual(len(app.error),1);self.assertEqual(len(app.get('audio')),0)
        self.payload=mp3_frame()*10;app.run()
        with patch('audio_inputs._decode_mp3',side_effect=AudioError('MP3 precisa do FFmpeg local disponível no PATH.')):
            self.validate(app)
        self.assertEqual(len(app.get('audio')),0)
        self.assertFalse('_audio_result' in app.session_state)
    def test_silence_and_no_speech_not_fabricated(self):
        app=self.app()
        with patch('audio_inputs._decode_mp3',return_value=bytes(32000)):
            self.validate(app)
        self.assertTrue(any('silêncio digital' in x.value for x in app.info))
        self.assertFalse(any(x.key=='_audio_run' for x in app.button))
        self.validate(app);self.transcript(app,empty=True)
        self.assertTrue(any('Nenhum trecho de fala detectado' in x.value for x in app.info))
        self.assertEqual(len(app.dataframe),0)
    def test_replacement_extension_removal_revalidation_and_reset_clear_old_results(self):
        app=self.validate(self.app());self.transcript(app)
        self.filename='same-bytes.wav';app.run()
        self.assertFalse('_audio_preview' in app.session_state);self.assertFalse('_audio_result' in app.session_state)
        self.filename='sample.mp3';app.run();self.validate(app);self.transcript(app)
        self.payload=None;app.run();self.assertFalse('_audio_preview' in app.session_state)
        self.payload=mp3_frame()*11;app.run();self.validate(app);self.transcript(app)
        self.validate(app);self.assertFalse('_audio_result' in app.session_state)
        self.transcript(app);before=app.session_state['_audio_selection']
        app.button(key='_audio_clear').click().run();self.assertFalse('_audio_preview' in app.session_state)
        self.validate(app);self.assertNotEqual(app.session_state['_audio_selection'],before)
    def test_inflight_normalization_success_and_error_are_discarded(self):
        import streamlit as st
        app=self.app()
        def stale_success(*args):
            st.session_state['_audio_selection']='newer-selection'
            return validate_wav(wav_bytes())
        with patch('audio_inputs.prepare_audio',side_effect=stale_success):self.validate(app)
        self.assertFalse('_audio_preview' in app.session_state);self.assertEqual(len(app.get('audio')),0)
        def stale_error(*args):
            st.session_state['_audio_selection']='newer-selection'
            raise AudioError('Old conversion failed')
        with patch('audio_inputs.prepare_audio',side_effect=stale_error):self.validate(app)
        self.assertFalse('_audio_error' in app.session_state);self.assertEqual(len(app.get('audio')),0)
    def test_inflight_asr_success_and_error_are_discarded(self):
        import streamlit as st
        app=self.validate(self.app())
        def stale_success(*args,**kw):
            st.session_state['_audio_selection']='newer-selection'
            return Transcript((Segment(0,1,'Old text'),),'en','old')
        with patch('audio_local.transcribe_local',side_effect=stale_success):app.button(key='_audio_run').click().run()
        self.assertFalse('_audio_result' in app.session_state)
        app.run();self.validate(app)
        def stale_error(*args,**kw):
            st.session_state['_audio_selection']='newer-selection'
            raise AudioError('Old ASR failed')
        with patch('audio_local.transcribe_local',side_effect=stale_error):app.button(key='_audio_run').click().run()
        self.assertFalse('_audio_error' in app.session_state);self.assertFalse(app.exception)

if __name__=='__main__':unittest.main()
