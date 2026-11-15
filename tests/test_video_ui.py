"""UI com fronteira upload/ML simulada; não valida estimativa de pose humana."""
from contextlib import ExitStack
import io
import os
from pathlib import Path
import sys
import unicodedata
import unittest
from unittest.mock import patch

try:
    from streamlit.testing.v1 import AppTest
except ImportError:
    AppTest = None
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'app'))
import video_movimento_local as video
from test_video_movimento_local import MP4_HEADER,clip


class Upload(io.BytesIO):
    @property
    def size(self):return len(self.getbuffer())


@unittest.skipIf(AppTest is None,'Streamlit/AppTest opcional não instalado.')
class PageTests(unittest.TestCase):
    def setUp(self):
        self.stack=ExitStack()
        self.stack.enter_context(patch.dict(os.environ,{'FIAP_VIDEO_PYTHON':'','FIAP_POSE_MODEL_PATH':'','FIAP_VIDEO_REFERENCE_PATH':''}))
        import streamlit
        original=streamlit.get_option
        self.stack.enter_context(patch('streamlit.get_option',side_effect=lambda name:{'server.address':'127.0.0.1','server.maxUploadSize':24}.get(name,original(name))))
        self.stack.enter_context(patch.object(video,'verify_model'))
        self.stack.enter_context(patch.object(video,'read_local_reference',return_value=MP4_HEADER+b'reference'))
    def tearDown(self):self.stack.close()
    def page(self):
        app=AppTest.from_file(str(ROOT/'app/pages/4_Video_Movimento_Local.py'),default_timeout=10).run()
        self.assertFalse(app.exception);return app
    def upload(self,value=None):
        return patch('streamlit.file_uploader',return_value=Upload(MP4_HEADER+b'observed') if value is None else value)
    def report(self,angle=125,invalid=()):return video.compare(clip(),clip(angle,invalid=invalid),video.Options(joint='shoulder'))
    def text(self,app):return ' '.join(str(x.value) for kind in ('title','subheader','caption','markdown','info','warning','error') for x in app.get(kind))

    def test_single_upload_fixed_scope_and_no_decode_on_selection(self):
        with patch.object(video,'analyze_pair') as analyze:
            app=self.page()
            self.assertEqual(len(app.get('file_uploader')),1)
            self.assertEqual(len(app.button),1)
            self.assertEqual(app.get('file_uploader')[0].proto.label,'Vídeo observado')
            self.assertTrue(app.button(key='_vid_run').disabled)
            self.assertEqual(len(app.slider)+len(app.selectbox)+len(app.checkbox)+len(app.text_input),0)
            self.assertIn('Elevação lateral do braço direito',self.text(app))
            self.assertIn('mesma fase',self.text(app))
            with self.upload():
                app.run();self.assertFalse(app.button(key='_vid_run').disabled)
                self.assertEqual(len(app.get('video')),0)
            analyze.assert_not_called()

    def test_missing_model_reference_or_worker_fail_closed(self):
        for name,message in (('verify_model','Modelo local ausente.'),('read_local_reference','Referência local ausente.'),('worker_python','Configuração do worker inválida.')):
            with self.subTest(name=name),patch.object(video,name,side_effect=video.VideoError(message)),patch.object(video,'analyze_pair') as analyze:
                app=self.page()
                self.assertEqual(len(app.get('file_uploader')),1)
                self.assertTrue(app.get('file_uploader')[0].proto.disabled)
                self.assertTrue(app.button(key='_vid_run').disabled)
                self.assertIn('Configure uma vez',self.text(app))
                self.assertEqual(len(app.get('video')),0);analyze.assert_not_called()

    def test_click_uses_configured_reference_and_fixed_options(self):
        with self.upload(),patch.object(video,'analyze_pair',return_value=self.report()) as analyze:
            app=self.page();app.button(key='_vid_run').click().run()
            reference,observed,model,options=analyze.call_args.args
            self.assertEqual(reference,MP4_HEADER+b'reference')
            self.assertEqual(observed,MP4_HEADER+b'observed')
            self.assertEqual(options,video.Options(joint='shoulder',side='right',tolerance_degrees=20.0,min_event_seconds=0.6,min_quality=0.7))
            self.assertIn('De 0.00 s a 2.80 s',self.text(app))
            self.assertIn('35.0°',self.text(app))
            self.assertEqual(len(app.get('video')),1)
            self.assertEqual(len(app.get('download_button')),1)
            self.assertEqual(app.session_state['_vid_result']['workflow']['reference_source'],'configured_local')
            self.assertFalse(app.session_state['_vid_result']['workflow']['automatic_exercise_recognition'])
            app.run();self.assertEqual(analyze.call_count,1)
            app.button(key='_vid_run').click().run();self.assertEqual(analyze.call_count,2)

    def test_reference_or_upload_change_clears_until_clicked(self):
        with self.upload(),patch.object(video,'analyze_pair',return_value=self.report()) as analyze:
            app=self.page();app.button(key='_vid_run').click().run()
            self.assertEqual(len(app.get('video')),1)
            with patch.object(video,'read_local_reference',return_value=MP4_HEADER+b'new-reference'):
                app.run();self.assertEqual(len(app.get('video')),0)
                self.assertEqual(len(app.get('download_button')),0)
                self.assertEqual(analyze.call_count,1)
                app.button(key='_vid_run').click().run()
                self.assertEqual(analyze.call_args.args[0],MP4_HEADER+b'new-reference')
            with self.upload(Upload(MP4_HEADER+b'new-observed')):
                app.run();self.assertEqual(len(app.get('video')),0)
                self.assertEqual(len(app.get('download_button')),0)

    def test_remove_upload_and_server_config_change_clear(self):
        with self.upload(),patch.object(video,'analyze_pair',return_value=self.report()):
            app=self.page();app.button(key='_vid_run').click().run()
            with patch('streamlit.file_uploader',return_value=None):
                app.run();self.assertEqual(len(app.get('video')),0)
                self.assertTrue(app.button(key='_vid_run').disabled)
            app.run();app.button(key='_vid_run').click().run()
            with patch.dict(os.environ,{'FIAP_VIDEO_REFERENCE_PATH':'/operator/hidden/reference.mp4'}):
                app.run();self.assertEqual(len(app.get('video')),0)
                self.assertNotIn('/operator/hidden',self.text(app))

    def test_setup_disappears_removes_previous_report(self):
        with self.upload(),patch.object(video,'analyze_pair',return_value=self.report()):
            app=self.page();app.button(key='_vid_run').click().run()
            with patch.object(video,'read_local_reference',side_effect=video.VideoError('Referência local ausente.')):
                app.run();self.assertEqual(len(app.get('video')),0)
                self.assertEqual(len(app.get('download_button')),0)
                self.assertTrue(app.button(key='_vid_run').disabled)

    def test_oversize_not_copied_and_invalid_upload_not_rendered(self):
        class Large:
            size=video.MAX_BYTES+1
            def getvalue(self):raise AssertionError('must reject before copy')
        for item in (Large(),Upload(b'not an MP4')):
            with self.subTest(kind=type(item).__name__),self.upload(item),patch.object(video,'analyze_pair') as analyze:
                app=self.page();self.assertEqual(len(app.error),1)
                self.assertTrue(app.button(key='_vid_run').disabled)
                self.assertEqual(len(app.get('video')),0);analyze.assert_not_called()
        with self.upload(),patch.object(video,'analyze_pair',return_value=self.report()):
            app=self.page();app.button(key='_vid_run').click().run()
            with self.upload(Large()):
                app.run();self.assertEqual(len(app.get('video')),0)
                self.assertEqual(len(app.get('download_button')),0)
                self.assertTrue(app.button(key='_vid_run').disabled)

    def test_decode_failure_has_no_report_or_preview(self):
        with self.upload(),patch.object(video,'analyze_pair',side_effect=video.VideoError('Duração inválida.')):
            app=self.page();app.button(key='_vid_run').click().run()
            self.assertEqual(len(app.error),1)
            self.assertEqual(len(app.get('video')),0)
            self.assertEqual(len(app.get('download_button')),0)

    def test_no_pose_not_normal_and_control_not_validation(self):
        with self.upload(),patch.object(video,'analyze_pair',return_value=self.report(invalid=tuple(range(15)))):
            app=self.page();app.button(key='_vid_run').click().run()
            self.assertIn('Não foi possível comparar com confiança',self.text(app))
            self.assertIn('não significa ausência de desvios',self.text(app))
            self.assertNotIn('Nenhum intervalo ultrapassou',self.text(app))
        report=self.report(angle=90);report['same_input_control']=True
        with self.upload(),patch.object(video,'analyze_pair',return_value=report):
            app=self.page();app.button(key='_vid_run').click().run()
            self.assertIn('controle de repetibilidade',self.text(app))
            self.assertIn('não comprova',self.text(app))

    def test_local_bind_and_shared_upload_cap(self):
        import streamlit
        original=streamlit.get_option
        for setting,value in (('server.address','0.0.0.0'),('server.maxUploadSize',200)):
            with patch('streamlit.get_option',side_effect=lambda name: value if name==setting else original(name)):
                app=self.page();self.assertEqual(len(app.get('file_uploader')),0)
                self.assertEqual(len(app.button),0)

    def test_native_navigation_and_clean_copy(self):
        from streamlit.source_util import get_pages
        page=ROOT/'app/pages/4_Video_Movimento_Local.py'
        pages=get_pages(str(ROOT/'app/app.py'));pages=dict(sorted(pages.items(),key=lambda item:item[1]['script_path']!=str(page)))
        with patch('streamlit.elements.widgets.button.get_main_script_directory',return_value=str(ROOT/'app')),patch('streamlit.elements.widgets.button.source_util.get_pages',return_value=pages):app=self.page()
        links=app.sidebar.get('page_link')
        self.assertEqual([x.proto.label for x in links],['Predição de risco','Monitoramento','Áudio','Vídeo'])
        self.assertEqual([x.proto.icon for x in links],['🫀','📈','🔊','🎥'])
        text=''.join(c for c in unicodedata.normalize('NFKD',self.text(app).lower()) if not unicodedata.combining(c))
        self.assertNotRegex(text,r'fictici\w*');self.assertEqual(text.count('uso academico'),1)
        self.assertNotIn('docs/VIDEO_LOCAL.md',self.text(app))


if __name__=='__main__':unittest.main()
