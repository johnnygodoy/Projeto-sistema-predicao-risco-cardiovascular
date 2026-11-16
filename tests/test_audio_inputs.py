"""Deterministic MP3 admission/process tests, not evidence of audio recognition."""
import hashlib
import io
import math
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'app'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from audio_local import AudioError,MAX_BYTES,MAX_SECONDS,validate_wav
from audio_inputs import (
    MAX_ID3_BYTES,MAX_PCM_BYTES,MAX_ENCODED_SECONDS,_decode_mp3,
    input_selection_key,inspect_mp3,prepare_audio,source_format,
)
from test_audio_local import wav_bytes


def mp3_frame(version=3,rate_index=0,bitrate_index=9,channels=1,padding=0):
    header=(0x7FF<<21)|(version<<19)|(1<<17)|(1<<16)|(bitrate_index<<12)|(rate_index<<10)|(padding<<9)|((3 if channels==1 else 0)<<6)
    rate=(44100,48000,32000)[rate_index]//(1 if version==3 else 2 if version==2 else 4)
    bitrates=(0,32,40,48,56,64,80,96,112,128,160,192,224,256,320) if version==3 else (0,8,16,24,32,40,48,56,64,80,96,112,128,144,160)
    size=(144000 if version==3 else 72000)*bitrates[bitrate_index]//rate+padding
    return header.to_bytes(4,'big')+bytes(size-4)


def id3_tag(content,version=4):
    n=len(content)
    size=bytes([(n>>21)&127,(n>>14)&127,(n>>7)&127,n&127])
    return b'ID3'+bytes([version,0,0])+size+content


class AdmissionTests(unittest.TestCase):
    def test_complete_frames_rates_channels_and_variable_bitrate(self):
        for version in [3,2,0]:
            for rate in range(3):
                for channels in [1,2]:
                    payload=mp3_frame(version,rate,9,channels)*10+mp3_frame(version,rate,10,channels)
                    result=inspect_mp3(payload)
                    self.assertEqual(result.frame_count,11)
                    self.assertEqual(result.channels,channels)
                    self.assertGreaterEqual(result.sample_rate,8000)
                    self.assertLessEqual(result.sample_rate,48000)
                    self.assertEqual(result.payload,payload)

    def test_metadata_is_bounded_discarded_and_never_decoded(self):
        audio=mp3_frame()*10
        marker=b'PRIVATE_TAG_MARKER'
        payload=id3_tag(marker)+audio+b'TAG'+bytes(125)
        result=inspect_mp3(payload)
        self.assertEqual(result.payload,audio)
        self.assertNotIn(marker,result.payload)
        self.assertTrue(result.metadata_discarded)
        for payload in [id3_tag(bytes(MAX_ID3_BYTES)),id3_tag(b'abc')[:-1],id3_tag(b'x',2), b'ID3\x04\x00\x10'+bytes(4)+audio,
                        b'ID3\x04\x00\x00\x80\x00\x00\x00'+audio]:
            with self.subTest(head=payload[:10]),self.assertRaises(AudioError):inspect_mp3(payload)

    def test_spoofed_renamed_malformed_truncated_and_mixed_frames(self):
        good=mp3_frame()*10
        cases=[b'not audio',b'https://example.com/a.mp3',wav_bytes(),id3_tag(b'')+b'\x00\x00\x00\x18ftypisom',
               good[:-1],good+b'garbage',good[:100]+b'bad'+good[100:],mp3_frame(),
               good+mp3_frame(rate_index=1),good+mp3_frame(channels=2),good+mp3_frame(version=2)]
        for bad in cases:
            with self.subTest(length=len(bad)),patch('audio_inputs._decode_mp3') as decoder,self.assertRaises(AudioError):
                prepare_audio(bad,'recording.mp3')
            decoder.assert_not_called()
        for filename in ['recording.ogg','recording.m4a','no-extension']:
            with patch('audio_inputs._decode_mp3') as decoder,self.assertRaises(AudioError):prepare_audio(good,filename)
            decoder.assert_not_called()
        with self.assertRaises(AudioError):prepare_audio(good,'renamed.wav')

    def test_reserved_versions_layers_bitrates_and_rate_rejected(self):
        header=int.from_bytes(mp3_frame()[:4],'big')
        headers=[(header&~(3<<19))|(1<<19),(header&~(3<<17))|(2<<17),
                 header&~(15<<12),(header&~(15<<12))|(15<<12),(header&~(3<<10))|(3<<10),header|2]
        for invalid in headers:
            with self.assertRaises(AudioError):inspect_mp3(invalid.to_bytes(4,'big')+bytes(1000))

    def test_bytes_and_encoded_duration_limit_before_decoder(self):
        for payload in [b'x'*(MAX_BYTES+1),bytearray(mp3_frame()*10)]:
            with patch('audio_inputs._decode_mp3') as decoder,self.assertRaises(AudioError):prepare_audio(payload,'a.mp3')
            decoder.assert_not_called()
        count=math.floor(MAX_ENCODED_SECONDS*44100/1152)
        self.assertLessEqual(inspect_mp3(mp3_frame()*count).encoded_seconds,MAX_ENCODED_SECONDS)
        with patch('audio_inputs._decode_mp3') as decoder,self.assertRaisesRegex(AudioError,'12 minutos'):
            prepare_audio(mp3_frame()*(count+1),'a.mp3')
        decoder.assert_not_called()

    def test_pcm_bound_short_silence_and_original_fingerprint(self):
        payload=id3_tag(b'metadata')+mp3_frame()*10
        pcm=validate_wav(wav_bytes()).pcm
        with patch('audio_inputs._decode_mp3',return_value=pcm) as decoder:
            result=prepare_audio(payload,'A.MP3')
            decoder.assert_called_once_with(mp3_frame()*10)
        self.assertEqual(result.pcm,pcm)
        self.assertEqual(result.fingerprint,hashlib.sha256(payload).hexdigest())
        self.assertEqual(validate_wav(result.payload).seconds,1)
        for invalid in [b'\0'*7998,b'\0'*8001,b'\0'*(MAX_PCM_BYTES+2)]:
            with patch('audio_inputs._decode_mp3',return_value=invalid),self.assertRaises(AudioError):prepare_audio(payload,'a.mp3')
        with patch('audio_inputs._decode_mp3',return_value=b'\0'*MAX_PCM_BYTES):
            result=prepare_audio(payload,'a.mp3')
            self.assertEqual(result.seconds,MAX_SECONDS)
            self.assertTrue(result.digital_silence)

    def test_wav_nonregression_does_not_need_ffmpeg(self):
        payload=wav_bytes()
        with patch('audio_inputs._decode_mp3',side_effect=AssertionError('No WAV decoder')):
            self.assertEqual(prepare_audio(payload,'sample.wav'),validate_wav(payload))
        with self.assertRaises(AudioError):prepare_audio(wav_bytes(channels=2),'sample.wav')

    def test_selection_tracks_extension_content_model_language_and_version(self):
        data=mp3_frame()*10
        base=input_selection_key(data,'a.mp3','/model','en')
        self.assertEqual(base,input_selection_key(data,'other.MP3','/model','en'))
        for args in [(data,'renamed.wav','/model','en'),(data+b'x','a.mp3','/model','en'),(data,'a.mp3','/other','en'),(data,'a.mp3','/model','pt')]:
            self.assertNotEqual(base,input_selection_key(*args))
        with patch('audio_inputs.VERSION','changed'):
            self.assertNotEqual(base,input_selection_key(data,'a.mp3','/model','en'))
        self.assertNotEqual(base,input_selection_key(data,'a.mp3','/model','en',generation=1))


class Writer:
    def __init__(self,fail=False):self.data=bytearray();self.fail=fail;self.closed=False
    def write(self,data):
        if self.fail:raise BrokenPipeError()
        n=min(7,len(data));self.data.extend(data[:n]);return n
    def close(self):self.closed=True

class Process:
    def __init__(self,output=b'\0'*32000,returncode=0,timeout=False,fail_input=False):
        self.stdin=Writer(fail_input);self.stdout=io.BytesIO(output);self.stderr=io.BytesIO(b'private-path-and-decoder-log')
        self.returncode=returncode;self.timeout=timeout;self.killed=False;self.wait_calls=0
    def kill(self):self.killed=True
    def wait(self,timeout):
        self.wait_calls+=1
        if self.timeout and self.wait_calls==1:raise subprocess.TimeoutExpired('ffmpeg',timeout)
        return self.returncode

class ProcessTests(unittest.TestCase):
    def run_process(self,process):
        return patch('audio_inputs.shutil.which',return_value='/trusted/ffmpeg'),patch('audio_inputs.subprocess.Popen',return_value=process)

    def test_forced_decoder_pipe_only_no_shell_or_file_log_and_partial_writes(self):
        p=Process();which,popen=self.run_process(p)
        with which,popen as launch,patch.dict('os.environ',{'FFREPORT':'file=must-not-write.log'}):
            output=_decode_mp3(b'input-frame-bytes')
        self.assertEqual(output,b'\0'*32000)
        self.assertEqual(p.stdin.data,b'input-frame-bytes')
        args,kw=launch.call_args
        command=args[0]
        self.assertIn('-protocol_whitelist',command);self.assertEqual(command[command.index('-protocol_whitelist')+1],'pipe')
        self.assertEqual(command[command.index('-i')+1],'pipe:0');self.assertEqual(command[-1],'pipe:1')
        self.assertEqual(command[command.index('-f')+1],'mp3')
        self.assertEqual(command[command.index('-codec:a')+1],'mp3')
        self.assertFalse(kw['shell']);self.assertNotIn('FFREPORT',kw['env'])

    def test_output_overflow_kills_and_discards(self):
        p=Process(output=b'\0'*(MAX_PCM_BYTES+2));which,popen=self.run_process(p)
        with which,popen,self.assertRaisesRegex(AudioError,'12 minutos'):_decode_mp3(b'input')
        self.assertTrue(p.killed)

    def test_timeout_kills_and_discards(self):
        p=Process(timeout=True);which,popen=self.run_process(p)
        with which,popen,self.assertRaisesRegex(AudioError,'60 segundos'):_decode_mp3(b'input')
        self.assertTrue(p.killed);self.assertEqual(p.wait_calls,2)

    def test_failures_do_not_echo_logs_or_accept_partial_output(self):
        for p in [Process(returncode=1),Process(fail_input=True),Process(output=b'odd'),Process(output=b'')]:
            which,popen=self.run_process(p)
            with which,popen,self.assertRaises(AudioError) as error:_decode_mp3(b'input')
            self.assertNotIn('private',str(error.exception))
        with patch('audio_inputs.shutil.which',return_value=None),self.assertRaisesRegex(AudioError,'PATH'):_decode_mp3(b'input')
        with patch('audio_inputs.shutil.which',return_value='/ffmpeg'),patch('audio_inputs.subprocess.Popen',side_effect=OSError),self.assertRaises(AudioError):_decode_mp3(b'input')

    def test_thread_start_failure_kills_reaps_and_closes_every_pipe(self):
        p=Process();which,popen=self.run_process(p)
        with which,popen,patch('audio_inputs.threading.Thread.start',side_effect=RuntimeError('private-start-failure')),self.assertRaises(AudioError) as error:
            _decode_mp3(b'input')
        self.assertTrue(p.killed);self.assertEqual(p.wait_calls,1)
        self.assertTrue(all(stream.closed for stream in [p.stdin,p.stdout,p.stderr]))
        self.assertNotIn('private',str(error.exception))

    def test_wait_error_and_cancellation_also_cleanup(self):
        p=Process();which,popen=self.run_process(p)
        original=p.wait
        calls=[]
        def failed_wait(timeout):
            calls.append(timeout)
            if len(calls)==1:raise OSError('private-wait-failure')
            return original(timeout)
        with which,popen,patch.object(p,'wait',side_effect=failed_wait),self.assertRaises(AudioError):_decode_mp3(b'input')
        self.assertTrue(p.killed)
        self.assertTrue(all(stream.closed for stream in [p.stdin,p.stdout,p.stderr]))
        p=Process();which,popen=self.run_process(p)
        with which,popen,patch('audio_inputs.threading.Thread.start',side_effect=KeyboardInterrupt),self.assertRaises(KeyboardInterrupt):_decode_mp3(b'input')
        self.assertTrue(p.killed)
        self.assertTrue(all(stream.closed for stream in [p.stdin,p.stdout,p.stderr]))

if __name__=='__main__':unittest.main()
