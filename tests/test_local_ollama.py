import http.client
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from ballontranslator.utils import local_ollama


@pytest.fixture
def server(monkeypatch):
    state = {'requests': [], 'redirect': False, 'truncated': False}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if state['redirect']:
                self.send_response(302)
                self.send_header('Location', 'https://example.invalid')
                self.end_headers()
                return
            result = {'version': '0.18.0'} if self.path == '/api/version' else {
                'models': [{'name': local_ollama.MODEL_TAG}]
            }
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            state['requests'].append((self.path, body))
            if state.get('chunked'):
                self.protocol_version = 'HTTP/1.1'
            self.send_response(200)
            if state.get('chunked'):
                self.send_header('Transfer-Encoding', 'chunked')
            self.end_headers()
            if state.get('stall'):
                state['started'].set()
                state['release'].wait(3)
                return
            if self.path == '/api/pull':
                items = [{'status': 'downloading', 'total': 100, 'completed': 50}, {'status': 'success'}]
            elif body.get('keep_alive') == 0:
                items = [{'model': local_ollama.MODEL_TAG, 'message': {'role': 'assistant', 'content': ''},
                          'done': True, 'done_reason': state.get('unload_reason', 'unload')}]
            else:
                items = [{'message': {'content': state.get('reply', '{"translation":"안녕하세요"}')}},
                         {'done': True, 'done_reason': 'stop', 'eval_count': 4}]
            if state['truncated']:
                items = items[:-1]
            for item in items:
                line = json.dumps(item).encode() + b'\n'
                if state.get('chunked'):
                    self.wfile.write(f'{len(line):x}\r\n'.encode() + line + b'\r\n')
                else:
                    self.wfile.write(line)
            if state.get('chunked'):
                self.wfile.write(b'0\r\n\r\n')

    httpd = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(local_ollama, 'OLLAMA_PORT', httpd.server_port)
    yield state
    httpd.shutdown()
    httpd.server_close()
    thread.join()


def test_model_status_uses_exact_tag_and_no_proxy(server, monkeypatch):
    monkeypatch.setenv('HTTP_PROXY', 'http://example.invalid:1234')
    assert local_ollama.OllamaClient().status() == {
        'version': '0.18.0', 'installed': True, 'model': local_ollama.MODEL_TAG,
    }


def test_native_chat_disables_thinking_and_keeps_schema(server):
    schema = {'type': 'object', 'properties': {'translation': {'type': 'string'}}}
    result = local_ollama.OllamaClient().chat({
        'model': local_ollama.MODEL_TAG,
        'messages': [{'role': 'user', 'content': 'こんにちは'}],
        'response_format': {'type': 'json_schema', 'json_schema': {'schema': schema}},
        'max_tokens': 200,
    })
    assert json.loads(result['content'])['translation'] == '안녕하세요'
    path, body = server['requests'][0]
    assert path == '/api/chat'
    assert body['think'] is False
    assert body['format'] == schema
    assert body['options']['num_ctx'] == 8192
    assert body['options']['num_predict'] == 200


def test_chunked_http11_stream_uses_normal_http_decoding(server):
    server['chunked'] = True
    assert local_ollama.OllamaClient().test_translation() == '안녕하세요'


def test_cancellable_reader_preserves_response_timeout(server):
    server.update(stall=True, started=threading.Event(), release=threading.Event())
    try:
        with pytest.raises(RuntimeError) as error:
            with local_ollama.OllamaClient()._response('/api/chat', {}, read_timeout=0.1) as response:
                response.read(1)
        assert isinstance(error.value.__cause__, TimeoutError)
    finally:
        server['release'].set()


def test_cloud_model_rejected_before_request(server):
    with pytest.raises(ValueError):
        local_ollama.OllamaClient().chat({'model': 'qwen3.5:cloud', 'messages': []})
    assert server['requests'] == []


def test_redirect_is_never_followed(server):
    server['redirect'] = True
    with pytest.raises(RuntimeError, match='302'):
        local_ollama.OllamaClient().status()


def test_download_requires_success_frame(server):
    messages = []
    local_ollama.OllamaClient().pull(messages.append, threading.Event())
    assert any('50%' in m for m in messages)
    server['truncated'] = True
    with pytest.raises(RuntimeError):
        local_ollama.OllamaClient().pull(messages.append, threading.Event())


def test_cancelled_request_does_not_start(server):
    stop = threading.Event()
    stop.set()
    with pytest.raises(local_ollama.LocalOperationCancelled):
        local_ollama.OllamaClient().pull(lambda _: None, stop)
    assert server['requests'] == []


def test_incomplete_translation_is_not_success(server):
    server['truncated'] = True
    with pytest.raises(RuntimeError):
        local_ollama.OllamaClient().chat({'model': local_ollama.MODEL_TAG, 'messages': []})


def test_model_load_test_checks_actual_korean_output(server):
    assert local_ollama.OllamaClient().test_translation() == '안녕하세요'


def test_unload_requires_native_acknowledgement_and_fixed_model(server):
    local_ollama.OllamaClient().unload()
    assert server['requests'] == [('/api/chat', {
        'model': local_ollama.MODEL_TAG, 'messages': [], 'keep_alive': 0, 'stream': False,
    })]
    server['unload_reason'] = 'stop'
    with pytest.raises(RuntimeError, match='해제'):
        local_ollama.OllamaClient().unload()


def test_oversized_context_fails_before_silent_truncation(server):
    with pytest.raises(local_ollama.LocalContextTooLong):
        local_ollama.OllamaClient().chat({
            'model': local_ollama.MODEL_TAG,
            'messages': [{'role': 'user', 'content': 'あ' * 9000}],
        })
    assert server['requests'] == []


def test_existing_translator_contract_reaches_native_ollama(server, monkeypatch):
    from ballontranslator.modules.translators.trans_llm import LLMTranslator
    from ballontranslator.modules.translators.llm_translation_contract import (
        TranslationPromptSpec, translation_system_prompt,
    )
    from ballontranslator.utils import local_edition
    from ballontranslator.utils.config import ProgramConfig
    config = ProgramConfig()
    local_edition.apply_local_profile(config)
    profile = config.module.llm_profiles[0]
    monkeypatch.setattr(local_edition, '_active', True)
    server['reply'] = '{"1":"안녕!","2":"내일 보자."}'
    spec = TranslationPromptSpec(
        'Japanese', 'Korean', translation_system_prompt(profile.prompt, 'Korean'), False,
    )
    translator = LLMTranslator('日本語', '한국어', raise_unsupported_lang=False)
    result = translator._translate(('こんにちは！', 'また明日。'), prompt_spec=spec, profile=profile)
    assert result == ['안녕!', '내일 보자.']
    _, body = server['requests'][0]
    assert body['format']['properties']['1']['type'] == 'string'
    assert 'また明日' in body['messages'][-1]['content']


@pytest.mark.parametrize('operation', ['test_translation', 'unload'])
def test_stalled_model_test_can_be_cancelled(server, operation, monkeypatch):
    server.update(stall=True, started=threading.Event(), release=threading.Event())
    stop = threading.Event()
    errors = []
    reading = threading.Event()
    method = 'read' if operation == 'unload' else 'readline'
    original_read = getattr(http.client.HTTPResponse, method)

    def signal_read(response, *args, **kwargs):
        reading.set()
        return original_read(response, *args, **kwargs)

    monkeypatch.setattr(http.client.HTTPResponse, method, signal_read)
    # Windows may leave a pending buffered read waiting after another thread
    # calls shutdown. Cancellation must also work without that wakeup.
    monkeypatch.setattr(socket.socket, 'shutdown', lambda *_: None)

    def request():
        try:
            getattr(local_ollama.OllamaClient(), operation)(stop_event=stop)
        except Exception as error:
            errors.append(error)

    worker = threading.Thread(target=request, daemon=True)
    worker.start()
    try:
        assert server['started'].wait(2)
        assert reading.wait(2)
        stop.set()
        worker.join(timeout=2)
        assert not worker.is_alive()
        assert len(errors) == 1
        assert isinstance(errors[0], local_ollama.LocalOperationCancelled)
    finally:
        server['release'].set()
        worker.join(timeout=2)

