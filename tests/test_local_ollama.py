import base64
import copy
import http.client
import io
import random
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from PIL import Image

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
            if state.get('error_status'):
                self.send_response(state['error_status'])
                self.end_headers()
                self.wfile.write(json.dumps({'error': state['runtime_error']}).encode())
                return
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
            if state.get('runtime_error'):
                items = [{'error': state['runtime_error']}]
            elif self.path == '/api/pull':
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


@pytest.mark.parametrize('operation', ['test_translation', 'unload', 'vision'])
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
            if operation == 'vision':
                local_ollama.OllamaClient().chat({
                    'model': local_ollama.MODEL_TAG,
                    'messages': [{'role': 'user', 'content': [_inline_image_part()]}],
                }, stop_event=stop)
            else:
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



def _inline_image_part(size=(64, 32), image_format='PNG', *, noisy=False):
    if noisy:
        rng = random.Random(0)
        image = Image.frombytes('RGB', size, bytes(rng.getrandbits(8) for _ in range(size[0] * size[1] * 3)))
    else:
        image = Image.new('RGB', size, 'white')
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    mime_type = 'image/jpeg' if image_format == 'JPEG' else 'image/png'
    encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
    return {'type': 'image_url', 'image_url': {'url': f'data:{mime_type};base64,{encoded}', 'detail': 'auto'}}


@pytest.mark.parametrize('image_format', ['JPEG', 'PNG'])
def test_inline_image_reaches_native_chat_without_mutating_input(server, image_format):
    image = _inline_image_part(image_format=image_format)
    messages = [
        {'role': 'system', 'content': 'Translate the dialogue.'},
        {'role': 'user', 'content': 'Previous page'},
        {'role': 'assistant', 'content': 'Previous translation'},
        {'role': 'user', 'content': [
            {'type': 'text', 'text': 'Before the image.\n'}, image,
            {'type': 'text', 'text': 'After the image.'},
        ]},
    ]
    original = copy.deepcopy(messages)
    result = local_ollama.OllamaClient().chat({'model': local_ollama.MODEL_TAG, 'messages': messages})
    assert json.loads(result['content'])['translation'] == '안녕하세요'
    _, payload = server['requests'][0]
    assert payload['messages'] == [*messages[:3], {
        'role': 'user', 'content': 'Before the image.\nAfter the image.',
        'images': [image['image_url']['url'].split(',', 1)[1]],
    }]
    assert 'image_url' not in json.dumps(payload)
    assert messages == original


def test_image_bytes_do_not_use_the_text_context_budget(server):
    image = _inline_image_part(size=(128, 128), noisy=True)
    assert len(image['image_url']['url']) > 8192
    local_ollama.OllamaClient().chat({
        'model': local_ollama.MODEL_TAG,
        'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'Translate.'}, image]}],
    })
    assert len(server['requests']) == 1
    assert server['requests'][0][1]['options']['num_ctx'] == 8192


def test_image_reserves_context_without_removing_source_text(server):
    messages = [{'role': 'user', 'content': 'a' * 4000}]
    local_ollama.OllamaClient().chat({'model': local_ollama.MODEL_TAG, 'messages': messages})
    messages[0]['content'] = [
        {'type': 'text', 'text': 'a' * 4000}, _inline_image_part(),
    ]
    with pytest.raises(local_ollama.LocalContextTooLong):
        local_ollama.OllamaClient().chat({'model': local_ollama.MODEL_TAG, 'messages': messages})
    assert len(server['requests']) == 1
    assert messages[0]['content'][0]['text'] == 'a' * 4000


@pytest.mark.parametrize('url', [
    'https://example.invalid/page.png', 'http://127.0.0.1/page.png',
    'file:///tmp/page.png', '/tmp/page.png',
    'data:image/gif;base64,R0lGODlh', 'data:image/png;base64,',
    'data:image/png;base64,not base64!', 'data:image/png;base64,SGVsbG8=',
])
def test_invalid_image_url_is_rejected_before_http(server, url):
    with pytest.raises(ValueError, match='이미지'):
        local_ollama.OllamaClient().chat({
            'model': local_ollama.MODEL_TAG,
            'messages': [{'role': 'user', 'content': [
                {'type': 'image_url', 'image_url': {'url': url}},
            ]}],
        })
    assert server['requests'] == []


@pytest.mark.parametrize('content', [
    None, 123, [{'type': 'audio', 'audio': 'sample'}],
    [{'type': 'text', 'text': None}], [{'type': 'image_url', 'image_url': 'invalid'}],
    [{'type': 'image_url', 'image_url': {'url': None}}], ['text'],
])
def test_malformed_content_is_rejected_before_http(server, content):
    with pytest.raises(ValueError):
        local_ollama.OllamaClient().chat({
            'model': local_ollama.MODEL_TAG, 'messages': [{'role': 'user', 'content': content}],
        })
    assert server['requests'] == []


def test_image_mime_must_match_actual_image_format(server):
    image = _inline_image_part()
    image['image_url']['url'] = image['image_url']['url'].replace('image/png', 'image/jpeg')
    with pytest.raises(ValueError, match='이미지'):
        local_ollama.OllamaClient().chat({
            'model': local_ollama.MODEL_TAG,
            'messages': [{'role': 'user', 'content': [image]}],
        })
    assert server['requests'] == []


@pytest.mark.parametrize('size', [(1025, 2), (2, 1025)])
def test_image_dimensions_are_bounded_before_http(server, size):
    with pytest.raises(ValueError, match='이미지'):
        local_ollama.OllamaClient().chat({
            'model': local_ollama.MODEL_TAG,
            'messages': [{'role': 'user', 'content': [_inline_image_part(size=size)]}],
        })
    assert server['requests'] == []


def test_image_encoded_bytes_are_bounded_before_http(server):
    encoded = base64.b64encode(b'x' * (2 * 1024 * 1024 + 1)).decode('ascii')
    with pytest.raises(ValueError, match='이미지'):
        local_ollama.OllamaClient().chat({
            'model': local_ollama.MODEL_TAG,
            'messages': [{'role': 'user', 'content': [
                {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{encoded}'}},
            ]}],
        })
    assert server['requests'] == []


@pytest.mark.parametrize('same_message', [False, True])
def test_only_one_page_image_is_allowed_per_request(server, same_message):
    images = [_inline_image_part(), _inline_image_part()]
    messages = ([{'role': 'user', 'content': images}] if same_message else
                [{'role': 'user', 'content': [image]} for image in images])
    with pytest.raises(ValueError, match='이미지'):
        local_ollama.OllamaClient().chat({'model': local_ollama.MODEL_TAG, 'messages': messages})
    assert server['requests'] == []


@pytest.mark.parametrize('error_status', [None, 400])
@pytest.mark.parametrize(('runtime_error', 'expected', 'exception'), [
    ('this model does not support images', '이미지 입력을 지원하지', RuntimeError),
    ('CUDA error: out of memory', '메모리', RuntimeError),
    ("unknown model architecture: qwen35", 'Ollama 버전', RuntimeError),
    ('input length exceeds the context window', '문맥', local_ollama.LocalContextTooLong),
])
def test_vision_runtime_failure_is_actionable_without_fallback(server, error_status, runtime_error, expected, exception):
    server.update(error_status=error_status, runtime_error=runtime_error)
    with pytest.raises(exception, match=expected):
        local_ollama.OllamaClient().chat({
            'model': local_ollama.MODEL_TAG,
            'messages': [{'role': 'user', 'content': [_inline_image_part()]}],
        })
    assert len(server['requests']) == 1
    assert server['requests'][0][1]['model'] == local_ollama.MODEL_TAG


@pytest.mark.parametrize('with_history', [False, True])
def test_actual_translator_vision_keeps_source_and_image_during_context_recovery(server, monkeypatch, with_history):
    from ballontranslator.modules.llm_vision import EncodedChatImage
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
    image = EncodedChatImage(_inline_image_part()['image_url']['url'], 'auto')
    context = None
    if with_history:
        from ballontranslator.modules.context.history import HistoryPage, RenderedHistoryPage
        from ballontranslator.modules.context.translation_context import RequestContext
        old_page = HistoryPage('previous.png', ('h' * 4000,), ('old translation',))
        context = RequestContext(history=(RenderedHistoryPage(
            old_page, (('user', old_page.sources[0]), ('assistant', old_page.translations[0])), 1100,
        ),), history_budget=2048)
    translator = LLMTranslator('日本語', '한국어', raise_unsupported_lang=False)
    assert translator._translate(
        ('こんにちは！', 'また明日。'), prompt_spec=spec, profile=profile, vision_request=image,
        request_context=context,
    ) == ['안녕!', '내일 보자.']
    _, body = server['requests'][0]
    assert len(server['requests']) == 1
    assert len(body['messages']) == 2
    assert body['messages'][-1]['images'] == [image.data_url.split(',', 1)[1]]
    assert 'また明日' in body['messages'][-1]['content']
    assert body['format']['properties']['1']['type'] == 'string'


@pytest.mark.parametrize('messages', [None, {}, [None], [{'role': 'user'}],
                                      [{'role': 'user', 'content': '', 'images': ['unvalidated']}]])
def test_malformed_messages_cannot_bypass_inline_image_validation(server, messages):
    with pytest.raises(ValueError):
        local_ollama.OllamaClient().chat({'model': local_ollama.MODEL_TAG, 'messages': messages})
    assert server['requests'] == []


def test_english_translation_smoke_test_uses_english_input_without_new_model():
    from unittest.mock import patch
    from ballontranslator.utils.local_ollama import OllamaClient
    with patch.object(OllamaClient, 'chat', return_value={'content': '{"translation":"안녕하세요"}'}) as chat:
        result = OllamaClient().test_translation(source_language='English')
    assert result == '안녕하세요'
    payload = chat.call_args.args[0]
    assert payload['messages'][1]['content'] == 'Hello, how are you?'
    assert 'English to Korean' in payload['messages'][0]['content']


def test_auto_translation_smoke_test_uses_both_source_languages():
    from unittest.mock import patch
    from ballontranslator.utils.local_ollama import OllamaClient
    with patch.object(OllamaClient, 'chat', return_value={'content': '{"translation":"안녕하세요"}'}) as chat:
        result = OllamaClient().test_translation(source_language='Auto')
    assert result == '안녕하세요'
    payload = chat.call_args.args[0]
    assert payload['model'] == local_ollama.MODEL_TAG
    assert payload['messages'][1]['content'] == 'Hello! こんにちは'
    assert 'English and Japanese to Korean' in payload['messages'][0]['content']
