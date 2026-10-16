"""Native transport for the one model served by the private local runtime."""

import http.client
import json
import re
import socket
import threading
from contextlib import contextmanager
from typing import Callable, Dict, Optional

MODEL_TAG = 'qwen3.5:9b-q4_K_M'
OLLAMA_HOST = '127.0.0.1'
OLLAMA_PORT = 11435
MAX_RESPONSE_BYTES = 8 * 1024 * 1024


class LocalOperationCancelled(RuntimeError):
    pass


class LocalContextTooLong(RuntimeError):
    pass


def _check_cancel(stop_event: Optional[threading.Event]) -> None:
    if stop_event is not None and stop_event.is_set():
        raise LocalOperationCancelled('작업이 중단되었습니다. 완료한 작업은 보존됩니다.')


class OllamaClient:
    """Use a fixed loopback socket; environment proxies and redirects are ignored.

    >>> OllamaClient.model
    'qwen3.5:9b-q4_K_M'
    """

    model = MODEL_TAG

    @contextmanager
    def _response(self, path: str, payload: Optional[dict] = None,
                  stop_event: Optional[threading.Event] = None):
        _check_cancel(stop_event)
        connection = http.client.HTTPConnection(OLLAMA_HOST, OLLAMA_PORT, timeout=5)
        finished = threading.Event()
        transport_socket = None

        def cancel_socket() -> None:
            while not finished.wait(0.1):
                if stop_event is not None and stop_event.is_set():
                    # HTTP/1.0 or Connection: close can transfer ownership to
                    # HTTPResponse and clear connection.sock before body reads.
                    sock = transport_socket
                    if sock is not None:
                        try:
                            sock.shutdown(socket.SHUT_RDWR)
                        except OSError:
                            pass
                    return

        watcher = None
        try:
            connection.connect()
            transport_socket = connection.sock
            _check_cancel(stop_event)
            connection.sock.settimeout(300 if payload is not None else 5)
            if stop_event is not None:
                watcher = threading.Thread(target=cancel_socket, daemon=True)
                watcher.start()
            body = json.dumps(payload, ensure_ascii=False).encode('utf-8') if payload is not None else None
            connection.request('POST' if payload is not None else 'GET', path, body,
                               {'Content-Type': 'application/json'})
            response = connection.getresponse()
            if response.status != 200:
                # Never follow a Location header, even if it points to another local service.
                detail = response.read(4096).decode('utf-8', errors='replace')
                raise RuntimeError(f'Ollama 오류 ({response.status}): {detail[:800]}')
            yield response
            _check_cancel(stop_event)
        except (OSError, http.client.HTTPException) as error:
            _check_cancel(stop_event)
            raise RuntimeError(
                '로컬 Ollama에 연결하지 못했습니다. launch_local.bat로 실행한 뒤 다시 검사해 주세요. '
                '모델 로드 중이라면 메모리와 Ollama 실행 로그도 확인해 주세요.'
            ) from error
        finally:
            finished.set()
            connection.close()
            if watcher is not None:
                watcher.join(timeout=0.3)

    def _json(self, path: str) -> dict:
        with self._response(path) as response:
            data = response.read(MAX_RESPONSE_BYTES + 1)
            if len(data) > MAX_RESPONSE_BYTES:
                raise RuntimeError('Ollama 응답 크기가 제한을 초과했습니다.')
            result = json.loads(data)
            if not isinstance(result, dict):
                raise RuntimeError('Ollama 응답 형식이 올바르지 않습니다.')
            return result

    def status(self) -> dict:
        version = self._json('/api/version').get('version', '')
        models = self._json('/api/tags').get('models', [])
        if not isinstance(models, list):
            raise RuntimeError('Ollama 모델 목록 응답이 올바르지 않습니다.')
        installed = any(isinstance(model, dict) and model.get('name') == MODEL_TAG
                        for model in models)
        return {'version': str(version), 'installed': installed, 'model': MODEL_TAG}

    def _stream(self, path: str, payload: dict, stop_event: Optional[threading.Event]):
        with self._response(path, payload, stop_event) as response:
            total = 0
            while True:
                _check_cancel(stop_event)
                line = response.readline(MAX_RESPONSE_BYTES + 1)
                if not line:
                    break
                total += len(line)
                if len(line) > MAX_RESPONSE_BYTES or (path == '/api/chat' and total > MAX_RESPONSE_BYTES):
                    raise RuntimeError('Ollama 응답 크기가 제한을 초과했습니다.')
                if not line.strip():
                    continue
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise RuntimeError('Ollama 응답 형식이 올바르지 않습니다.')
                if item.get('error'):
                    raise RuntimeError(f'Ollama: {str(item["error"])[:800]}')
                yield item

    def pull(self, progress: Callable[[str], None], stop_event: threading.Event) -> None:
        success = False
        for item in self._stream('/api/pull', {'model': MODEL_TAG, 'stream': True}, stop_event):
            status = str(item.get('status', '다운로드 중'))
            total, completed = item.get('total', 0), item.get('completed', 0)
            if isinstance(total, (int, float)) and total > 0 and isinstance(completed, (int, float)):
                status += f' · {min(100, int(completed / total * 100))}%'
            progress(status)
            success = item.get('status') == 'success'
        _check_cancel(stop_event)
        if not success:
            raise RuntimeError('모델 다운로드가 완료되지 않았습니다. 다시 설치하면 이어서 다운로드합니다.')

    def chat(self, api_args: Dict, stop_event: Optional[threading.Event] = None) -> dict:
        if api_args.get('model') != MODEL_TAG:
            raise ValueError('로컬 전용 번역 모델만 사용할 수 있습니다.')
        response_format = api_args.get('response_format', {})
        schema = response_format.get('json_schema', {}).get('schema') or 'json'
        messages = api_args.get('messages', [])
        # UTF-8 bytes are a conservative upper bound for byte-fallback tokenizers.
        # Reserve output and template overhead rather than let Ollama drop page text.
        prompt_bytes = len(json.dumps(messages, ensure_ascii=False).encode('utf-8'))
        if prompt_bytes + 512 > 8192 - 2048:
            raise LocalContextTooLong('문맥이 로컬 모델 예산을 초과했습니다. 대사를 나누어 처리해 주세요.')
        payload = {
            'model': MODEL_TAG,
            'messages': messages,
            'stream': True,
            'think': False,
            'format': schema,
            'keep_alive': '5m',
            'options': {
                'num_ctx': 8192,
                'num_predict': min(2048, max(1, int(api_args.get('max_tokens', 2048)))),
                'temperature': 0.2,
                'top_p': 0.9,
            },
        }
        parts, final = [], None
        for item in self._stream('/api/chat', payload, stop_event):
            message = item.get('message', {})
            content = message.get('content', '') if isinstance(message, dict) else ''
            if content:
                if not isinstance(content, str):
                    raise RuntimeError('번역 응답이 텍스트가 아닙니다.')
                parts.append(content)
            if item.get('done'):
                final = item
        _check_cancel(stop_event)
        if final is None or not parts:
            raise RuntimeError('번역 응답이 완료되지 않았습니다. 이 페이지를 다시 처리해 주세요.')
        if final.get('done_reason') == 'length':
            raise RuntimeError('번역 응답이 길이 제한에 도달했습니다. 선택한 대사 수를 줄여 주세요.')
        return {'content': ''.join(parts), 'finish_reason': final.get('done_reason', 'stop'),
                'usage': {'prompt_tokens': final.get('prompt_eval_count', 0),
                          'completion_tokens': final.get('eval_count', 0)}}

    def test_translation(self, stop_event: Optional[threading.Event] = None) -> str:
        result = self.chat({
            'model': MODEL_TAG,
            'messages': [{'role': 'system', 'content': 'Translate Japanese to Korean. Return JSON with one key: translation.'},
                         {'role': 'user', 'content': 'こんにちは'}],
            'response_format': {'type': 'json_schema', 'json_schema': {'schema': {
                'type': 'object', 'properties': {'translation': {'type': 'string'}},
                'required': ['translation'], 'additionalProperties': False,
            }}},
            'max_tokens': 128,
        }, stop_event)
        data = json.loads(result['content'])
        text = data.get('translation', '') if isinstance(data, dict) else ''
        if not isinstance(text, str) or not re.search('[가-힣]', text):
            raise RuntimeError('모델은 응답했지만 한국어 번역 결과를 확인하지 못했습니다.')
        return text

