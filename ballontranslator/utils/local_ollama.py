"""Native transport for the one model served by the private local runtime."""

import base64
import binascii
import http.client
import io
import json
import re
import select
import threading
import time
import warnings
from contextlib import contextmanager
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple

MODEL_TAG = 'qwen3.5:9b-q4_K_M'
OLLAMA_HOST = '127.0.0.1'
OLLAMA_PORT = 11435
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_DIMENSION = 1024
MAX_IMAGE_BYTES = 2 * 1024 * 1024
MAX_IMAGE_BASE64_BYTES = 4 * ((MAX_IMAGE_BYTES + 2) // 3)
# This is a conservative allowance, not a model-specific image-token estimate.
# Keep image pixels and output bounded within the existing local 8192 context.
VISION_CONTEXT_RESERVE = 2048


class LocalOperationCancelled(RuntimeError):
    pass


class LocalContextTooLong(RuntimeError):
    pass


def _check_cancel(stop_event: Optional[threading.Event]) -> None:
    if stop_event is not None and stop_event.is_set():
        raise LocalOperationCancelled('작업이 중단되었습니다. 완료한 작업은 보존됩니다.')


def _native_image_data(image_url: object) -> str:
    """Validate one in-memory app image; never resolve a URL or file path."""
    if not isinstance(image_url, dict):
        raise ValueError('이미지 URL은 인라인 JPEG/PNG 데이터여야 합니다.')
    url = image_url.get('url')
    if not isinstance(url, str):
        raise ValueError('이미지 URL은 인라인 JPEG/PNG 데이터여야 합니다.')
    header, separator, encoded = url.partition(',')
    formats = {'data:image/jpeg;base64': 'JPEG', 'data:image/png;base64': 'PNG'}
    if not separator or header not in formats or not encoded:
        raise ValueError('이미지는 인라인 JPEG/PNG 데이터만 사용할 수 있습니다. 외부 URL과 파일 경로는 지원하지 않습니다.')
    if len(encoded) > MAX_IMAGE_BASE64_BYTES:
        raise ValueError('이미지 데이터가 로컬 모델의 2 MiB 제한을 초과했습니다.')
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError('이미지의 base64 데이터가 올바르지 않습니다.') from error
    if len(decoded) > MAX_IMAGE_BYTES:
        raise ValueError('이미지 데이터가 로컬 모델의 2 MiB 제한을 초과했습니다.')

    # Pillow is already required for project images. Read only memory here, and
    # check the header dimensions before allocating the decompressed pixels.
    from PIL import Image
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(decoded), formats=('JPEG', 'PNG')) as image:
                if image.format != formats[header]:
                    raise ValueError('이미지 MIME 형식과 실제 데이터가 일치하지 않습니다.')
                if min(image.size) < 1 or max(image.size) > MAX_IMAGE_DIMENSION:
                    raise ValueError(f'이미지의 긴 변은 {MAX_IMAGE_DIMENSION}px 이하여야 합니다.')
                image.verify()
            with Image.open(io.BytesIO(decoded), formats=('JPEG', 'PNG')) as image:
                image.load()
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as error:
        raise ValueError(f'이미지 데이터가 올바르지 않거나 크기 제한을 초과했습니다: {error}') from error
    return encoded


def _native_messages(messages: object) -> Tuple[List[dict], int]:
    """Convert inline content without mutating messages or merging page images.

    Ollama attaches images to a message rather than an inline text position.
    Text parts retain their order; the current page is the only allowed image.

    >>> _native_messages([{'role': 'user', 'content': 'hello'}])
    ([{'role': 'user', 'content': 'hello'}], 0)
    """
    if not isinstance(messages, list):
        raise ValueError('로컬 모델의 messages는 메시지 목록이어야 합니다.')
    native, image_count = [], 0
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError('로컬 모델의 메시지 형식이 올바르지 않습니다.')
        if 'images' in message:
            raise ValueError('이미지는 인라인 JPEG/PNG 콘텐츠로 전달해야 합니다.')
        content = message.get('content')
        converted = dict(message)
        if isinstance(content, str):
            native.append(converted)
            continue
        if not isinstance(content, list):
            raise ValueError('로컬 모델의 콘텐츠는 텍스트 또는 텍스트/이미지 목록이어야 합니다.')
        text_parts, images = [], []
        for part in content:
            if not isinstance(part, dict):
                raise ValueError('로컬 모델의 콘텐츠 항목 형식이 올바르지 않습니다.')
            if part.get('type') == 'text' and isinstance(part.get('text'), str):
                text_parts.append(part['text'])
            elif part.get('type') == 'image_url':
                image_count += 1
                if image_count > 1:
                    raise ValueError('로컬 모델 요청에는 페이지 또는 영역 이미지 한 장만 첨부할 수 있습니다.')
                images.append(_native_image_data(part.get('image_url')))
            else:
                raise ValueError('로컬 모델은 텍스트와 인라인 JPEG/PNG 이미지만 지원합니다.')
        converted['content'] = ''.join(text_parts)
        if images:
            converted['images'] = images
        native.append(converted)
    return native, image_count


def _runtime_error(detail: str, payload: Optional[dict]) -> RuntimeError:
    """Explain image/runtime failures while retaining the server diagnostic."""
    has_image = payload is not None and any(
        message.get('images') for message in payload.get('messages', [])
    )
    if not has_image:
        return RuntimeError(detail)
    lowered = detail.lower()
    if any(phrase in lowered for phrase in ('context length', 'context window', 'prompt is too long')):
        return LocalContextTooLong(f'이미지와 대사의 문맥이 로컬 모델 예산을 초과했습니다. {detail}')
    if any(phrase in lowered for phrase in (
        'out of memory', 'not enough memory', 'requires more system memory',
        'failed to allocate', 'unable to allocate',
    )):
        return RuntimeError(f'이미지 번역 중 메모리가 부족합니다. 다른 GPU 작업을 종료하거나 준비 화면의 이미지 문맥 자동 확인을 끄고 다시 시도해 주세요. {detail}')
    if any(phrase in lowered for phrase in (
        'does not support images', 'does not support image input', 'does not support vision',
        'image input is not supported', 'vision is not supported',
    )):
        return RuntimeError(f'현재 Ollama 실행기가 이 모델의 이미지 입력을 지원하지 않습니다. 로컬 Ollama 버전을 확인해 주세요. {detail}')
    if any(phrase in lowered for phrase in ('unknown model architecture', 'requires a newer version')):
        return RuntimeError(f'현재 Ollama 버전에서 이미지 모델을 실행하지 못했습니다. 로컬 실행기 업데이트를 확인해 주세요. {detail}')
    return RuntimeError(detail)


class _CancellableReader(io.RawIOBase):
    """Poll a nonblocking socket inside the consuming worker on every raw read."""

    def __init__(self, raw, sock, stop_event, timeout):
        self.raw, self.sock = raw, sock
        self.stop_event, self.timeout = stop_event, timeout

    def readable(self):
        return True

    def readinto(self, buffer):
        if not len(buffer):
            return 0
        deadline = time.monotonic() + self.timeout
        while True:
            _check_cancel(self.stop_event)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Timed out waiting for the local model response')
            ready, _, _ = select.select([self.sock], [], [], min(0.1, remaining))
            if ready:
                # SocketIO returns None if readiness changed before recv. The
                # nonblocking socket never re-enters a long Windows wait.
                count = self.raw.readinto(buffer)
                if count is not None:
                    return count

    def close(self):
        try:
            self.raw.close()
        finally:
            super().close()


class _CancellableResponse(http.client.HTTPResponse):
    def __init__(self, sock, *args, stop_event=None, read_timeout=300, **kwargs):
        super().__init__(sock, *args, **kwargs)
        # No headers have been read yet. Preserve the socket file's ownership
        # while retaining HTTPResponse's normal buffering and chunk decoding.
        raw = self.fp.detach()
        self.fp = io.BufferedReader(_CancellableReader(raw, sock, stop_event, read_timeout))


class OllamaClient:
    """Use a fixed loopback socket; environment proxies and redirects are ignored.

    >>> OllamaClient.model
    'qwen3.5:9b-q4_K_M'
    """

    model = MODEL_TAG

    @contextmanager
    def _response(self, path: str, payload: Optional[dict] = None,
                  stop_event: Optional[threading.Event] = None,
                  read_timeout: Optional[float] = None):
        _check_cancel(stop_event)
        connection = http.client.HTTPConnection(OLLAMA_HOST, OLLAMA_PORT, timeout=5)
        timeout = read_timeout if read_timeout is not None else (300 if payload is not None else 5)
        connection.response_class = partial(
            _CancellableResponse, stop_event=stop_event, read_timeout=timeout,
        )
        response = None
        try:
            connection.connect()
            _check_cancel(stop_event)
            body = json.dumps(payload, ensure_ascii=False).encode('utf-8') if payload is not None else None
            connection.request('POST' if payload is not None else 'GET', path, body,
                               {'Content-Type': 'application/json'})
            connection.sock.setblocking(False)
            response = connection.getresponse()
            if response.status != 200:
                # Never follow a Location header, even if it points to another local service.
                detail = response.read(4096).decode('utf-8', errors='replace')
                raise _runtime_error(f'Ollama 오류 ({response.status}): {detail[:800]}', payload)
            yield response
            _check_cancel(stop_event)
        except (OSError, http.client.HTTPException) as error:
            _check_cancel(stop_event)
            raise RuntimeError(
                '로컬 Ollama에 연결하지 못했습니다. launch_local.bat로 실행한 뒤 다시 검사해 주세요. '
                '모델 로드 중이라면 메모리와 Ollama 실행 로그도 확인해 주세요.'
            ) from error
        finally:
            if response is not None:
                response.close()
            connection.close()

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
                    raise _runtime_error(f'Ollama: {str(item["error"])[:800]}', payload)
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

    def unload(self, stop_event: Optional[threading.Event] = None) -> None:
        """Wait for Ollama to release the translation model before image work.

        The native chat API returns one object with done_reason="unload".
        https://github.com/ollama/ollama/blob/main/docs/api.md#unload-a-model-1
        """
        payload = {'model': MODEL_TAG, 'messages': [], 'keep_alive': 0, 'stream': False}
        with self._response('/api/chat', payload, stop_event, read_timeout=30) as response:
            data = response.read(MAX_RESPONSE_BYTES + 1)
            _check_cancel(stop_event)
            if len(data) > MAX_RESPONSE_BYTES:
                raise RuntimeError('Ollama 응답 크기가 제한을 초과했습니다.')
            result = json.loads(data)
            if (not isinstance(result, dict) or result.get('done') is not True
                    or result.get('done_reason') != 'unload' or result.get('error')):
                raise RuntimeError('번역 모델의 메모리 해제를 확인하지 못했습니다. Ollama를 다시 실행해 주세요.')

    def chat(self, api_args: Dict, stop_event: Optional[threading.Event] = None) -> dict:
        if api_args.get('model') != MODEL_TAG:
            raise ValueError('로컬 전용 번역 모델만 사용할 수 있습니다.')
        response_format = api_args.get('response_format', {})
        schema = response_format.get('json_schema', {}).get('schema') or 'json'
        _check_cancel(stop_event)
        messages, image_count = _native_messages(api_args.get('messages', []))
        # Base64 is transport data, not prompt text. Account for text separately
        # and reserve bounded visual context instead of counting encoded bytes.
        text_messages = [{key: value for key, value in message.items() if key != 'images'}
                         for message in messages]
        prompt_bytes = len(json.dumps(text_messages, ensure_ascii=False).encode('utf-8'))
        if prompt_bytes + 512 + image_count * VISION_CONTEXT_RESERVE > 8192 - 2048:
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

