"""Gemini는 batchEmbedContents에 한 번에 100개까지만 받는다.

저장소 하나만 수집해도 청크가 그보다 많아 통째로 보내면 400으로 색인 전체가 실패했다.
"""

import services


class _FakeEmbedding:
    def __init__(self, value: float) -> None:
        self.values = [value]


class _FakeResponse:
    def __init__(self, count: int) -> None:
        self.embeddings = [_FakeEmbedding(float(i)) for i in range(count)]


def test_requests_are_split_into_batches_within_the_api_limit(monkeypatch) -> None:
    sent_batch_sizes: list[int] = []

    def fake_embed_content(*, model: str, contents: list[str]):
        sent_batch_sizes.append(len(contents))
        return _FakeResponse(len(contents))

    monkeypatch.setattr(services.client.models, "embed_content", fake_embed_content)

    result = services.create_embeddings([f"chunk-{i}" for i in range(250)])

    assert sent_batch_sizes == [100, 100, 50]
    # 나눠 보내도 순서와 개수는 그대로여야 한다 — 청크와 벡터가 어긋나면 인용이 엉킨다.
    assert len(result) == 250


def test_blank_texts_are_dropped_before_batching(monkeypatch) -> None:
    sent_batch_sizes: list[int] = []

    def fake_embed_content(*, model: str, contents: list[str]):
        sent_batch_sizes.append(len(contents))
        return _FakeResponse(len(contents))

    monkeypatch.setattr(services.client.models, "embed_content", fake_embed_content)

    assert services.create_embeddings(["", "   ", None]) == []
    assert sent_batch_sizes == []
