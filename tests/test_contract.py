from datetime import datetime, timezone

from fastapi.testclient import TestClient

import ai_contract
import services
from ai_app import app
from services import SimpleVectorStore


def _chunk(artifact_id: int, text: str, seq: int = 0) -> dict:
    return {
        "artifactId": artifact_id,
        "type": "DOC",
        "title": f"artifact-{artifact_id}",
        "seq": seq,
        "text": text,
    }


def test_index_replaces_existing_artifact_chunks_and_deletes_them(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(services, "STORE_PATH", tmp_path / "store.json")
    monkeypatch.setattr(
        ai_contract,
        "create_embeddings",
        lambda texts: [[float(index + 1), 0.0] for index, _ in enumerate(texts)],
    )
    client = TestClient(app)

    first = client.post(
        "/index",
        json={"projectId": 7, "chunks": [_chunk(10, "old-a"), _chunk(10, "old-b", 1)]},
    )
    assert first.status_code == 200
    assert first.json()["indexed"] == 2

    second = client.post(
        "/index",
        json={"projectId": 7, "chunks": [_chunk(10, "new")]},
    )
    assert second.status_code == 200

    documents = SimpleVectorStore().documents
    assert [document["text"] for document in documents] == ["new"]

    deleted = client.post(
        "/index/delete",
        json={"projectId": 7, "artifactIds": [10]},
    )
    assert deleted.status_code == 200
    assert deleted.json() == {"deleted": 1}
    assert SimpleVectorStore().documents == []


def test_delete_project_only_removes_that_projects_documents(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(services, "STORE_PATH", tmp_path / "store.json")
    store = SimpleVectorStore()
    store.add_documents(
        [
            {"text": "one", "embedding": [1.0], "metadata": {"project_id": 1}},
            {"text": "two", "embedding": [1.0], "metadata": {"project_id": 2}},
        ]
    )

    response = TestClient(app).delete("/index/projects/1")

    assert response.status_code == 200
    assert response.json() == {"deleted": 1}
    assert [document["metadata"]["project_id"] for document in SimpleVectorStore().documents] == [2]


def test_search_filters_documents_before_summary_cutoff(tmp_path) -> None:
    store = SimpleVectorStore(tmp_path / "store.json")
    store.add_documents(
        [
            {
                "text": "old",
                "embedding": [1.0, 0.0],
                "metadata": {
                    "user_id": 0,
                    "project_id": 3,
                    "occurred_at": "2026-01-01T00:00:00+00:00",
                },
            },
            {
                "text": "recent",
                "embedding": [1.0, 0.0],
                "metadata": {
                    "user_id": 0,
                    "project_id": 3,
                    "occurred_at": "2026-07-29T00:00:00+00:00",
                },
            },
        ]
    )

    found = store.search(
        [1.0, 0.0],
        user_id=0,
        project_id=3,
        occurred_since=datetime(2026, 7, 20, tzinfo=timezone.utc),
    )

    assert [document["text"] for document in found] == ["recent"]


def test_citations_are_deduplicated_by_artifact_without_losing_metadata() -> None:
    documents = [
        {
            "text": "first chunk",
            "metadata": {
                "artifact_id": 11,
                "source_name": "README.md",
                "source_url": "https://example.com/readme",
            },
        },
        {
            "text": "second chunk",
            "metadata": {
                "artifact_id": 11,
                "source_name": "README.md",
                "source_url": "https://example.com/readme",
            },
        },
        {
            "text": "another artifact",
            "metadata": {
                "artifact_id": 12,
                "source_name": "회의록",
                "source_url": None,
            },
        },
    ]

    citations = ai_contract._citations(documents)

    assert citations == [
        {
            "artifactId": 11,
            "title": "README.md",
            "url": "https://example.com/readme",
            "snippet": "first chunk",
        },
        {
            "artifactId": 12,
            "title": "회의록",
            "url": None,
            "snippet": "another artifact",
        },
    ]
