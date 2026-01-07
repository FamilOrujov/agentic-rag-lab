from agentic_rag.rag.retrieval import build_metadata_filter


def test_build_metadata_filter_none() -> None:
    assert build_metadata_filter(None) is None
    assert build_metadata_filter([]) is None


def test_build_metadata_filter_list() -> None:
    assert build_metadata_filter(["id1", "id2"]) == {"doc_id": {"$in": ["id1", "id2"]}}
