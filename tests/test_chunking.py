from artemis_api.services.chunking import chunk_text


def test_chunking_overlap_and_indices():
    text = " ".join(str(i) for i in range(20))
    chunks = chunk_text(text, chunk_words=8, overlap_words=2)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert chunks[0].text.split()[-2:] == chunks[1].text.split()[:2]


def test_empty_text_returns_no_chunks():
    assert chunk_text("   ") == []
