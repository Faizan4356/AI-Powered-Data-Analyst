import io

import pytest

from core.file_loader import CSVParseError, read_csv_robust


def make_file(content: bytes):
    return io.BytesIO(content)


def test_reads_plain_comma_csv():
    df = read_csv_robust(make_file(b"a,b,c\n1,2,3\n4,5,6\n"))
    assert list(df.columns) == ["a", "b", "c"]
    assert len(df) == 2


def test_reads_semicolon_delimited_csv():
    df = read_csv_robust(make_file(b"a;b;c\n1;2;3\n4;5;6\n"))
    assert list(df.columns) == ["a", "b", "c"]
    assert df["a"].tolist() == [1, 4]


def test_reads_tab_delimited_csv():
    df = read_csv_robust(make_file(b"a\tb\tc\n1\t2\t3\n"))
    assert list(df.columns) == ["a", "b", "c"]


def test_reads_pipe_delimited_csv():
    df = read_csv_robust(make_file(b"a|b|c\n1|2|3\n"))
    assert list(df.columns) == ["a", "b", "c"]


def test_reads_utf8_bom_csv():
    content = "a,b\n1,2\n".encode("utf-8-sig")
    df = read_csv_robust(make_file(content))
    assert list(df.columns) == ["a", "b"]


def test_reads_latin1_encoded_csv_with_accents():
    content = "name,city\nJos\xe9,S\xe3o Paulo\n".encode("latin-1")
    df = read_csv_robust(make_file(content))
    assert df["name"].iloc[0] == "José"
    assert df["city"].iloc[0] == "São Paulo"


def test_reads_cp1252_encoded_csv():
    content = "name,note\nCafe’s,test\n".encode("cp1252")
    df = read_csv_robust(make_file(content))
    assert "Cafe" in df["name"].iloc[0]


def test_skips_blank_lines():
    df = read_csv_robust(make_file(b"a,b\n1,2\n\n3,4\n"))
    assert len(df) == 2


def test_raises_on_empty_file():
    with pytest.raises(CSVParseError):
        read_csv_robust(make_file(b""))


def test_raises_clear_error_when_unparseable():
    # Binary garbage that won't decode cleanly under any tried encoding as valid CSV rows of consistent shape
    with pytest.raises(CSVParseError):
        read_csv_robust(make_file(b"\xff\xfe\x00\x01\x02\x03"))
