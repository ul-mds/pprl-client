import csv
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Callable

import pytest
from click.testing import CliRunner
from pprl_model import BaseMatchRequest, MatchConfig, SimilarityMeasure, MatchMethod
from pydantic import BaseModel

from pprl_client._cli import app


pytestmark = pytest.mark.integration


@pytest.fixture()
def cli_runner():
    return CliRunner()


def h256(p: Path):
    h = hashlib.sha256()
    buf_size = 64 * 1_024

    with open(p, "rb", buffering=0) as f:
        while True:
            data = f.read(buf_size)

            if not data:
                break

            h.update(data)

    return h.hexdigest()


def count(i: Iterable):
    return sum(1 for _ in i)


def write_random_vectors_to(path: Path, base64_factory: Callable[[], str], n: int = 1_000):
    with open(path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "value"])
        writer.writeheader()

        writer.writerows([{"id": str(i), "value": base64_factory()} for i in range(n)])


def write_model_to(path: Path, model: BaseModel):
    with open(path, mode="w", encoding="utf-8") as f:
        json.dump(model.model_dump(mode="json", exclude_none=True), f)


def test_match_pairwise(tmp_path_factory, base64_factory, cli_runner, pprl_client):
    # set up vectors
    tmp_dir = tmp_path_factory.mktemp("output")

    domain_path = tmp_dir / "domain.csv"
    range_path = tmp_dir / "range.csv"

    vector_count = 100

    write_random_vectors_to(domain_path, base64_factory, vector_count)
    write_random_vectors_to(range_path, base64_factory, vector_count)

    # check that different files were actually written
    assert h256(domain_path) != h256(range_path)

    # create base match request and export it
    base_match_request = BaseMatchRequest(
        config=MatchConfig(
            measure=SimilarityMeasure.jaccard,
            threshold=0,
            method=MatchMethod.pairwise,
        )
    )

    base_match_request_path = tmp_dir / "match-request.json"
    write_model_to(base_match_request_path, base_match_request)

    # set up output file
    output_path = tmp_dir / "output.csv"

    result = cli_runner.invoke(
        app,
        [
            "--base-url",
            str(pprl_client._client.base_url),
            "--batch-size",
            "10",
            "match",
            str(base_match_request_path),
            str(domain_path),
            str(range_path),
            str(output_path),
        ],
    )

    # check that everything went fine
    assert result.exit_code == 0
    assert output_path.exists()

    # check output file contents
    with open(output_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        assert set(reader.fieldnames) == {"domain_id", "domain_file", "range_id", "range_file", "similarity"}
        assert count(reader) == vector_count


def test_match_crosswise(tmp_path_factory, base64_factory, cli_runner, pprl_client):
    # set up vectors
    tmp_dir = tmp_path_factory.mktemp("output")

    domain_path = tmp_dir / "domain.csv"
    range_path = tmp_dir / "range.csv"

    vector_count = 100

    write_random_vectors_to(domain_path, base64_factory, vector_count)
    write_random_vectors_to(range_path, base64_factory, vector_count)

    # check that different files were actually written
    assert h256(domain_path) != h256(range_path)

    # create base match request and export it
    base_match_request = BaseMatchRequest(
        config=MatchConfig(
            measure=SimilarityMeasure.jaccard,
            threshold=0,
            method=MatchMethod.crosswise,
        )
    )

    base_match_request_path = tmp_dir / "match-request.json"
    write_model_to(base_match_request_path, base_match_request)

    # set up output file
    output_path = tmp_dir / "output.csv"

    result = cli_runner.invoke(
        app,
        [
            "--base-url",
            str(pprl_client._client.base_url),
            "--batch-size",
            "10",
            "match",
            str(base_match_request_path),
            str(domain_path),
            str(range_path),
            str(output_path),
        ],
    )

    # check that everything went fine
    assert result.exit_code == 0
    assert output_path.exists()

    # check output file contents
    with open(output_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        assert set(reader.fieldnames) == {"domain_id", "domain_file", "range_id", "range_file", "similarity"}
        assert count(reader) == vector_count * vector_count
