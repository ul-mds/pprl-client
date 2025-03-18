import contextlib
import csv
import itertools
import json
from pathlib import Path

import click
import httpx
from pprl_model import BitVectorEntity, BaseMatchRequest, MatchMethod
from pydantic import BaseModel
from typing_extensions import TypeVar

from ._client import PPRLClient


def create_client(ctx: click.Context) -> PPRLClient:
    return PPRLClient(client=httpx.Client(base_url=ctx.obj["BASE_URL"], timeout=int(ctx.obj["TIMEOUT_SECS"])))


def read_bit_vector_entity_file(reader: csv.DictReader, id_column: str, value_column: str):
    """
    Read a CSV file containing bit vector entities.

    Args:
        reader: CSV dict reader instance
        id_column: name of ID column
        value_column: name of value column

    Returns:
        list of bit vector entities
    """
    return [BitVectorEntity(id=row[id_column], value=row[value_column]) for row in reader]


_M = TypeVar("_M", bound=BaseModel)


def parse_json_file_into(ctx: click.Context, path: str | Path, model: type[_M]) -> _M:
    with open(path, mode="r", encoding=ctx.obj["ENCODING"]) as f:
        return model(**json.load(f))


@contextlib.contextmanager
def read_csv_file(ctx: click.Context, path: str | Path, mode: str = "r"):
    with open(path, mode=mode, encoding=ctx.obj["ENCODING"], newline="") as f:
        yield csv.DictReader(f, delimiter=ctx.obj["DELIMITER"])


@contextlib.contextmanager
def write_csv_file(
    ctx: click.Context, path: str | Path, fieldnames: list[str], mode: str = "w", write_header: bool = True
):
    with open(path, mode=mode, encoding=ctx.obj["ENCODING"], newline="") as f:
        writer = csv.DictWriter(f, delimiter=ctx.obj["DELIMITER"], fieldnames=fieldnames)

        if write_header:
            writer.writeheader()

        yield writer


@click.group()
@click.pass_context
@click.option("--base-url", default="http://localhost:8080", help="base URL to HTTP-based PPRL service")
@click.option(
    "-b", "--batch-size", type=click.IntRange(min=1), default=1_000, help="amount of bit vectors to match at a time"
)
@click.option("--timeout-secs", type=click.IntRange(min=1), default=30, help="seconds until a request times out")
@click.option("--delimiter", type=str, default=",", help="column delimiter for CSV files")
@click.option("--encoding", type=str, default="utf-8", help="character encoding for files")
def app(ctx: click.Context, base_url: str, batch_size: int, timeout_secs: int, delimiter: str, encoding: str):
    ctx.ensure_object(dict)
    ctx.obj["BASE_URL"] = base_url
    ctx.obj["BATCH_SIZE"] = batch_size
    ctx.obj["TIMEOUT_SECS"] = timeout_secs
    ctx.obj["DELIMITER"] = delimiter
    ctx.obj["ENCODING"] = encoding


@app.command()
@click.pass_context
@click.argument("base_match_request_file_path", type=click.Path(exists=True, path_type=Path))
@click.argument("vector_file_path", type=click.Path(exists=True, path_type=Path, dir_okay=False), nargs=-1)
@click.argument("output_file_path", type=click.Path(path_type=Path, dir_okay=False))
@click.option("--id-column", type=str, default="id", help="column name in input CSV file containing vector ID")
@click.option("--value-column", type=str, default="value", help="column name in input CSV file containing vector value")
def match(
    ctx: click.Context,
    base_match_request_file_path: Path,
    vector_file_path: tuple[Path, ...],
    output_file_path: Path,
    id_column: str,
    value_column: str,
):
    """
    Match bit vectors from CSV files against each other.

    BASE_MATCH_REQUEST_FILE_PATH is the path to a JSON file containing the base match request.
    VECTOR_FILE_PATH is the path to a CSV file containing bit vectors.
    At least two files must be specified.
    OUTPUT_FILE_PATH is the path of the CSV file where the matches should be written to.
    """
    if len(vector_file_path) < 2:
        click.echo("Must specify at least two CSV files containing vectors", err=True)
        ctx.exit(1)

    client = create_client(ctx)
    base_match_request = parse_json_file_into(ctx, base_match_request_file_path, BaseMatchRequest)

    batch_size = int(ctx.obj["BATCH_SIZE"])
    file_count = len(vector_file_path)
    vectors_per_file: list[list[BitVectorEntity]] = []

    for p in vector_file_path:
        with read_csv_file(ctx, p, mode="r") as reader:
            vectors_per_file.append(read_bit_vector_entity_file(reader, id_column, value_column))

    # check that all files have the same amount of entries
    do_pairwise_matching = base_match_request.config.method == MatchMethod.pairwise

    if do_pairwise_matching:
        vector_lens = set(len(v) for v in vectors_per_file)

        if len(vector_lens) != 1:
            click.echo(
                "All bit vector files must have the same amount of vectors for pairwise matching, got"
                f"{', '.join([str(len(v) for v in vectors_per_file)])}"
            )
            ctx.exit(1)

    with write_csv_file(
        ctx,
        output_file_path,
        ["domain_id", "domain_file", "range_id", "range_file", "similarity"],
        mode="w",
        write_header=True,
    ) as writer:
        for domain_idx in range(0, file_count - 1):
            for range_idx in range(domain_idx + 1, file_count):
                # get domain and range vectors
                domain_vectors, range_vectors = vectors_per_file[domain_idx], vectors_per_file[range_idx]
                # these are tracked for user feedback
                domain_file_path, range_file_path = vector_file_path[domain_idx], vector_file_path[range_idx]

                # construct the starting indices for batch-wise processing
                domain_start_idx = list(range(0, len(domain_vectors), batch_size))
                range_start_idx = list(range(0, len(range_vectors), batch_size))

                # when doing pairwise matching, matching will be performed row-wise
                if do_pairwise_matching:
                    idx_pairs = zip(domain_start_idx, range_start_idx)
                # otherwise, cross-wise matching is performed
                else:
                    idx_pairs = itertools.product(domain_start_idx, range_start_idx)

                with click.progressbar(
                    idx_pairs, label=f"Matching bit vectors from {domain_file_path.name} and {range_file_path.name}"
                ) as pbar:
                    # iterate over pairs of starting indices for domain and range
                    for idx_tpl in pbar:
                        domain_idx, range_idx = idx_tpl[0], idx_tpl[1]

                        # retrieve batch of vectors
                        domain_vector_batch = domain_vectors[domain_idx : domain_idx + batch_size]
                        range_vector_batch = range_vectors[range_idx : range_idx + batch_size]

                        # and perform matching
                        r = client.match(
                            base_match_request.with_vectors(
                                domain_lst=domain_vector_batch, range_lst=range_vector_batch
                            )
                        )

                        writer.writerows(
                            [
                                {
                                    "domain_id": m.domain.id,
                                    "domain_file": domain_file_path.name,
                                    "range_id": m.range.id,
                                    "range_file": range_file_path.name,
                                    "similarity": m.similarity,
                                }
                                for m in r.matches
                            ]
                        )
