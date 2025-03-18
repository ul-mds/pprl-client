__all__ = [
    "split_into_wordlist",
    "tokenize_wordlist",
    "compute_average_tokens_for_token_list",
    "count_tokens_in_token_list",
    "compute_ngram_entropy",
    "compute_attribute_stats",
    "AttributeStats",
]

import math
from collections import defaultdict, Counter
from typing import TypedDict

import pprl_core
from pprl_model import AttributeValueEntity, BaseTransformRequest

from ._client import PPRLClient


class AttributeStats(TypedDict):
    average_tokens: float
    ngram_entropy: float


def split_into_wordlist(entities: list[AttributeValueEntity]) -> dict[str, list[str]]:
    """Split a list of entities into a dictionary of attribute names to values."""
    attr_name_to_wordlist: dict[str, list[str]] = defaultdict(list)

    for entity in entities:
        for attr_name, attr_value in entity.attributes.items():
            attr_name_to_wordlist[attr_name].append(attr_value)

    return attr_name_to_wordlist


def tokenize_wordlist(wordlist: list[str], token_size=2, padding="_") -> list[set[str]]:
    return [pprl_core.common.tokenize(word, q=token_size, padding=padding) for word in wordlist]


def compute_average_tokens_for_token_list(token_list: list[set[str]]) -> float:
    total_token_count = sum(len(tokens) for tokens in token_list)

    if total_token_count == 0:
        return 0

    return total_token_count / len(token_list)


def count_tokens_in_token_list(token_list: list[set[str]]) -> dict[str, int]:
    token_counter: dict[str, int] = Counter()

    for word_tokens in token_list:
        for token in word_tokens:
            token_counter[token] += 1

    return token_counter


def compute_ngram_entropy(token_counts: dict[str, int]) -> float:
    total_ngram_count = sum(c for c in token_counts.values())
    entropy = 0

    for count in token_counts.values():
        p = count / total_ngram_count
        entropy += p * math.log2(p)

    return -entropy


def compute_attribute_stats(
    client: PPRLClient,
    entities: list[AttributeValueEntity],
    base_transform_request: BaseTransformRequest,
    token_size: int = 2,
    padding: str = "_",
    batch_size: int = 100,
):
    processed_entities: list[AttributeValueEntity] = []

    for i in range(0, len(entities), batch_size):
        req = base_transform_request.with_entities(entities[i : i + batch_size])
        res = client.transform(req)
        processed_entities.extend(res.entities)

    attribute_name_to_wordlist = split_into_wordlist(processed_entities)

    def _compute_stats_for_wordlist(wordlist: list[str]) -> AttributeStats:
        token_list = tokenize_wordlist(wordlist, token_size=token_size, padding=padding)
        average_tokens = compute_average_tokens_for_token_list(token_list)
        token_counts = count_tokens_in_token_list(token_list)
        ngram_entropy = compute_ngram_entropy(token_counts)

        return {"average_tokens": average_tokens, "ngram_entropy": ngram_entropy}

    return {
        attr_name: _compute_stats_for_wordlist(wordlist) for attr_name, wordlist in attribute_name_to_wordlist.items()
    }
