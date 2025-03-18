import pytest
from pprl_model import (
    BaseTransformRequest,
    TransformConfig,
    EmptyValueHandling,
    GlobalTransformerConfig,
    NormalizationTransformer,
    AttributeTransformerConfig,
    DateTimeTransformer,
)

from pprl_client import estimate
from tests.helpers import generate_person
from pprl_client.types import AttributeStats


def test_split_into_wordlist(uuid4_factory, faker):
    entities = [generate_person(uuid4_factory(), faker) for _ in range(10)]
    entity_0 = entities[0]

    attribute_name_to_wordlist_dict = estimate.split_into_wordlist(entities)

    # check that all attribute names are present as keys
    assert set(attribute_name_to_wordlist_dict.keys()) == set(entity_0.attributes.keys())
    # check that each attribute name has values from all entities assigned to them
    assert all([set(v) == set([e.attributes[k] for e in entities]) for k, v in attribute_name_to_wordlist_dict.items()])


def test_tokenize_wordlist():
    expected = [
        {"_f", "fo", "oo", "ob", "ba", "ar", "r_"},
        {"_f", "fo", "oo", "ob", "ba", "az", "z_"},
    ]

    assert expected == estimate.tokenize_wordlist(["foobar", "foobaz"], token_size=2, padding="_")


def test_compute_average_tokens_for_token_list():
    token_list = [set("012345"), set("012"), set("012345678")]
    expected = sum(len(x) for x in token_list) / len(token_list)

    assert expected == estimate.compute_average_tokens_for_token_list(token_list)


def test_count_tokens_in_token_list():
    token_list = [
        {"_f", "fo", "oo", "ob", "ba", "ar", "r_"},
        {"_f", "fo", "oo", "ob", "ba", "az", "z_"},
    ]

    expected = {"_f": 2, "fo": 2, "oo": 2, "ob": 2, "ba": 2, "ar": 1, "az": 1, "r_": 1, "z_": 1}

    assert expected == estimate.count_tokens_in_token_list(token_list)


@pytest.mark.integration
def test_compute_attribute_stats(pprl_client, uuid4_factory, faker):
    entities = [generate_person(uuid4_factory(), faker) for _ in range(100)]
    entity_0 = entities[0]

    attribute_name_to_attribute_stats_dict = estimate.compute_attribute_stats(
        pprl_client,
        entities,
        BaseTransformRequest(
            config=TransformConfig(empty_value=EmptyValueHandling.skip),
            global_transformers=GlobalTransformerConfig(before=[NormalizationTransformer()]),
        ),
    )

    assert set(entity_0.attributes.keys()) == set(attribute_name_to_attribute_stats_dict.keys())
    assert all(
        v["ngram_entropy"] > 0 and v["average_tokens"] > 0 for v in attribute_name_to_attribute_stats_dict.values()
    )


def _is_attribute_stat_pair_equal(d0: dict[str, AttributeStats], d1: dict[str, AttributeStats]):
    assert set(d0.keys()) == set(d1.keys())

    def _float_equals(f0: float, f1: float, epsilon: float = 0.000001) -> bool:
        return abs(f0 - f1) < epsilon

    for k, v0 in d0.items():
        v1 = d1[k]

        if not _float_equals(v0["average_tokens"], v1["average_tokens"]):
            return False

        if not _float_equals(v0["ngram_entropy"], v1["ngram_entropy"]):
            return False

    return True


@pytest.mark.integration
def test_compute_attribute_stats_with_different_padding(pprl_client, uuid4_factory, faker):
    entities = [generate_person(uuid4_factory(), faker) for _ in range(100)]
    computed_attribute_stats: list[dict[str, AttributeStats]] = []

    # choice of padding SHOULD NOT affect the generated weights
    for padding in ("_", "#"):
        computed_attribute_stats.append(
            estimate.compute_attribute_stats(
                pprl_client,
                entities,
                BaseTransformRequest(
                    config=TransformConfig(empty_value=EmptyValueHandling.skip),
                    global_transformers=GlobalTransformerConfig(before=[NormalizationTransformer()]),
                ),
                padding=padding,
            )
        )

    d0, d1 = computed_attribute_stats
    assert _is_attribute_stat_pair_equal(d0, d1)


@pytest.mark.integration
def test_compute_attribute_stats_with_different_token_sizes(pprl_client, uuid4_factory, faker):
    entities = [generate_person(uuid4_factory(), faker) for _ in range(100)]
    computed_attribute_stats: list[dict[str, AttributeStats]] = []

    # choice of token size SHOULD affect the generated weights
    for token_size in (2, 3):
        computed_attribute_stats.append(
            estimate.compute_attribute_stats(
                pprl_client,
                entities,
                BaseTransformRequest(
                    config=TransformConfig(empty_value=EmptyValueHandling.skip),
                    global_transformers=GlobalTransformerConfig(before=[NormalizationTransformer()]),
                ),
                token_size=token_size,
            )
        )

    d0, d1 = computed_attribute_stats
    assert not _is_attribute_stat_pair_equal(d0, d1)


@pytest.mark.integration
def test_compute_attribute_stats_with_different_transform_config(pprl_client, uuid4_factory, faker):
    entities = [generate_person(uuid4_factory(), faker) for _ in range(100)]
    base_requests = [
        BaseTransformRequest(
            config=TransformConfig(empty_value=EmptyValueHandling.skip),
            global_transformers=GlobalTransformerConfig(before=[NormalizationTransformer()]),
        ),
        BaseTransformRequest(
            config=TransformConfig(empty_value=EmptyValueHandling.skip),
            attribute_transformers=[
                AttributeTransformerConfig(
                    attribute_name="date_of_birth",
                    transformers=[DateTimeTransformer(input_format="%Y-%m-%d", output_format="%d.%m.%Y")],
                )
            ],
        ),
    ]

    computed_attribute_stats: list[dict[str, AttributeStats]] = []

    # choice of transformer config SHOULD affect the generated weights
    for transform_base in base_requests:
        computed_attribute_stats.append(
            estimate.compute_attribute_stats(
                pprl_client,
                entities,
                transform_base,
            )
        )

    d0, d1 = computed_attribute_stats
    assert not _is_attribute_stat_pair_equal(d0, d1)
