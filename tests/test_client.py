import pytest
from pprl_model import (
    BitVectorEntity,
    VectorMatchRequest,
    MatchConfig,
    SimilarityMeasure,
    MatchMethod,
    EntityTransformRequest,
    TransformConfig,
    EmptyValueHandling,
    GlobalTransformerConfig,
    NormalizationTransformer,
    EntityMaskRequest,
    MaskConfig,
    HashConfig,
    HashFunction,
    HashAlgorithm,
    RandomHash,
    CLKFilter,
)

from tests.helpers import generate_person

pytestmark = pytest.mark.integration


def test_match(pprl_client, base64_factory, uuid4_factory):
    domain_vectors = [
        BitVectorEntity(
            id=uuid4_factory(),
            value=base64_factory(),
        )
        for _ in range(10)
    ]

    range_vectors = [
        BitVectorEntity(
            id=uuid4_factory(),
            value=base64_factory(),
        )
        for _ in range(10)
    ]

    r = pprl_client.match(
        VectorMatchRequest(
            config=MatchConfig(
                measure=SimilarityMeasure.jaccard,
                threshold=0,
                method=MatchMethod.crosswise,
            ),
            domain=domain_vectors,
            range=range_vectors,
        )
    )

    assert len(r.matches) == len(domain_vectors) * len(range_vectors)


def test_transform(pprl_client, uuid4_factory, faker):
    entities = [generate_person(uuid4_factory(), faker) for _ in range(100)]

    r = pprl_client.transform(
        EntityTransformRequest(
            config=TransformConfig(empty_value=EmptyValueHandling.error),
            entities=entities,
            global_transformers=GlobalTransformerConfig(before=[NormalizationTransformer()]),
        )
    )

    input_ids = set(e.id for e in entities)
    output_ids = set(e.id for e in r.entities)

    assert len(entities) == len(r.entities)
    assert input_ids == output_ids


def test_mask(pprl_client, uuid4_factory, faker):
    entities = [generate_person(uuid4_factory(), faker) for _ in range(100)]

    r = pprl_client.mask(
        EntityMaskRequest(
            config=MaskConfig(
                token_size=2,
                hash=HashConfig(
                    function=HashFunction(algorithms=[HashAlgorithm.sha256], key="s3cr3t_k3y"), strategy=RandomHash()
                ),
                filter=CLKFilter(hash_values=5, filter_size=1_024),
            ),
            entities=entities,
        )
    )

    input_ids = set(e.id for e in entities)
    output_ids = set(e.id for e in r.entities)

    assert len(entities) == len(r.entities)
    assert input_ids == output_ids
