import base64
import os
import uuid
from random import Random

import httpx
import pytest
from testcontainers.core.container import DockerContainer
from testcontainers.core.waiting_utils import wait_for_logs

from pprl_client import PPRLClient


@pytest.fixture(scope="session")
def pprl_base_url():
    # check if environment variable is set
    base_url = os.environ.get("PPRL_TEST_BASE_URL", None)

    if base_url is not None:
        yield base_url
        return

    # if not, spin up a testcontainer
    pprl_service_tag = os.environ.get("PPRL_TEST_SERVICE_VERSION", "0.1.5")
    pprl_service_port = int(os.environ.get("PPRL_TEST_SERVICE_PORT", "8080"))

    with DockerContainer(f"ghcr.io/ul-mds/pprl-service:{pprl_service_tag}").with_exposed_ports(
        pprl_service_port
    ) as container:
        wait_for_logs(container, "Application startup complete")
        yield f"http://{container.get_container_host_ip()}:{container.get_exposed_port(pprl_service_port)}"


@pytest.fixture(scope="session")
def client(pprl_base_url):
    client = httpx.Client(base_url=pprl_base_url, follow_redirects=True)

    # sanity check
    assert client.get("healthz").status_code == httpx.codes.OK.value
    return client


@pytest.fixture(scope="session")
def pprl_client(client):
    return PPRLClient(client=client)


@pytest.fixture(scope="session")
def rng_factory():
    def _rng():
        return Random(727)

    return _rng


@pytest.fixture()
def rng(rng_factory):
    return rng_factory()


@pytest.fixture(scope="session")
def uuid4_factory():
    def _uuid4():
        return str(uuid.uuid4())

    return _uuid4


@pytest.fixture(scope="session")
def base64_factory(rng_factory):
    rng = rng_factory()

    def _b64():
        return base64.b64encode(rng.randbytes(16)).decode("utf-8")

    return _b64


@pytest.fixture(scope="session", autouse=True)
def faker_session_locale():
    return ["en_US"]


@pytest.fixture(scope="session", autouse=True)
def faker_seed():
    return 727
