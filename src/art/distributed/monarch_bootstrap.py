from __future__ import annotations

"""Provider-neutral bootstrap for pinned torchmonarch 0.2.

Both worker and controller endpoints must be reachable only on one trusted private
network. Monarch 0.2 requires ``trust_all_connections`` because authenticated
transport is unavailable in that release.
"""

import argparse
import asyncio
from collections.abc import Mapping, Sequence
import multiprocessing
import os
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .rollout import InstalledAsyncCallable

DEFAULT_MONARCH_PORT = 22222


class _BootstrapContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExplicitHostBootstrap(_BootstrapContract):
    worker_addresses: tuple[str, ...]
    controller_rank: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _validate_workers(self) -> "ExplicitHostBootstrap":
        if not self.worker_addresses:
            raise ValueError("worker_addresses must not be empty")
        if len(set(self.worker_addresses)) != len(self.worker_addresses):
            raise ValueError("worker_addresses must be unique")
        if self.controller_rank >= len(self.worker_addresses):
            raise ValueError("controller_rank must identify a worker address")
        return self


class SkyPilotBootstrap(_BootstrapContract):
    node_rank: int = Field(ge=0)
    node_ips: tuple[str, ...]
    port: int = Field(default=DEFAULT_MONARCH_PORT, ge=1, le=65535)

    @classmethod
    def from_environ(
        cls, environ: Mapping[str, str] | None = None
    ) -> "SkyPilotBootstrap":
        environ = os.environ if environ is None else environ
        try:
            node_rank = int(environ["SKYPILOT_NODE_RANK"])
            node_ips = tuple(environ["SKYPILOT_NODE_IPS"].replace(",", "\n").split())
        except KeyError as error:
            raise RuntimeError(
                f"missing SkyPilot environment variable {error.args[0]}"
            ) from None
        declared_nodes = int(environ.get("SKYPILOT_NUM_NODES", len(node_ips)))
        if declared_nodes != len(node_ips):
            raise ValueError(
                f"SKYPILOT_NUM_NODES={declared_nodes} but received {len(node_ips)} IPs"
            )
        return cls(node_rank=node_rank, node_ips=node_ips)

    @model_validator(mode="after")
    def _validate_rank(self) -> "SkyPilotBootstrap":
        if not self.node_ips or self.node_rank >= len(self.node_ips):
            raise ValueError("SkyPilot node rank must identify a node IP")
        return self

    @property
    def worker_addresses(self) -> tuple[str, ...]:
        return tuple(f"tcp://{ip}:{self.port}" for ip in self.node_ips)


def run_worker(address: str) -> None:
    """Run a pinned Monarch 0.2 worker on a trusted private network.

    Monarch 0.2 does not implement transport authentication, so its required
    trust-all mode must never be exposed to an untrusted or public network.
    """

    from monarch.actor import run_worker_loop_forever  # ty: ignore[unresolved-import]

    run_worker_loop_forever(address=address, ca="trust_all_connections")


async def attach_controller(
    worker_addresses: Sequence[str], *, name: str = "art"
) -> Any:
    """Attach a controller to already-started workers on a trusted network."""

    from monarch.actor import attach_to_workers  # ty: ignore[unresolved-import]

    hosts = attach_to_workers(
        workers=list(worker_addresses), ca="trust_all_connections", name=name
    )
    await hosts.initialized
    return hosts


async def run_explicit_controller(
    spec: ExplicitHostBootstrap, program: InstalledAsyncCallable
) -> Any:
    hosts = await attach_controller(spec.worker_addresses)
    try:
        return await program.resolve()(hosts)
    finally:
        await hosts.shutdown()


def run_skypilot(program: InstalledAsyncCallable) -> None:
    """Translate SkyPilot topology and run one worker per node plus rank-0 controller."""

    spec = SkyPilotBootstrap.from_environ()
    address = spec.worker_addresses[spec.node_rank]
    if spec.node_rank != 0:
        run_worker(address)
        return
    worker = multiprocessing.Process(target=run_worker, args=(address,), daemon=True)
    worker.start()

    async def controller() -> None:
        try:
            hosts = await attach_controller(spec.worker_addresses)
            try:
                await program.resolve()(hosts)
            finally:
                await hosts.shutdown()
        finally:
            worker.terminate()
            worker.join(timeout=10)

    asyncio.run(controller())


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="ART Monarch bootstrap (trusted private networks only)",
        epilog=(
            "torchmonarch 0.2 has no transport authentication; never expose worker "
            "addresses to a public or untrusted network."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--address", required=True)
    controller = subparsers.add_parser("controller")
    controller.add_argument("--worker", action="append", required=True)
    controller.add_argument("--program", required=True, help="module:qualname")
    sky = subparsers.add_parser("skypilot")
    sky.add_argument("--program", required=True, help="module:qualname")
    args = parser.parse_args(argv)
    if args.command == "worker":
        run_worker(args.address)
        return
    module, separator, qualname = args.program.partition(":")
    if not separator:
        parser.error("--program must use module:qualname")
    program = InstalledAsyncCallable(module=module, qualname=qualname)
    if args.command == "skypilot":
        run_skypilot(program)
    else:
        asyncio.run(
            run_explicit_controller(
                ExplicitHostBootstrap(worker_addresses=tuple(args.worker)), program
            )
        )
