"""SOCKS5 tunnel management for KRX data access.

KRX geo-blocks US IPs. This module opens/closes an SSH SOCKS5 tunnel
through the Oracle Cloud Seoul VM on localhost:1080.

Requires:
  - SSH config entry 'oracle-seoul' with HostName, User, IdentityFile set
  - pip install "requests[socks]"

Usage:
    from krx_tunnel import krx_tunnel

    with krx_tunnel():
        from pykrx import stock
        df = stock.get_market_ohlcv(...)
"""
from __future__ import annotations

import contextlib
import os
import subprocess
import time
from typing import Iterator

import requests

SOCKS_PORT = 1080
SSH_HOST = "oracle-seoul"
PROXY_URL = f"socks5h://localhost:{SOCKS_PORT}"
TUNNEL_READY_TIMEOUT_S = 15


def _tunnel_is_routing_through_kr() -> bool:
    """Return True if the SOCKS tunnel is up and exits through KR."""
    try:
        r = requests.get(
            "https://ipinfo.io",
            proxies={"http": PROXY_URL, "https": PROXY_URL},
            timeout=5,
        )
        return r.json().get("country") == "KR"
    except Exception:
        return False


def _open_tunnel() -> None:
    """Open a backgrounded SSH SOCKS tunnel. Idempotent."""
    if _tunnel_is_routing_through_kr():
        return

    # -D: SOCKS proxy on local port
    # -N: no remote command
    # -f: background after auth
    # -o ExitOnForwardFailure=yes: fail fast if port is bound
    # -o ServerAliveInterval=60: keep NAT pinholes open during long pulls
    subprocess.run(
        [
            "ssh",
            "-D", str(SOCKS_PORT),
            "-N",
            "-f",
            "-o", "ExitOnForwardFailure=yes",
            "-o", "ServerAliveInterval=60",
            "-o", "ServerAliveCountMax=3",
            SSH_HOST,
        ],
        check=True,
    )

    # Wait for the tunnel to become reachable
    deadline = time.time() + TUNNEL_READY_TIMEOUT_S
    while time.time() < deadline:
        if _tunnel_is_routing_through_kr():
            return
        time.sleep(0.5)

    raise RuntimeError(
        f"SSH tunnel opened but not routing through KR after "
        f"{TUNNEL_READY_TIMEOUT_S}s. Check VM status and SSH config."
    )


def _close_tunnel() -> None:
    """Kill any backgrounded SSH SOCKS tunnel on SOCKS_PORT."""
    subprocess.run(
        ["pkill", "-f", f"ssh -D {SOCKS_PORT}"],
        check=False,  # pkill exits 1 if nothing matched; that's fine
    )


def _set_proxy_env() -> None:
    os.environ["HTTP_PROXY"] = PROXY_URL
    os.environ["HTTPS_PROXY"] = PROXY_URL


def _unset_proxy_env() -> None:
    os.environ.pop("HTTP_PROXY", None)
    os.environ.pop("HTTPS_PROXY", None)


@contextlib.contextmanager
def krx_tunnel() -> Iterator[None]:
    """Context manager: opens tunnel + sets proxy env, tears down on exit.

    Any `from pykrx import stock` must happen INSIDE the `with` block,
    because pykrx reads proxy env vars at import time.
    """
    _open_tunnel()
    _set_proxy_env()
    try:
        yield
    finally:
        _unset_proxy_env()
        _close_tunnel()


if __name__ == "__main__":
    # Manual smoke test: python krx_tunnel.py
    with krx_tunnel():
        r = requests.get(
            "https://ipinfo.io",
            proxies={"http": PROXY_URL, "https": PROXY_URL},
        )
        print(r.json())