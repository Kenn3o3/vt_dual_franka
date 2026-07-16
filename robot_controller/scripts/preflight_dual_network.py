from __future__ import annotations

import argparse
import socket
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class Endpoint:
    name: str
    host_ip: str
    robot_ip: str
    controller_port: int
    robot_port: int
    gripper_port: int


DEFAULT_ENDPOINTS = (
    Endpoint("left", "172.16.0.3", "172.16.0.2", 8092, 50051, 50052),
    Endpoint("right", "172.16.1.3", "172.16.1.2", 8093, 50061, 50062),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check dual Franka host networking before launching controllers.")
    parser.add_argument("--skip-ping", action="store_true", help="Skip ICMP reachability checks.")
    args = parser.parse_args()
    failures: list[str] = []
    for endpoint in DEFAULT_ENDPOINTS:
        if not _host_has_ip(endpoint.host_ip):
            failures.append(f"{endpoint.name}: host IP {endpoint.host_ip} is not configured")
        for port in (endpoint.controller_port, endpoint.robot_port, endpoint.gripper_port):
            if _port_in_use("127.0.0.1", port):
                failures.append(f"{endpoint.name}: localhost:{port} is already in use")
        if not args.skip_ping and not _ping(endpoint.robot_ip):
            failures.append(f"{endpoint.name}: robot {endpoint.robot_ip} did not respond to ping")
    if failures:
        print("Dual Franka preflight failed:")
        for failure in failures:
            print(f" - {failure}")
        raise SystemExit(1)
    print("Dual Franka preflight passed.")


def _host_has_ip(ip: str) -> bool:
    result = subprocess.run(["ip", "-o", "addr", "show"], text=True, capture_output=True, check=False)
    return f" {ip}/" in result.stdout


def _port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) == 0


def _ping(host: str) -> bool:
    return subprocess.run(["ping", "-c", "1", "-W", "1", host], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


if __name__ == "__main__":
    main()
