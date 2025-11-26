import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

try:  # pragma: no cover - optional dependency
    from cassandra.auth import PlainTextAuthProvider  # type: ignore
    from cassandra.cluster import Cluster, Session  # type: ignore
    from cassandra.query import PreparedStatement, SimpleStatement  # type: ignore
except ImportError:  # pragma: no cover - driver optional for local dev
    PlainTextAuthProvider = None  # type: ignore
    Cluster = None  # type: ignore
    Session = None  # type: ignore
    PreparedStatement = None  # type: ignore
    SimpleStatement = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover
    from cassandra.auth import PlainTextAuthProvider as PlainTextAuthProviderType
    from cassandra.cluster import Cluster as ClusterType, Session as SessionType
    from cassandra.query import PreparedStatement as PreparedStatementType
else:  # pragma: no cover
    PlainTextAuthProviderType = Any
    ClusterType = Any
    SessionType = Any
    PreparedStatementType = Any

logger = logging.getLogger("bankTrading.cassandra")


class CassandraUnavailable(RuntimeError):
    """Raised when Cassandra operations are requested without a live session."""


@dataclass
class CassandraConfig:
    contact_points: Sequence[str]
    port: int
    username: Optional[str]
    password: Optional[str]
    rt_keyspace: str
    audit_keyspace: str
    enabled: bool


def _parse_contact_points(raw: Optional[str]) -> Sequence[str]:
    if not raw:
        return ["127.0.0.1"]
    if raw.startswith("jdbc:"):
        raw = raw.replace("jdbc:", "", 1)
        raw = raw.replace("cassandra://", "", 1)
    host_part = raw
    if "://" in raw:
        host_part = raw.split("://", 1)[1]
    items = [item.strip() for item in host_part.split(",") if item.strip()]
    cleaned: List[str] = []
    for item in items:
        if item:
            cleaned.append(item)
    return cleaned or ["127.0.0.1"]


def load_config() -> CassandraConfig:
    contact_points = _parse_contact_points(os.environ.get("CASSANDRA_CONTACT_POINTS"))
    port = int(os.environ.get("CASSANDRA_PORT", "9042"))
    username = os.environ.get("CASSANDRA_USERNAME")
    password = os.environ.get("CASSANDRA_PASSWORD")
    rt_keyspace = os.environ.get("CASSANDRA_KEYSPACE_RT", "pldt_rt")
    audit_keyspace = os.environ.get("CASSANDRA_KEYSPACE_AUDIT", "pldt_audit")
    enabled_env = os.environ.get("ENABLE_CASSANDRA", "").lower()
    disabled_env = os.environ.get("DISABLE_CASSANDRA", "").lower()
    enabled = True
    if disabled_env in {"1", "true", "yes"}:
        enabled = False
    elif enabled_env in {"0", "false", "no"}:
        enabled = False
    return CassandraConfig(
        contact_points=contact_points,
        port=port,
        username=username,
        password=password,
        rt_keyspace=rt_keyspace,
        audit_keyspace=audit_keyspace,
        enabled=enabled,
    )


def _normalize_amount(amount: float) -> Decimal:
    return Decimal(str(amount))


def _json_dumps(obj: Optional[Dict[str, Any]]) -> Optional[str]:
    if obj is None:
        return None
    try:
        return json.dumps(obj)
    except (TypeError, ValueError):
        return None


def _json_loads(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except (TypeError, ValueError):
        pass
    return None


def _cassandra_ts(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class CassandraSessionManager:
    """
    Centralizes the lifecycle of the Cassandra cluster and associated sessions.
    """

    def __init__(self, config: Optional[CassandraConfig] = None):
        self.config = config or load_config()
        self.cluster: Optional[ClusterType] = None
        self.session_rt: Optional[SessionType] = None
        self.session_audit: Optional[SessionType] = None

    def connect(self) -> None:
        if not self.config.enabled:
            logger.info("Cassandra integration disabled via environment.")
            self.shutdown()
            return
        if Cluster is None:
            logger.warning("cassandra-driver not installed; running in memory-only mode.")
            self.shutdown()
            return
        try:
            auth_provider: Optional[PlainTextAuthProviderType] = None
            if self.config.username and self.config.password and PlainTextAuthProvider:
                auth_provider = PlainTextAuthProvider(
                    username=self.config.username,
                    password=self.config.password,
                )

            contact_points: List[str] = []
            port = self.config.port
            for cp in self.config.contact_points:
                if ":" in cp:
                    host, _, port_str = cp.partition(":")
                    contact_points.append(host.strip())
                    if port_str.strip().isdigit():
                        port = int(port_str.strip())
                else:
                    contact_points.append(cp)

            cluster_kwargs = {
                "contact_points": contact_points,
                "port": port,
            }
            if auth_provider:
                cluster_kwargs["auth_provider"] = auth_provider

            cluster_instance = Cluster(**cluster_kwargs)  # type: ignore[arg-type]
            if cluster_instance is None:
                raise CassandraUnavailable("Unable to instantiate Cassandra cluster.")

            session_rt = cluster_instance.connect(self.config.rt_keyspace)
            session_audit = cluster_instance.connect(self.config.audit_keyspace)

            self.cluster = cluster_instance
            self.session_rt = session_rt
            self.session_audit = session_audit
            self.config.port = port
            logger.info(
                "Connected to Cassandra at %s:%s (keyspace=%s)",
                contact_points,
                port,
                self.config.rt_keyspace,
            )
        except Exception:
            logger.exception("Failed to connect to Cassandra. Falling back to in-memory store.")
            self.shutdown()

    def available(self) -> bool:
        return self.session_rt is not None

    def shutdown(self) -> None:
        try:
            if self.cluster:
                self.cluster.shutdown()
        except Exception:
            logger.exception("Error while shutting down Cassandra cluster.")
        finally:
            self.cluster = None
            self.session_rt = None
            self.session_audit = None


__all__ = [
    "CassandraConfig",
    "CassandraSessionManager",
    "CassandraUnavailable",
    "ClusterType",
    "PreparedStatementType",
    "SessionType",
    "SimpleStatement",
    "_cassandra_ts",
    "_json_dumps",
    "_json_loads",
    "_normalize_amount",
    "load_config",
    "logger",
]
