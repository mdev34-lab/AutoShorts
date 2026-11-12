"""New replacement regression qualification for the supplied public HTTP source.

This is newly authored test source, not recovered historical test bytes or a
claim about a historical test count. Client, adapter, pool, and connection
constructors remain real. Session.request/send are denied except for explicit
test-local response doubles. Socket entry points and native TLS are denied;
adapter integration stops at a test-local TLS sentinel after DNS/peer checks.

Run qualification externally under the unchanged startup deny and network
namespace. These tests neither install a replacement bootstrap nor relax that
boundary. Mocked networking checks source logic, not remote TLS/certificate
interoperability, actual HTTP wire behavior, or operating-system isolation.
No user assets, external providers, models, or TTS are used.
"""

from __future__ import annotations

import socket
import ssl
from unittest.mock import Mock, call, patch

import pytest
import requests

from autoshorts.modules import public_http as http

DDG_URL = "https://duckduckgo.com/"
PHOTO_HOST = "upload.wikimedia.org"
PHOTO_URL = "https://upload.wikimedia.org/commons/start.jpg"
PHOTO_ALLOWLIST = ((PHOTO_HOST, "/commons/"),)
PUBLIC_V4 = "8.8.8.8"
OTHER_PUBLIC_V4 = "1.1.1.1"
PUBLIC_V6 = "2606:4700:4700::1111"
SHARED_STORAGE_HOSTS = (
    "s3.amazonaws.com",
    "storage.googleapis.com",
    "blob.core.windows.net",
    "cloudfront.net",
    "objects.githubusercontent.com",
    "r2.cloudflarestorage.com",
)
NONPUBLIC_ANSWERS = (
    "0.0.0.0",
    "127.0.0.1",
    "10.1.2.3",
    "172.16.0.1",
    "192.168.0.1",
    "169.254.169.254",
    "100.64.0.1",
    "192.0.2.1",
    "198.51.100.1",
    "203.0.113.1",
    "224.0.0.1",
    "240.0.0.1",
    "255.255.255.255",
    "::",
    "::1",
    "fc00::1",
    "fd12::1",
    "fe80::1",
    "ff02::1",
    "2001:db8::1",
    "::ffff:127.0.0.1",
    "::ffff:10.1.2.3",
    "::ffff:169.254.169.254",
    "::ffff:192.0.2.1",
)


class _StoppedBeforeTLSError(RuntimeError):
    """Test-only stop: the real adapter reached TLS, but no TLS is performed."""


def _deny_unexpected_transport(*_args, **_kwargs):
    raise AssertionError("Unexpected transport: an explicit local double is required")


@pytest.fixture(autouse=True)
def deny_live_transport():
    """Additional local denials; never enable or replace the startup boundary."""
    with (
        patch.object(
            requests.Session, "request", side_effect=_deny_unexpected_transport
        ) as request,
        patch.object(
            requests.Session, "send", side_effect=_deny_unexpected_transport
        ) as send,
        patch.object(
            socket, "getaddrinfo", side_effect=_deny_unexpected_transport
        ) as dns,
        patch.object(
            socket, "create_connection", side_effect=_deny_unexpected_transport
        ) as connect,
        patch.object(
            socket, "socket", side_effect=_deny_unexpected_transport
        ) as raw_socket,
        patch.object(
            ssl.SSLContext, "wrap_socket", side_effect=_deny_unexpected_transport
        ) as native_tls,
        patch.object(
            ssl.SSLContext, "wrap_bio", side_effect=_deny_unexpected_transport
        ) as native_bio,
        patch.object(
            http.urllib3.connection,
            "ssl_wrap_socket",
            side_effect=_deny_unexpected_transport,
        ) as tls,
    ):
        guards = {
            "request": request,
            "send": send,
            "dns": dns,
            "connect": connect,
            "raw_socket": raw_socket,
            "native_tls": native_tls,
            "native_bio": native_bio,
            "tls": tls,
        }
        yield guards
        for guard in guards.values():
            guard.assert_not_called()


@pytest.fixture
def make_client():
    """Construct real clients and close their real sessions without sending."""
    clients = []

    def create(**kwargs):
        result = http.BoundedPublicClient(**kwargs)
        clients.append(result)
        return result

    yield create
    for result in clients:
        result.session.close()


@pytest.fixture
def client(make_client):
    return make_client()


def _response(status=200, *, headers=None, chunks=(b"data",)):
    response = Mock(name="synthetic_response")
    response.status_code = status
    response.headers = (
        {"Content-Type": "image/jpeg"} if headers is None else dict(headers)
    )
    response.iter_content.return_value = iter(chunks)
    return response


def _fake_socket(peer):
    result = Mock(name="synthetic_socket", spec=["getpeername", "close"])
    result.getpeername.return_value = (peer, 443)
    return result


def _dns_rows(answers):
    """Make local getaddrinfo records, including deliberately malformed answers."""
    rows = []
    for answer in answers:
        is_v6 = isinstance(answer, str) and ":" in answer
        family = socket.AF_INET6 if is_v6 else socket.AF_INET
        address = (answer, 443, 0, 0) if is_v6 else (answer, 443)
        rows.append((family, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", address))
    return rows


def _poison_session(session):
    """Use synthetic values only; never inspect credentials or environment."""
    session.trust_env = True
    session.auth = ("synthetic-user", "synthetic-password")
    session.proxies.update(
        {"http": "http://proxy.invalid", "https": "http://proxy.invalid"}
    )
    session.headers.update(
        {
            "Authorization": "synthetic-authorization",
            "Proxy-Authorization": "synthetic-proxy-authorization",
            "Cookie": "synthetic-header-cookie=value",
            "X-Synthetic-Ambient": "remove-me",
        }
    )
    session.cookies.set("synthetic-cookie", "remove-me")
    session.verify = False


def _assert_clean_session(session):
    assert session.trust_env is False
    assert session.auth is None
    assert dict(session.proxies) == {}
    assert dict(session.headers) == {}
    assert len(session.cookies) == 0
    assert session.verify is True


def _assert_bounded_request(method, options):
    assert method == "GET"
    assert set(options) == {
        "params",
        "headers",
        "allow_redirects",
        "timeout",
        "stream",
        "proxies",
    }
    assert options["headers"] == {
        "User-Agent": "autoshorts/1.0 (public bounded client; no auth)",
        "Accept-Encoding": "identity",
    }
    assert options["allow_redirects"] is False
    assert options["stream"] is True
    assert options["proxies"] == {}
    connect_timeout, read_timeout = options["timeout"]
    assert 0 < connect_timeout <= http.MAX_TIMEOUT_CEILING
    assert read_timeout == connect_timeout


class TestURLSecurityDenials:
    @pytest.mark.parametrize("scheme", ("http", "ftp", "file"))
    def test_reject_http_scheme(self, scheme):
        with pytest.raises(http.URLSecurityError, match="Only HTTPS is permitted"):
            http.assert_safe_https_url(f"{scheme}://duckduckgo.com/")

    @pytest.mark.parametrize(
        "url",
        (
            "https://duckduckgo.com:8443/",
            "https://upload.wikimedia.org:8080/image.jpg",
            "https://example.com:80/",
        ),
    )
    def test_reject_non_443_ports(self, url):
        with pytest.raises(http.URLSecurityError, match="Only port 443 is permitted"):
            http.assert_safe_https_url(url)

    @pytest.mark.parametrize(
        "url",
        (
            "https://user:pass@duckduckgo.com/",
            "https://token@upload.wikimedia.org/img.jpg",
            "https://user:@example.com/",
            "https://:password@example.com/",
        ),
    )
    def test_reject_url_credentials(self, url):
        with pytest.raises(http.URLSecurityError, match="URL credentials"):
            http.assert_safe_https_url(url)

    @pytest.mark.parametrize(
        "url",
        (
            "https://google.com/",
            "https://www.duckduckgo.com/",
            "https://duckduckgo.com.evil.example/",
            "https://duckduckgo.com/search",
            "https://duckduckgo.com/html/",
            "https://duckduckgo.com/i.js/",
            "https://duckduckgo.com/I.JS",
        ),
    )
    def test_reject_non_ddg_endpoints(self, url):
        with pytest.raises(http.URLSecurityError):
            http.validate_ddg_endpoint(url)

    @pytest.mark.parametrize("url", (None, 7, "", "   "))
    def test_nonempty_string_required(self, url):
        with pytest.raises(http.URLSecurityError):
            http.assert_safe_https_url(url)

    @pytest.mark.parametrize("url", (DDG_URL, "https://duckduckgo.com/i.js?q=mountain"))
    def test_exact_ddg_endpoints_allowed(self, url):
        assert http.validate_ddg_endpoint(url) is None

    @pytest.mark.parametrize(
        "url, expected",
        (
            ("HTTPS://EXAMPLE.COM:443", ("example.com", "/")),
            ("https://example.com./photos/a.jpg", ("example.com", "/photos/a.jpg")),
            (
                "https://example.com/photos/a%20b.jpg?q=x",
                ("example.com", "/photos/a%20b.jpg"),
            ),
        ),
    )
    def test_safe_host_and_path_normalization(self, url, expected):
        assert http.assert_safe_https_url(url) == expected


class TestURLHardening:
    @pytest.mark.parametrize("character", (" ", "\t", "\n", "\r"))
    def test_reject_whitespace(self, character):
        with pytest.raises(http.URLSecurityError, match="whitespace"):
            http.assert_safe_https_url("https://example.com/a" + character + "b")

    @pytest.mark.parametrize("character", ("\x00", "\x01", "\x1f", "\x7f"))
    def test_reject_control_chars(self, character):
        # Some control characters are also whitespace; either denial is valid.
        with pytest.raises(http.URLSecurityError, match="whitespace|control"):
            http.assert_safe_https_url("https://example.com/a" + character)

    def test_reject_backslash(self):
        with pytest.raises(http.URLSecurityError, match="backslash"):
            http.assert_safe_https_url("https://example.com/a\\b")

    @pytest.mark.parametrize("suffix", ("#", "#fragment"))
    def test_reject_hash(self, suffix):
        with pytest.raises(http.URLSecurityError, match="#"):
            http.assert_safe_https_url("https://example.com/a" + suffix)

    @pytest.mark.parametrize(
        "host",
        (
            "127.0.0.1",
            "2130706433",
            "0x7f.1",
            "[::1]",
            "[2606:4700:4700::1111]",
            "localhost",
            "a.localhost",
            "x.local",
            "x.internal",
            "a.123",
        ),
    )
    def test_reject_ip_literals_and_local_or_non_dns_hosts(self, host):
        with pytest.raises(http.URLSecurityError):
            http.assert_safe_https_url(f"https://{host}/x")

    @pytest.mark.parametrize(
        "path",
        (
            "/a/%2e%2e/b",
            "/a/%2E/b",
            "/a/%2F/b",
            "/a/%2f/b",
            "/a/%5c/b",
            "/a/%5C/b",
            "/a/%252e/b",
            "/a/./b",
            "/a/../b",
        ),
    )
    def test_reject_dot_segments_and_encoded_traversal(self, path):
        with pytest.raises(http.URLSecurityError):
            http.assert_safe_https_url("https://example.com" + path)


class TestPhotoAllowlistDenials:
    def test_reject_empty_allowlist(self):
        with pytest.raises(
            http.AllowlistDeniedError, match="must be supplied and non-empty"
        ):
            http.validate_photo_allowlist([])

    @pytest.mark.parametrize("host", ("*.wikimedia.org", "upload?.wikimedia.org", ""))
    def test_reject_wildcards(self, host):
        with pytest.raises(
            http.AllowlistDeniedError, match="Wildcards and empty hosts forbidden"
        ):
            http.validate_photo_allowlist([(host, "/images/")])

    @pytest.mark.parametrize("host", SHARED_STORAGE_HOSTS)
    @pytest.mark.parametrize("prefix", ("/", "", "   "))
    def test_reject_shared_storage_root(self, host, prefix):
        for candidate in (host, "bucket." + host):
            with pytest.raises(
                http.AllowlistDeniedError, match="requires explicit specific pathprefix"
            ):
                http.validate_photo_allowlist([(candidate, prefix)])

    @pytest.mark.parametrize(
        "url",
        (
            "https://otherhost.org/commons/img.jpg",
            "https://sub.upload.wikimedia.org/commons/img.jpg",
            "https://upload.wikimedia.org.evil.example/commons/img.jpg",
            "https://upload.wikimedia.org/private/img.jpg",
            "https://upload.wikimedia.org/private?path=/commons/img.jpg",
        ),
    )
    def test_reject_url_not_in_allowlist(self, url):
        with pytest.raises(
            http.AllowlistDeniedError, match="not permitted by explicit photo allowlist"
        ):
            http.validate_photo_target(url, PHOTO_ALLOWLIST)

    @pytest.mark.parametrize(
        "entry",
        (
            None,
            "example.com/photos",
            {"host": "example.com"},
            ("example.com",),
            ("example.com", "/x", "/y"),
            (7, "/x"),
            ("example.com", 7),
        ),
    )
    def test_malformed_entries_denied(self, entry):
        with pytest.raises(http.AllowlistDeniedError):
            http.validate_photo_allowlist([entry])

    @pytest.mark.parametrize(
        "host",
        (
            "127.0.0.1",
            "::1",
            "2130706433",
            "0x7f.1",
            "localhost",
            "a.localhost",
            "x.local",
            "x.internal",
            "https://example.com",
            "example.com:443",
        ),
    )
    def test_ip_local_and_nondns_allowlist_hosts_denied(self, host):
        with pytest.raises(http.AllowlistDeniedError):
            http.validate_photo_allowlist([(host, "/x")])

    @pytest.mark.parametrize("unsafe", ("..", "%", "?", "#", "\\"))
    def test_unsafe_prefix_characters_denied(self, unsafe):
        with pytest.raises(http.AllowlistDeniedError):
            http.validate_photo_allowlist([("example.com", "/photos/" + unsafe)])

    @pytest.mark.parametrize(
        "host", ("127.0.0.1", "[::1]", "2130706433", "0x7f.1", "localhost")
    )
    def test_loopback_denied_even_with_matching_bad_allowlist(self, host):
        with pytest.raises((http.URLSecurityError, http.AllowlistDeniedError)):
            http.validate_photo_target(f"https://{host}/x", [(host, "/x")])

    @pytest.mark.parametrize(
        "suffix",
        (
            "../other/file.jpg",
            "./file.jpg",
            "%2e%2e/file.jpg",
            "%2Ffile",
            "%5cfile",
            "%252efile",
        ),
    )
    def test_target_traversal_denied(self, suffix):
        with pytest.raises(http.URLSecurityError):
            http.validate_photo_target(
                "https://storage.googleapis.com/owner/photos/" + suffix,
                [("storage.googleapis.com", "/owner/photos/")],
            )

    @pytest.mark.parametrize("prefix", ("/photos", "/photos/"))
    def test_segment_boundary_and_exact_prefix(self, prefix):
        allowlist = [("example.com", prefix)]
        for path in ("/photos", "/photos/", "/photos/a.jpg"):
            assert http.validate_photo_target(
                "https://example.com" + path, allowlist
            ) == ("example.com", path)
        for path in ("/photos-evil/a.jpg", "/photos2/a.jpg", "/PHOTOS/a.jpg"):
            with pytest.raises(http.AllowlistDeniedError):
                http.validate_photo_target("https://example.com" + path, allowlist)

    def test_allowlist_normalization(self):
        assert http.validate_photo_allowlist(
            [(" Upload.Wikimedia.Org. ", " commons/photos ")]
        ) == [("upload.wikimedia.org", "/commons/photos")]

    @pytest.mark.parametrize("host", SHARED_STORAGE_HOSTS)
    def test_specific_shared_storage_prefix_allowed(self, host):
        assert http.validate_photo_target(
            f"https://{host}/owner/photos/a.jpg", [(host, "/owner/photos/")]
        ) == (host, "/owner/photos/a.jpg")

    def test_nonshared_explicit_root_allowed(self):
        assert http.validate_photo_allowlist([("example.com", "")]) == [
            ("example.com", "/")
        ]
        assert http.validate_photo_target(
            "https://example.com/a.jpg", [("example.com", "/")]
        ) == ("example.com", "/a.jpg")

    @pytest.mark.parametrize("kind", ("ddg", "photo"))
    def test_invalid_initial_target_never_requests(
        self, client, deny_live_transport, kind
    ):
        with pytest.raises(http.URLSecurityError):
            if kind == "ddg":
                client.get_ddg("http://duckduckgo.com/")
            else:
                client.get_photo(
                    "http://upload.wikimedia.org/commons/a.jpg", PHOTO_ALLOWLIST
                )
        deny_live_transport["request"].assert_not_called()
        deny_live_transport["dns"].assert_not_called()


class TestDNSBoundary:
    def test_default_resolver_uses_stream_lookup_and_preserves_order(self):
        answers = [PUBLIC_V6, PUBLIC_V4, PUBLIC_V4]
        with patch.object(
            socket, "getaddrinfo", return_value=_dns_rows(answers)
        ) as dns:
            assert http._default_resolver(PHOTO_HOST, 443) == answers
        dns.assert_called_once_with(PHOTO_HOST, 443, type=socket.SOCK_STREAM)

    @pytest.mark.parametrize("answers", ([], (), None))
    def test_empty_dns_denied(self, answers):
        resolver = Mock(return_value=answers)
        with pytest.raises(http.URLSecurityError, match="no answers"):
            http._resolve_public(PHOTO_HOST, 443, resolver)
        resolver.assert_called_once_with(PHOTO_HOST, 443)

    @pytest.mark.parametrize(
        "answer", (None, 7, b"8.8.8.8", "", "not-an-ip", " 8.8.8.8", "8.8.8.8:443")
    )
    def test_malformed_dns_answers_denied(self, answer):
        resolver = Mock(return_value=[answer])
        with pytest.raises(http.URLSecurityError, match="Unparsable DNS answer"):
            http._resolve_public(PHOTO_HOST, 443, resolver)
        resolver.assert_called_once_with(PHOTO_HOST, 443)

    @pytest.mark.parametrize("answer", NONPUBLIC_ANSWERS)
    def test_every_nonpublic_address_class_denied(self, answer):
        resolver = Mock(return_value=[answer])
        with pytest.raises(http.URLSecurityError, match="not publicly routable"):
            http._resolve_public(PHOTO_HOST, 443, resolver)
        resolver.assert_called_once_with(PHOTO_HOST, 443)

    @pytest.mark.parametrize(
        "bad", ("10.0.0.1", "::1", "::ffff:192.168.0.1", "bad-ip", None)
    )
    @pytest.mark.parametrize("bad_first", (False, True))
    def test_mixed_answers_are_all_validated_before_selecting_first(
        self, bad, bad_first
    ):
        answers = (
            [bad, PUBLIC_V4, PUBLIC_V6] if bad_first else [PUBLIC_V4, PUBLIC_V6, bad]
        )
        resolver = Mock(return_value=answers)
        with pytest.raises(http.URLSecurityError):
            http._resolve_public(PHOTO_HOST, 443, resolver)
        resolver.assert_called_once_with(PHOTO_HOST, 443)

    @pytest.mark.parametrize(
        "answers",
        (
            [PUBLIC_V4],
            [PUBLIC_V6],
            [PUBLIC_V4, PUBLIC_V6],
            [PUBLIC_V6, PUBLIC_V4, PUBLIC_V6],
        ),
    )
    def test_all_global_answers_return_first_without_reordering(self, answers):
        resolver = Mock(return_value=answers)
        assert http._resolve_public(PHOTO_HOST, 443, resolver) == answers[0]
        resolver.assert_called_once_with(PHOTO_HOST, 443)

    @pytest.mark.parametrize(
        "error",
        (socket.gaierror("synthetic DNS failure"), RuntimeError("synthetic failure")),
    )
    def test_resolution_error_is_closed_failure_with_cause(self, error):
        resolver = Mock(side_effect=error)
        with pytest.raises(
            http.URLSecurityError, match="DNS resolution failed"
        ) as caught:
            http._resolve_public(PHOTO_HOST, 443, resolver)
        assert caught.value.__cause__ is error
        resolver.assert_called_once_with(PHOTO_HOST, 443)


class TestPinnedConnection:
    @pytest.mark.parametrize(
        "validated, peer",
        (
            (PUBLIC_V4, PUBLIC_V4),
            (PUBLIC_V6, PUBLIC_V6),
            (PUBLIC_V6, "2606:4700:4700:0000:0000:0000:0000:1111"),
        ),
    )
    def test_connects_only_to_validated_ip_and_preserves_hostname(
        self, validated, peer
    ):
        resolver = Mock(side_effect=[[validated], ["127.0.0.1"]])
        connection = http._PinnedHTTPSConnection(
            PHOTO_HOST, port=443, timeout=5, resolver=resolver
        )
        fake = _fake_socket(peer)
        with patch.object(socket, "create_connection", return_value=fake) as connect:
            assert connection._new_conn() is fake
        resolver.assert_called_once_with(PHOTO_HOST, 443)
        connect.assert_called_once_with((validated, 443), 5)
        assert connection.host == PHOTO_HOST
        assert getattr(connection, "_dns_host", connection.host) == PHOTO_HOST
        fake.getpeername.assert_called_once_with()
        fake.close.assert_not_called()

    @pytest.mark.parametrize(
        "answers", ([], ["127.0.0.1"], [PUBLIC_V4, "10.0.0.1"], ["bad-ip"])
    )
    def test_dns_failure_precedes_socket_creation(self, answers):
        resolver = Mock(return_value=answers)
        connection = http._PinnedHTTPSConnection(
            PHOTO_HOST, port=443, timeout=5, resolver=resolver
        )
        with patch.object(
            socket, "create_connection", return_value=_fake_socket(PUBLIC_V4)
        ) as connect:
            with pytest.raises(http.URLSecurityError):
                connection._new_conn()
        resolver.assert_called_once_with(PHOTO_HOST, 443)
        connect.assert_not_called()

    @pytest.mark.parametrize(
        "peer", (OTHER_PUBLIC_V4, "127.0.0.1", "10.0.0.1", "::ffff:8.8.8.8")
    )
    def test_peer_mismatch_closes_socket(self, peer):
        resolver = Mock(return_value=[PUBLIC_V4])
        connection = http._PinnedHTTPSConnection(
            PHOTO_HOST, port=443, timeout=5, resolver=resolver
        )
        fake = _fake_socket(peer)
        with patch.object(socket, "create_connection", return_value=fake) as connect:
            with pytest.raises(
                http.URLSecurityError, match="does not match validated IP"
            ):
                connection._new_conn()
        connect.assert_called_once_with((PUBLIC_V4, 443), 5)
        fake.close.assert_called_once_with()

    def test_unparsable_peer_closes_socket(self):
        connection = http._PinnedHTTPSConnection(
            PHOTO_HOST, port=443, timeout=5, resolver=Mock(return_value=[PUBLIC_V4])
        )
        fake = _fake_socket("not-an-ip")
        with patch.object(socket, "create_connection", return_value=fake):
            with pytest.raises(ValueError):
                connection._new_conn()
        fake.close.assert_called_once_with()

    def test_getpeername_error_closes_socket(self):
        error = OSError("synthetic peer lookup failure")
        connection = http._PinnedHTTPSConnection(
            PHOTO_HOST, port=443, timeout=5, resolver=Mock(return_value=[PUBLIC_V4])
        )
        fake = _fake_socket(PUBLIC_V4)
        fake.getpeername.side_effect = error
        with patch.object(socket, "create_connection", return_value=fake):
            with pytest.raises(OSError) as caught:
                connection._new_conn()
        assert caught.value is error
        fake.close.assert_called_once_with()

    @pytest.mark.parametrize(
        "error",
        (OSError("synthetic connect failure"), socket.timeout("synthetic timeout")),
    )
    def test_socket_creation_error_propagates_without_retry_or_reresolution(
        self, error
    ):
        resolver = Mock(side_effect=[[PUBLIC_V4], ["127.0.0.1"]])
        connection = http._PinnedHTTPSConnection(
            PHOTO_HOST, port=443, timeout=5, resolver=resolver
        )
        with patch.object(socket, "create_connection", side_effect=error) as connect:
            with pytest.raises(OSError) as caught:
                connection._new_conn()
        assert caught.value is error
        resolver.assert_called_once_with(PHOTO_HOST, 443)
        connect.assert_called_once_with((PUBLIC_V4, 443), 5)


class TestAdapterBoundary:
    def test_real_constructor_installs_private_pinned_and_deny_adapters(
        self, make_client
    ):
        first_resolver = Mock(return_value=[PUBLIC_V4])
        second_resolver = Mock(return_value=[PUBLIC_V6])
        first = make_client(resolver=first_resolver)
        second = make_client(resolver=second_resolver)
        assert isinstance(first.session, requests.Session)
        first_adapter = first.session.get_adapter(PHOTO_URL)
        second_adapter = second.session.get_adapter(PHOTO_URL)
        assert isinstance(first_adapter, http._PinnedAdapter)
        assert isinstance(
            first.session.get_adapter("http://example.com/"), http._DenyHttpAdapter
        )
        assert first_adapter.resolver is first_resolver
        assert second_adapter.resolver is second_resolver
        first_pool = first_adapter.poolmanager.pool_classes_by_scheme["https"]
        second_pool = second_adapter.poolmanager.pool_classes_by_scheme["https"]
        assert first_pool is not second_pool
        assert issubclass(first_pool.ConnectionCls, http._PinnedHTTPSConnection)
        assert first_pool.ConnectionCls.resolver is first_resolver
        assert second_pool.ConnectionCls.resolver is second_resolver
        global_pool = http.urllib3.poolmanager.pool_classes_by_scheme["https"]
        assert first_pool is not global_pool
        assert second_pool is not global_pool
        first_resolver.assert_not_called()
        second_resolver.assert_not_called()

    def test_real_http_adapter_denies_before_dns_or_socket(self, client):
        url = "http://upload.wikimedia.org/commons/a.jpg"
        prepared = requests.Request("GET", url).prepare()
        with (
            patch.object(
                socket, "getaddrinfo", return_value=_dns_rows([PUBLIC_V4])
            ) as dns,
            patch.object(
                socket, "create_connection", return_value=_fake_socket(PUBLIC_V4)
            ) as connect,
        ):
            with pytest.raises(
                http.URLSecurityError, match="Plain HTTP is not permitted"
            ):
                client.session.get_adapter(url).send(
                    prepared, timeout=(5, 5), proxies={}
                )
        dns.assert_not_called()
        connect.assert_not_called()

    @pytest.mark.parametrize(
        "answers",
        (
            [],
            ["not-an-ip"],
            [None],
            ["127.0.0.1"],
            ["::1"],
            [PUBLIC_V4, "10.0.0.1"],
            [PUBLIC_V6, "fc00::1"],
            ["::ffff:192.168.0.1"],
        ),
    )
    def test_real_https_adapter_enforces_default_dns_before_connection(
        self, client, answers
    ):
        prepared = requests.Request("GET", PHOTO_URL).prepare()
        fake = _fake_socket(PUBLIC_V4)
        with (
            patch.object(socket, "getaddrinfo", return_value=_dns_rows(answers)) as dns,
            patch.object(socket, "create_connection", return_value=fake) as connect,
        ):
            with pytest.raises(http.URLSecurityError):
                client.session.get_adapter(PHOTO_URL).send(
                    prepared, timeout=(5, 5), proxies={}
                )
        dns.assert_called_once_with(PHOTO_HOST, 443, type=socket.SOCK_STREAM)
        connect.assert_not_called()
        fake.close.assert_not_called()

    def test_real_https_adapter_denies_dns_socket_error(self, client):
        error = socket.gaierror("synthetic DNS socket error")
        prepared = requests.Request("GET", PHOTO_URL).prepare()
        with (
            patch.object(socket, "getaddrinfo", side_effect=error) as dns,
            patch.object(
                socket, "create_connection", return_value=_fake_socket(PUBLIC_V4)
            ) as connect,
        ):
            with pytest.raises(
                http.URLSecurityError, match="DNS resolution failed"
            ) as caught:
                client.session.get_adapter(PHOTO_URL).send(
                    prepared, timeout=(5, 5), proxies={}
                )
        assert caught.value.__cause__ is error
        dns.assert_called_once_with(PHOTO_HOST, 443, type=socket.SOCK_STREAM)
        connect.assert_not_called()

    @pytest.mark.parametrize("peer", (OTHER_PUBLIC_V4, "127.0.0.1"))
    def test_real_https_adapter_closes_mismatched_peer_before_tls(self, client, peer):
        prepared = requests.Request("GET", PHOTO_URL).prepare()
        fake = _fake_socket(peer)
        with (
            patch.object(
                socket, "getaddrinfo", return_value=_dns_rows([PUBLIC_V4])
            ) as dns,
            patch.object(socket, "create_connection", return_value=fake) as connect,
        ):
            with pytest.raises(
                http.URLSecurityError, match="does not match validated IP"
            ):
                client.session.get_adapter(PHOTO_URL).send(
                    prepared, timeout=(5, 5), proxies={}
                )
        dns.assert_called_once_with(PHOTO_HOST, 443, type=socket.SOCK_STREAM)
        connect.assert_called_once_with((PUBLIC_V4, 443), 5)
        fake.close.assert_called_once_with()

    @pytest.mark.parametrize("address", (PUBLIC_V4, PUBLIC_V6))
    def test_real_https_adapter_pins_once_and_retains_tls_hostname(
        self, client, address
    ):
        prepared = requests.Request("GET", PHOTO_URL).prepare()
        fake = _fake_socket(address)
        with (
            patch.object(
                socket,
                "getaddrinfo",
                side_effect=[_dns_rows([address]), _dns_rows(["127.0.0.1"])],
            ) as dns,
            patch.object(socket, "create_connection", return_value=fake) as connect,
            patch.object(
                http.urllib3.connection,
                "ssl_wrap_socket",
                side_effect=_StoppedBeforeTLSError,
            ) as tls,
        ):
            with pytest.raises(_StoppedBeforeTLSError):
                client.session.get_adapter(PHOTO_URL).send(
                    prepared, timeout=(5, 5), proxies={}
                )
        dns.assert_called_once_with(PHOTO_HOST, 443, type=socket.SOCK_STREAM)
        connect.assert_called_once_with((address, 443), 5)
        tls.assert_called_once()
        assert tls.call_args.kwargs["sock"] is fake
        assert tls.call_args.kwargs["server_hostname"] == PHOTO_HOST
        assert prepared.url == PHOTO_URL
        assert (
            client.session.get_adapter(PHOTO_URL)
            .poolmanager.connection_from_url(PHOTO_URL)
            .host
            == PHOTO_HOST
        )

    def test_each_redirect_hop_gets_fresh_dns_and_clean_session(self, client):
        """Only HTTP responses/TLS are synthetic; adapter/pool/DNS logic is real."""
        first = _response(302, headers={"Location": "/i.js?next=1"})
        second = _response(chunks=(b"ok",))
        responses = iter((first, second))
        sockets = [_fake_socket(PUBLIC_V4), _fake_socket(OTHER_PUBLIC_V4)]
        prepared_requests = []
        _poison_session(client.session)

        def local_request(method, url, **options):
            _assert_clean_session(client.session)
            _assert_bounded_request(method, options)
            prepared = requests.Request(
                method, url, headers=options["headers"], params=options["params"]
            ).prepare()
            assert "Authorization" not in prepared.headers
            assert "Proxy-Authorization" not in prepared.headers
            assert "Cookie" not in prepared.headers
            with pytest.raises(_StoppedBeforeTLSError):
                client.session.get_adapter(url).send(
                    prepared, timeout=(5, 5), proxies={}
                )
            prepared_requests.append(prepared)
            response = next(responses)
            if response is first:
                _poison_session(client.session)
            else:
                client.session.cookies.set("synthetic-response-cookie", "discard-me")
            return response

        with (
            patch.object(
                socket,
                "getaddrinfo",
                side_effect=[_dns_rows([PUBLIC_V4]), _dns_rows([OTHER_PUBLIC_V4])],
            ) as dns,
            patch.object(socket, "create_connection", side_effect=sockets) as connect,
            patch.object(
                http.urllib3.connection,
                "ssl_wrap_socket",
                side_effect=_StoppedBeforeTLSError,
            ) as tls,
            patch.object(
                client.session, "request", side_effect=local_request
            ) as request,
        ):
            assert (
                client.get_ddg(
                    DDG_URL,
                    params={"q": "synthetic public query"},
                    timeout=10,
                    max_redirects=1,
                )
                == b"ok"
            )
        assert (
            dns.call_args_list
            == [call("duckduckgo.com", 443, type=socket.SOCK_STREAM)] * 2
        )
        assert connect.call_args_list == [
            call((PUBLIC_V4, 443), 5),
            call((OTHER_PUBLIC_V4, 443), 5),
        ]
        assert [entry.kwargs["server_hostname"] for entry in tls.call_args_list] == [
            "duckduckgo.com"
        ] * 2
        assert len(prepared_requests) == 2
        assert request.call_args_list[0].kwargs["params"] == {
            "q": "synthetic public query"
        }
        assert request.call_args_list[1].kwargs["params"] is None
        first.close.assert_called_once_with()
        second.close.assert_called_once_with()
        _assert_clean_session(client.session)

    def test_same_host_redirect_rebinding_to_private_is_denied(self, client):
        first = _response(302, headers={"Location": "/i.js"})
        fake = _fake_socket(PUBLIC_V4)

        def local_request(method, url, **options):
            _assert_clean_session(client.session)
            _assert_bounded_request(method, options)
            prepared = requests.Request(
                method, url, headers=options["headers"]
            ).prepare()
            if url == DDG_URL:
                with pytest.raises(_StoppedBeforeTLSError):
                    client.session.get_adapter(url).send(
                        prepared, timeout=(5, 5), proxies={}
                    )
                return first
            return client.session.get_adapter(url).send(
                prepared, timeout=(5, 5), proxies={}
            )

        with (
            patch.object(
                socket,
                "getaddrinfo",
                side_effect=[_dns_rows([PUBLIC_V4]), _dns_rows(["127.0.0.1"])],
            ) as dns,
            patch.object(socket, "create_connection", return_value=fake) as connect,
            patch.object(
                http.urllib3.connection,
                "ssl_wrap_socket",
                side_effect=_StoppedBeforeTLSError,
            ) as tls,
            patch.object(
                client.session, "request", side_effect=local_request
            ) as request,
        ):
            with pytest.raises(http.URLSecurityError, match="not publicly routable"):
                client.get_ddg(DDG_URL, max_redirects=1)
        assert request.call_count == 2
        assert (
            dns.call_args_list
            == [call("duckduckgo.com", 443, type=socket.SOCK_STREAM)] * 2
        )
        connect.assert_called_once_with((PUBLIC_V4, 443), 5)
        tls.assert_called_once()
        first.close.assert_called_once_with()


class TestRedirectAndHopDenials:
    def test_redirect_to_insecure_target_denied(self, client):
        response = _response(302, headers={"Location": "http://duckduckgo.com/i.js"})
        with patch.object(client.session, "request", return_value=response) as request:
            with pytest.raises(http.URLSecurityError, match="Only HTTPS is permitted"):
                client.get_ddg(DDG_URL)
        request.assert_called_once()
        response.close.assert_called_once_with()
        response.iter_content.assert_not_called()

    def test_redirect_to_unauthorized_host_denied(self, client):
        response = _response(302, headers={"Location": "https://evil.example/img.jpg"})
        with patch.object(client.session, "request", return_value=response) as request:
            with pytest.raises(http.AllowlistDeniedError):
                client.get_photo(PHOTO_URL, PHOTO_ALLOWLIST)
        request.assert_called_once()
        response.close.assert_called_once_with()

    @pytest.mark.parametrize("limit", (0, 1, 2, 5))
    def test_redirect_limit_exceeded_denied(self, client, limit):
        responses = [
            _response(302, headers={"Location": "/i.js"}) for _ in range(limit + 1)
        ]
        with patch.object(client.session, "request", side_effect=responses) as request:
            with pytest.raises(
                http.RedirectDeniedError, match="Redirect limit .* exceeded"
            ):
                client.get_ddg(DDG_URL, max_redirects=limit)
        assert request.call_count == limit + 1
        for response in responses:
            response.close.assert_called_once_with()
            response.iter_content.assert_not_called()

    @pytest.mark.parametrize("headers", ({}, {"Location": ""}))
    def test_redirect_missing_location_denied(self, client, headers):
        response = _response(302, headers=headers)
        with patch.object(client.session, "request", return_value=response) as request:
            with pytest.raises(
                http.RedirectDeniedError, match="Redirect missing Location header"
            ):
                client.get_ddg(DDG_URL)
        request.assert_called_once()
        response.close.assert_called_once_with()

    @pytest.mark.parametrize("status", (301, 302, 303, 307, 308))
    def test_each_supported_redirect_is_manual_get_and_drops_original_params(
        self, client, status
    ):
        first = _response(status, headers={"Location": "/i.js?next=1"})
        second = _response(
            headers={"Content-Type": "application/json"}, chunks=(b"{}",)
        )
        params = {"q": "synthetic query", "iax": "images"}
        with patch.object(
            client.session, "request", side_effect=[first, second]
        ) as request:
            assert client.get_ddg(DDG_URL, params=params, max_redirects=1) == b"{}"
        assert request.call_count == 2
        assert request.call_args_list[0].args == ("GET", DDG_URL)
        assert request.call_args_list[0].kwargs["params"] == params
        assert request.call_args_list[1].args == (
            "GET",
            "https://duckduckgo.com/i.js?next=1",
        )
        assert request.call_args_list[1].kwargs["params"] is None
        for entry in request.call_args_list:
            _assert_bounded_request(entry.args[0], entry.kwargs)
        first.close.assert_called_once_with()
        first.iter_content.assert_not_called()
        second.close.assert_called_once_with()

    @pytest.mark.parametrize(
        "kind, location, error",
        (
            ("ddg", "https://other.example/", http.URLSecurityError),
            ("ddg", "/html/", http.URLSecurityError),
            ("ddg", "https://user:pass@duckduckgo.com/i.js", http.URLSecurityError),
            ("ddg", "https://duckduckgo.com:8443/i.js", http.URLSecurityError),
            ("photo", "/private/a.jpg", http.AllowlistDeniedError),
            ("photo", "/commons-evil/a.jpg", http.AllowlistDeniedError),
            # urljoin normalizes literal dot segments before allowlist validation.
            ("photo", "/commons/../private/a.jpg", http.AllowlistDeniedError),
            ("photo", "/commons/%2e%2e/a.jpg", http.URLSecurityError),
            ("photo", "//127.0.0.1/commons/a.jpg", http.URLSecurityError),
            (
                "photo",
                "http://upload.wikimedia.org/commons/a.jpg",
                http.URLSecurityError,
            ),
        ),
    )
    def test_every_hop_revalidates_before_another_request(
        self, client, kind, location, error
    ):
        response = _response(302, headers={"Location": location})
        with patch.object(client.session, "request", return_value=response) as request:
            with pytest.raises(error):
                if kind == "ddg":
                    client.get_ddg(DDG_URL)
                else:
                    client.get_photo(PHOTO_URL, PHOTO_ALLOWLIST)
        request.assert_called_once()
        response.close.assert_called_once_with()
        response.iter_content.assert_not_called()

    @pytest.mark.parametrize(
        "location",
        ("next.jpg", "/commons/next.jpg", "//upload.wikimedia.org/commons/next.jpg"),
    )
    def test_relative_photo_redirect_stays_within_prefix(self, client, location):
        first = _response(302, headers={"Location": location})
        second = _response(chunks=(b"image",))
        with patch.object(
            client.session, "request", side_effect=[first, second]
        ) as request:
            assert (
                client.get_photo(PHOTO_URL, PHOTO_ALLOWLIST, max_redirects=1)
                == b"image"
            )
        assert request.call_args_list[1].args == (
            "GET",
            "https://upload.wikimedia.org/commons/next.jpg",
        )
        assert all(entry.kwargs["params"] is None for entry in request.call_args_list)
        first.close.assert_called_once_with()
        second.close.assert_called_once_with()

    def test_cross_host_photo_redirect_requires_explicit_destination_prefix(
        self, client
    ):
        destination = "https://images.example.com/photos/next.jpg"
        allowlist = (*PHOTO_ALLOWLIST, ("images.example.com", "/photos/"))
        first = _response(302, headers={"Location": destination})
        second = _response(chunks=(b"image",))
        with patch.object(
            client.session, "request", side_effect=[first, second]
        ) as request:
            assert client.get_photo(PHOTO_URL, allowlist, max_redirects=1) == b"image"
        assert request.call_args_list[1].args == ("GET", destination)
        first.close.assert_called_once_with()
        second.close.assert_called_once_with()


class TestCookieAuthAndCapDenials:
    def test_trust_env_false_and_no_ambient_cookies(self, client):
        _assert_clean_session(client.session)

    def test_byte_cap_exceeded_denied(self, client):
        response = _response(chunks=(b"x" * 100,) * 10)
        with patch.object(client.session, "request", return_value=response):
            with pytest.raises(
                http.ByteCapExceededError, match="Response exceeded maximum byte cap"
            ):
                client.get_photo(PHOTO_URL, PHOTO_ALLOWLIST, max_bytes=500)
        response.iter_content.assert_called_once_with(chunk_size=32768)
        response.close.assert_called_once_with()


class TestSessionHygiene:
    def test_repeated_operations_reset_all_ambient_state_and_discard_response_cookies(
        self, client
    ):
        responses = []

        def local_request(method, url, **options):
            assert url == PHOTO_URL
            _assert_clean_session(client.session)
            _assert_bounded_request(method, options)
            client.session.cookies.set("synthetic-response-cookie", "discard-me")
            response = _response()

            def body(*, chunk_size):
                assert chunk_size == 32768
                assert len(client.session.cookies) == 0
                yield b"data"

            response.iter_content.side_effect = body
            responses.append(response)
            return response

        with patch.object(
            client.session, "request", side_effect=local_request
        ) as request:
            for _ in range(2):
                _poison_session(client.session)
                assert client.get_photo(PHOTO_URL, PHOTO_ALLOWLIST) == b"data"
                _assert_clean_session(client.session)
        assert request.call_count == 2
        for response in responses:
            response.close.assert_called_once_with()

    def test_explicit_get_options_and_identity_encoding(self, client):
        response = _response(headers={"Content-Type": "text/html"}, chunks=(b"search",))
        params = {"q": "synthetic public query"}
        with (
            patch.object(http.time, "monotonic", return_value=100.0),
            patch.object(client.session, "request", return_value=response) as request,
        ):
            assert (
                client.get_ddg(DDG_URL, params=params, timeout=2, max_redirects=0)
                == b"search"
            )
        request.assert_called_once()
        assert request.call_args.args == ("GET", DDG_URL)
        _assert_bounded_request("GET", request.call_args.kwargs)
        assert request.call_args.kwargs["params"] is params
        assert request.call_args.kwargs["timeout"] == (2.0, 2.0)
        response.close.assert_called_once_with()


class TestCapValidation:
    def test_fixed_ceiling_values_and_inclusive_boundaries(self):
        assert http.MAX_BYTES_CEILING == 20_971_520
        assert http.MAX_TIMEOUT_CEILING == 30.0
        assert http.MAX_REDIRECTS_CEILING == 5
        assert http.validate_caps(1, 0.01, 0) is None
        assert http.validate_caps(20_971_520, 30.0, 5) is None

    @pytest.mark.parametrize("kind", ("ddg", "photo"))
    @pytest.mark.parametrize(
        "option, bad",
        (
            ("max_bytes", 0),
            ("max_bytes", -1),
            ("max_bytes", 20_971_521),
            ("max_bytes", float("inf")),
            ("max_bytes", float("nan")),
            ("timeout", 0),
            ("timeout", -1),
            ("timeout", 30.01),
            ("timeout", float("inf")),
            ("timeout", float("-inf")),
            ("timeout", float("nan")),
            ("max_redirects", -1),
            ("max_redirects", 6),
            ("max_redirects", float("inf")),
            ("max_redirects", float("nan")),
        ),
    )
    def test_out_of_range_caps_denied_before_transport(
        self, client, deny_live_transport, kind, option, bad
    ):
        with pytest.raises(http.PublicHttpError):
            if kind == "ddg":
                client.get_ddg(DDG_URL, **{option: bad})
            else:
                client.get_photo(PHOTO_URL, PHOTO_ALLOWLIST, **{option: bad})
        deny_live_transport["request"].assert_not_called()
        deny_live_transport["dns"].assert_not_called()


class TestResponseBoundsAndCleanup:
    def test_happy_path_concatenates_chunks_ignoring_empty_keepalives(self, client):
        response = _response(
            headers={
                "Content-Type": "IMAGE/JPEG; charset=binary",
                "Content-Length": "10",
            },
            chunks=(b"", b"hello", b"", b"world"),
        )
        with patch.object(client.session, "request", return_value=response):
            assert (
                client.get_photo(PHOTO_URL, PHOTO_ALLOWLIST, max_bytes=10)
                == b"helloworld"
            )
        response.iter_content.assert_called_once_with(chunk_size=32768)
        response.close.assert_called_once_with()

    @pytest.mark.parametrize("length", ("1000", 1000))
    def test_content_length_over_cap_rejected_before_body(self, client, length):
        response = _response(
            headers={"Content-Type": "image/jpeg", "Content-Length": length}
        )
        with patch.object(client.session, "request", return_value=response):
            with pytest.raises(http.ByteCapExceededError):
                client.get_photo(PHOTO_URL, PHOTO_ALLOWLIST, max_bytes=500)
        response.iter_content.assert_not_called()
        response.close.assert_called_once_with()

    @pytest.mark.parametrize("length", (None, "garbage", "-1", "0", "1"))
    def test_missing_invalid_or_understated_length_never_bypasses_stream_cap(
        self, client, length
    ):
        headers = {"Content-Type": "image/jpeg"}
        if length is not None:
            headers["Content-Length"] = length
        response = _response(headers=headers, chunks=(b"ab", b"cd"))
        with patch.object(client.session, "request", return_value=response):
            with pytest.raises(http.ByteCapExceededError):
                client.get_photo(PHOTO_URL, PHOTO_ALLOWLIST, max_bytes=3)
        response.close.assert_called_once_with()

    @pytest.mark.parametrize(
        "content_type", (None, "", "text/html", "application/json")
    )
    def test_photo_requires_image_content_type_before_body(self, client, content_type):
        headers = {} if content_type is None else {"Content-Type": content_type}
        response = _response(headers=headers)
        with patch.object(client.session, "request", return_value=response):
            with pytest.raises(
                http.PublicHttpError, match="Expected image/\\* content-type"
            ):
                client.get_photo(PHOTO_URL, PHOTO_ALLOWLIST)
        response.iter_content.assert_not_called()
        response.close.assert_called_once_with()

    @pytest.mark.parametrize("status", (201, 204, 304, 400, 401, 403, 404, 429, 500))
    def test_non_200_nonredirect_status_closes_without_body(self, client, status):
        response = _response(status)
        with patch.object(client.session, "request", return_value=response):
            with pytest.raises(
                http.PublicHttpError, match=f"Server returned HTTP {status}"
            ):
                client.get_ddg(DDG_URL)
        response.iter_content.assert_not_called()
        response.close.assert_called_once_with()

    @pytest.mark.parametrize("chunks", ((), (b"",), (b"", b"")))
    def test_empty_body_denied_and_closed(self, client, chunks):
        response = _response(chunks=chunks)
        with patch.object(client.session, "request", return_value=response):
            with pytest.raises(http.PublicHttpError, match="Response body was empty"):
                client.get_photo(PHOTO_URL, PHOTO_ALLOWLIST)
        response.close.assert_called_once_with()

    def test_stream_failure_still_closes_response(self, client):
        error = requests.ConnectionError("synthetic stream failure")
        response = _response()

        def body(*, chunk_size):
            assert chunk_size == 32768
            yield b"a"
            raise error

        response.iter_content.side_effect = body
        with patch.object(client.session, "request", return_value=response):
            with pytest.raises(requests.ConnectionError) as caught:
                client.get_photo(PHOTO_URL, PHOTO_ALLOWLIST)
        assert caught.value is error
        response.close.assert_called_once_with()

    def test_request_timeout_translates_with_cause(self, client):
        error = requests.Timeout("synthetic request timeout")
        with patch.object(client.session, "request", side_effect=error) as request:
            with pytest.raises(
                http.TimeCapExceededError, match="Request timed out"
            ) as caught:
                client.get_ddg(DDG_URL)
        assert caught.value.__cause__ is error
        request.assert_called_once()

    def test_request_failure_translates_with_cause(self, client):
        error = requests.ConnectionError("synthetic request failure")
        with patch.object(client.session, "request", side_effect=error) as request:
            with pytest.raises(
                http.PublicHttpError, match="HTTP request failed"
            ) as caught:
                client.get_photo(PHOTO_URL, PHOTO_ALLOWLIST)
        assert caught.value.__cause__ is error
        request.assert_called_once()


class TestTotalDeadline:
    def test_deadline_exhausted_before_first_request(self, client):
        with (
            patch.object(http.time, "monotonic", side_effect=[0.0, 10.0]),
            patch.object(client.session, "request") as request,
        ):
            with pytest.raises(http.TimeCapExceededError):
                client.get_ddg(DDG_URL, timeout=10)
        request.assert_not_called()

    @pytest.mark.parametrize("chunks", ((b"a", b"b"), (b"", b"b")))
    def test_body_deadline_exceeded_and_response_closed(self, client, chunks):
        response = _response(chunks=chunks)
        with (
            patch.object(http.time, "monotonic", side_effect=[0.0, 1.0, 4.0, 10.0]),
            patch.object(client.session, "request", return_value=response) as request,
        ):
            with pytest.raises(http.TimeCapExceededError):
                client.get_photo(PHOTO_URL, PHOTO_ALLOWLIST, timeout=10)
        request.assert_called_once()
        assert request.call_args.kwargs["timeout"] == (9.0, 9.0)
        response.close.assert_called_once_with()

    def test_empty_keepalive_cannot_extend_deadline(self, client):
        response = _response(chunks=(b"",))
        with (
            patch.object(http.time, "monotonic", side_effect=[0.0, 1.0, 10.0]),
            patch.object(client.session, "request", return_value=response),
        ):
            with pytest.raises(http.TimeCapExceededError):
                client.get_photo(PHOTO_URL, PHOTO_ALLOWLIST, timeout=10)
        response.close.assert_called_once_with()

    def test_redirect_chain_uses_one_deadline_not_a_per_hop_reset(self, client):
        first = _response(302, headers={"Location": "/i.js"})
        with (
            patch.object(http.time, "monotonic", side_effect=[0.0, 1.0, 10.0]),
            patch.object(client.session, "request", return_value=first) as request,
        ):
            with pytest.raises(http.TimeCapExceededError):
                client.get_ddg(DDG_URL, timeout=10, max_redirects=5)
        request.assert_called_once()
        first.close.assert_called_once_with()
        first.iter_content.assert_not_called()

    def test_remaining_budget_shrinks_across_redirect_and_body(self, client):
        first = _response(302, headers={"Location": "/i.js"})
        second = _response(chunks=(b"ok",))
        with (
            patch.object(
                http.time, "monotonic", side_effect=[100.0, 101.0, 104.0, 107.0]
            ),
            patch.object(
                client.session, "request", side_effect=[first, second]
            ) as request,
        ):
            assert client.get_ddg(DDG_URL, timeout=10, max_redirects=1) == b"ok"
        assert request.call_args_list[0].kwargs["timeout"] == (9.0, 9.0)
        assert request.call_args_list[1].kwargs["timeout"] == (6.0, 6.0)
        first.close.assert_called_once_with()
        second.close.assert_called_once_with()
