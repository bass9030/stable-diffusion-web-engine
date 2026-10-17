"""Playwright(headless Chromium) 기반 HTML 렌더러.

JS를 끄고 모든 네트워크 요청을 차단해 외부 리소스 없이 렌더링한다.
페이지 하나를 재사용하며 set_content로 문서를 교체한다.
"""
from __future__ import annotations

from types import TracebackType
from typing import Any

from playwright.sync_api import Browser, BrowserContext, Page, Playwright, Route, sync_playwright


class Renderer:
    def __init__(self, width: int, height: int, device_scale_factor: float, base_css: str) -> None:
        self.width = width
        self.height = height
        self.device_scale_factor = device_scale_factor
        self.base_css = base_css
        self._pw: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None

    @classmethod
    def from_config(cls, render_cfg: dict[str, Any]) -> "Renderer":
        vp = render_cfg["viewport"]
        return cls(vp["width"], vp["height"], render_cfg["device_scale_factor"], render_cfg["base_css"])

    def __enter__(self) -> "Renderer":
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch()
        self._context = self._browser.new_context(
            viewport={"width": self.width, "height": self.height},
            device_scale_factor=self.device_scale_factor,
            java_script_enabled=False,
        )
        self._context.route("**/*", self._block_request)
        self._page = self._context.new_page()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._browser is not None:
            self._browser.close()
        if self._pw is not None:
            self._pw.stop()

    @staticmethod
    def _block_request(route: Route) -> None:
        route.abort()

    @property
    def browser_version(self) -> str:
        assert self._browser is not None, "Renderer는 with 블록 안에서 사용해야 한다"
        return self._browser.version

    def render(self, html: str) -> bytes:
        """HTML을 렌더링해 viewport 크기의 PNG 바이트를 반환한다."""
        assert self._page is not None, "Renderer는 with 블록 안에서 사용해야 한다"
        self._page.set_content(f"<style>{self.base_css}</style>{html}", wait_until="load")
        return self._page.screenshot(type="png")
