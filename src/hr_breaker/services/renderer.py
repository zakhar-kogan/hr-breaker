"""Abstract renderer interface and implementations."""

import html
import os
import re
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from urllib.parse import urlparse

from jinja2 import Environment, FileSystemLoader

from hr_breaker.models.resume_data import ContactInfo, ResumeData, RenderResult

# Template directory
TEMPLATE_DIR = Path(__file__).parent.parent.parent.parent / "templates"


def _setup_macos_library_path():
    """Set up library path for WeasyPrint on macOS with Homebrew."""
    if sys.platform != "darwin":
        return

    # Check if DYLD_FALLBACK_LIBRARY_PATH is already set
    if os.environ.get("DYLD_FALLBACK_LIBRARY_PATH"):
        return

    # Try common Homebrew paths
    homebrew_paths = [
        "/opt/homebrew/lib",  # Apple Silicon
        "/usr/local/lib",  # Intel
    ]

    for path in homebrew_paths:
        gobject_lib = Path(path) / "libgobject-2.0.dylib"
        if gobject_lib.exists():
            os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = path
            return


def _canonical_profile_url(raw: str, *, domain: str, path_prefix: str = "") -> str:
    value = raw.strip()
    if not value:
        return value
    if value.startswith("http://") or value.startswith("https://"):
        return value

    normalized = value.lstrip("@/")
    if normalized.lower().startswith(f"{domain}/"):
        return f"https://{normalized}"
    if normalized.lower().startswith(domain):
        return f"https://{normalized}"
    return f"https://{domain}/{path_prefix}{normalized}"


def _canonical_website_url(raw: str) -> str:
    value = raw.strip()
    if not value:
        return value
    if value.startswith(("http://", "https://", "mailto:", "tel:")):
        return value
    return f"https://{value}"


def _display_url(raw: str) -> str:
    value = raw.strip()
    if not value:
        return value
    if value.startswith(("mailto:", "tel:")):
        return value.split(":", 1)[1]
    if not value.startswith(("http://", "https://")):
        value = f"https://{value}"
    parsed = urlparse(value)
    display = parsed.netloc + parsed.path
    if parsed.query:
        display += f"?{parsed.query}"
    return display.rstrip("/")


def _render_contact_link(href: str, label: str) -> str:
    return f'<a href="{html.escape(href, quote=True)}">{html.escape(label)}</a>'


def _render_header_html(contact_info: ContactInfo | None) -> str:
    if contact_info is None or not contact_info.name:
        return ""

    contact_parts: list[str] = []
    if contact_info.email:
        contact_parts.append(_render_contact_link(f"mailto:{contact_info.email}", contact_info.email))
    if contact_info.phone:
        contact_parts.append(html.escape(contact_info.phone))
    if contact_info.location:
        contact_parts.append(html.escape(contact_info.location))
    if contact_info.linkedin:
        href = _canonical_profile_url(contact_info.linkedin, domain="linkedin.com", path_prefix="in/")
        contact_parts.append(_render_contact_link(href, _display_url(href)))
    if contact_info.github:
        href = _canonical_profile_url(contact_info.github, domain="github.com")
        contact_parts.append(_render_contact_link(href, _display_url(href)))
    if contact_info.website:
        href = _canonical_website_url(contact_info.website)
        contact_parts.append(_render_contact_link(href, _display_url(href)))

    contact_line = '<span class="sep">|</span>'.join(contact_parts)
    return (
        '<header class="header">'
        f'<h1 class="name">{html.escape(contact_info.name)}</h1>'
        f'<div class="contact-line">{contact_line}</div>'
        '</header>'
    )


def _strip_generated_header(html_body: str) -> str:
    """Remove an LLM-generated header when the renderer injects the canonical one."""
    return re.sub(
        r"^\s*<header[^>]*class=[\"\'][^\"\']*header[^\"\']*[\"\'][^>]*>.*?</header>\s*",
        "",
        html_body,
        count=1,
        flags=re.IGNORECASE | re.DOTALL,
    )


class RenderError(Exception):
    """Raised when rendering fails."""

    pass


class BaseRenderer(ABC):
    """Abstract base class for resume renderers."""

    @abstractmethod
    def render(self, html_body: str, contact_info: ContactInfo | None = None) -> RenderResult:
        """Render resume HTML body to PDF."""
        pass


class HTMLRenderer(BaseRenderer):
    """Render resume using HTML + WeasyPrint."""

    _weasyprint_imported = False

    def __init__(self):
        self._ensure_weasyprint()
        self.env = Environment(
            loader=FileSystemLoader(TEMPLATE_DIR),
            autoescape=True,
        )
        from weasyprint.text.fonts import FontConfiguration
        self.font_config = FontConfiguration()
        self._wrapper_html = (TEMPLATE_DIR / "resume_wrapper.html").read_text(encoding="utf-8")

    @classmethod
    def _ensure_weasyprint(cls):
        """Lazily import WeasyPrint with proper library path setup."""
        if cls._weasyprint_imported:
            return

        # Set up library path before importing
        _setup_macos_library_path()

        try:
            # Import WeasyPrint - this will fail if libs not found
            import weasyprint  # noqa: F401
            cls._weasyprint_imported = True
        except OSError as e:
            err = str(e)
            if any(lib in err for lib in ("libgobject", "libpango", "libcairo", "libgdk_pixbuf")):
                if sys.platform == "darwin":
                    msg = (
                        "WeasyPrint libraries not found. On macOS, run:\n"
                        "  brew install pango gdk-pixbuf libffi\n"
                        "Then either:\n"
                        "  1. Add to ~/.zshrc: export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib\n"
                        "  2. Or run with: DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib uv run hr-breaker ..."
                    )
                elif sys.platform == "win32":
                    msg = (
                        "WeasyPrint libraries (GTK3) not found on Windows.\n"
                        "Download and install the GTK3 runtime from:\n"
                        "  https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer/releases\n"
                        "Ensure the GTK3 bin folder is in your PATH and restart your terminal."
                    )
                else:
                    msg = (
                        "WeasyPrint libraries (Pango, Cairo, GdkPixbuf) not found.\n"
                        "Install them using your system package manager:\n"
                        "  Ubuntu/Debian: sudo apt-get install libpango-1.0-0 libharfbuzz0b libpangoft2-1.0-0\n"
                        "  Fedora: sudo dnf install pango cairo gdk-pixbuf2"
                    )
                raise RenderError(msg) from e
            raise

    def render(self, html_body: str, contact_info: ContactInfo | None = None) -> RenderResult:
        """Render LLM-generated HTML body to PDF.

        Args:
            html_body: HTML content for the <body> (no wrapper needed)
            contact_info: Optional structured contact data for deterministic header rendering
        """
        from weasyprint import HTML

        body_html = _strip_generated_header(html_body) if contact_info else html_body
        header_html = _render_header_html(contact_info)
        html_content = (
            self._wrapper_html
            .replace("{{HEADER}}", header_html)
            .replace("{{BODY}}", body_html)
        )

        html = HTML(string=html_content, base_url=str(TEMPLATE_DIR))
        doc = html.render(font_config=self.font_config)
        pdf_bytes = doc.write_pdf()
        page_count = len(doc.pages)

        warnings = []
        if page_count > 1:
            warnings.append(f"Resume is {page_count} pages, should be 1 page")

        return RenderResult(
            pdf_bytes=pdf_bytes,
            page_count=page_count,
            warnings=warnings,
        )

    def render_data(self, data: ResumeData) -> RenderResult:
        """Legacy: Render ResumeData to PDF via Jinja template."""
        from weasyprint import HTML, CSS

        template = self.env.get_template("resume.html")
        html_content = template.render(resume=data)

        html = HTML(string=html_content, base_url=str(TEMPLATE_DIR))
        css_path = TEMPLATE_DIR / "resume.css"
        stylesheets = []
        if css_path.exists():
            stylesheets.append(CSS(filename=str(css_path), font_config=self.font_config))

        doc = html.render(stylesheets=stylesheets, font_config=self.font_config)
        pdf_bytes = doc.write_pdf()
        page_count = len(doc.pages)

        warnings = []
        if page_count > 1:
            warnings.append(f"Resume is {page_count} pages, should be 1 page")

        return RenderResult(
            pdf_bytes=pdf_bytes,
            page_count=page_count,
            warnings=warnings,
        )


def get_renderer() -> HTMLRenderer:
    """Get the HTML renderer."""
    return HTMLRenderer()