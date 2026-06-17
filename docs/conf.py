"""
Shared Sphinx configuration using sphinx-multiproject.

To build each project, the ``PROJECT`` environment variable is used.

.. code:: console

   $ make html  # build default project
   $ PROJECT=en make html  # build the English project
   $ PROJECT=zh make html  # build the Chinese project

for more information read https://sphinx-multiproject.readthedocs.io/.
"""

import os
import sys

# Fix imports: check different import methods
try:
    # First try sphinx_multiproject
    from sphinx_multiproject.utils import get_project
    print("INFO: Using sphinx_multiproject")
except ImportError:
    try:
        # Then try multiproject
        from multiproject.utils import get_project
        print("INFO: Using multiproject")
    except ImportError:
        # If both fail, create a simple get_project function
        print("WARNING: sphinx-multiproject not found. Using simple project selection.")
        def get_project(projects):
            return os.environ.get("PROJECT", "en")

sys.path.append(os.path.abspath("_ext"))

# Base extensions - only include actually installed ones
extensions = [
    "multiproject",  # Sphinx extension name, not Python module name
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
    # Temporarily comment out extensions that might cause issues
    # "sphinx_tabs.tabs",  # Module name might be different
    # "sphinx_prompt",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    # Comment out uninstalled extensions
    # "sphinxcontrib.httpdomain",
    # "sphinxcontrib.video",
    # "sphinxemoji.sphinxemoji",
    "sphinxext.opengraph",
    "sphinx_tippy",
    "sphinx_togglebutton",
]

# Check and add actually installed extensions
try:
    import sphinx_tabs
    extensions.append("sphinx_tabs.tabs")
    print("INFO: sphinx_tabs extension added")
except ImportError:
    print("INFO: sphinx_tabs not available")

try:
    import sphinx_prompt
    extensions.append("sphinx_prompt")
    print("INFO: sphinx_prompt extension added")
except ImportError:
    print("INFO: sphinx_prompt not available")

multiproject_projects = {
    "en": {
        "use_config_file": False,
        "config": {
            "project": "KernelGen Documentation",
            "html_title": "KernelGen Documentation",
        },
    },
    "zh": {
        "use_config_file": False,
        "config": {
            "project": "KernelGen 文档中心",
            "html_title": "KernelGen 文档中心",
        },
    },
}

docset = get_project(multiproject_projects)

ogp_site_name = "KernelGen Documentation"
ogp_use_first_image = True
ogp_image = "https://docs.readthedocs.io/en/latest/_static/img/logo-opengraph.png"
ogp_custom_meta_tags = (
    '<meta name="twitter:card" content="summary_large_image" />',
)
ogp_enable_meta_description = True
ogp_description_length = 300

templates_path = ["_templates"]
html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "/")

master_doc = "index"
copyright = '2025-2026, FlagOS Community'
author = 'FlagOS Community'
release = '1.0.0'
# release = version
exclude_patterns = ["_build", "shared", "_includes"]
if docset == "zh":
    exclude_patterns.append("en")
elif docset == "en":
    exclude_patterns.append("zh")
default_role = "obj"
intersphinx_cache_limit = 14
intersphinx_timeout = 3
intersphinx_mapping = {
    "python": ("https://docs.python.org/3.10/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}

intersphinx_disabled_reftypes = ["*"]

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "fieldlist",
    "html_admonition",
    "html_image",
    "colon_fence",
    "smartquotes",
    "replacements",
    # "linkify",
    "strikethrough",
    "substitution",
    "tasklist",
    "attrs_inline",
    "attrs_block",
]
htmlhelp_basename = "KernelGendoc"
latex_documents = [
    (
        "index",
        "KernelGen.tex",
        "KernelGen Documentation",
        "KernelGen Team",
        "manual",
    ),
]
man_pages = [
    (
        "index",
        "kernelgen",
        "KernelGen Documentation",
        ["KernelGen Team"],
        1,
    )
]

language = "en" if docset == "en" else "zh_CN"

# 强制保障：只要当前构建的目标不是英文，或者当前环境被 RTD 强行指定为了中文
if docset == "zh" or language in ["zh_CN", "zh"]:
    language = "zh_CN"
    html_search_language = "zh"  # 确保给 Sphinx 服务端/前端搜索注入正确的中文化标识

locale_dirs = [
    f"{docset}/locale/",
]
gettext_compact = False

html_theme = "sphinx_book_theme"
html_static_path = ["_static", f"{docset}/_static"]
# html_css_files = ["css/custom.css"]
# Don't add sphinx_prompt_css.css for now, it might not exist
# html_js_files = []

# html_logo = "img/logo.svg"
html_favicon = "img/favicon.svg"

# if docset == 'en':
#     html_favicon = 'img/en-logo.svg'
# else:
#     html_favicon = 'img/zh-logo.svg'


# html_theme_options = {
#     "logo_only": True,
# }

# html_theme_options = {
#     "logo": {
#       "image_light": "_static/kernelgen-logo.svg",
#       "image_dark": "_static/kernelgen-logo.svg",
#    },
#     "home_page_in_toc": True,
#     "use_download_button": False,
#     "repository_url": "https://github.com/flagos-ai/KernelGen",
#     "use_edit_page_button": True,
#     # "github_url": "https://github.com/flagos-ai/KernelGen",
#     # "repository_branch": "master",
#     # "path_to_docs": "docs",
#     "use_repository_button": True,
#     # "announcement": "<b>v3.0.0</b> is now out! See the Changelog for details",
# }

# 根据语言选择 logo（浅色/深色）
if docset == 'en':
    logo_light = '_static/en-logo.svg'
    logo_dark = '_static/en-logo-dark.svg'
else:  # zh
    logo_light = '_static/zh-logo.svg'
    logo_dark = '_static/zh-logo-dark.svg'

html_theme_options = {
    "logo": {
        "image_light": logo_light,
        "image_dark": logo_dark,
    },
    "home_page_in_toc": True,
    "use_download_button": False,
    "repository_url": "https://github.com/flagos-ai/KernelGen",
    "use_edit_page_button": True,
    "use_repository_button": True,
}

# 如果有其他地方还用到 html_logo 或 html_favicon，建议删除或注释掉
# html_logo = "img/logo.svg"
# html_favicon = "img/logo.svg"

# html_context = {
#     "conf_py_path": f"/docs/{docset}/",
#     "display_github": True,
#     "github_user": "armstrongttwalker-alt",
#     "github_repo": "test-i18n-KernelGen",
#     "github_version": "main",
#     "plausible_domain": f"{os.environ.get('READTHEDOCS_PROJECT')}.readthedocs.io",
# }

rst_epilog = """
.. |org_brand| replace:: KernelGen Community
.. |com_brand| replace:: KernelGen for Business
.. |git_providers_and| replace:: GitHub, Bitbucket, and GitLab
.. |git_providers_or| replace:: GitHub, Bitbucket, or GitLab
"""

autosectionlabel_prefix_document = True

linkcheck_retries = 2
linkcheck_timeout = 1
linkcheck_workers = 10
linkcheck_ignore = [
    r"http://127\.0\.0\.1",
    r"http://localhost",
    r"https://github\.com.+?#L\d+",
]

extlinks = {
    "issue": ("https://github.com/armstrongttwalker-alt/test-i18n-KernelGen/issues/%s", "#%s"),
}

suppress_warnings = ["epub.unknown_project_files"]

# =====================================================================
# Adaptive Search & Path Fixes for Sphinx 9+ and sphinx-multiproject
# =====================================================================

# 1. Inject front-end fallback scripts early in HTML <head> for Chinese localization.
# This bypasses the Sphinx 9+ 'stemmer.stemWord is not a function' runtime crash.
# html_context = {}
# if docset == "zh":
#     html_context["metatags"] = """
#     <script>
#         if (typeof Stemmer === 'undefined') {
#             window.Stemmer = function() { 
#                 return { 
#                     stem: function(w){ return w; },
#                     stemWord: function(w){ return w; }
#                 }; 
#             };
#         }
#         if (typeof ChineseStemmer === 'undefined') {
#             window.ChineseStemmer = window.Stemmer;
#         }
#     </script>
#     """

# def setup(app):
#     # Prevent MIME type errors (404 returned as text/html) caused by missing custom.css
#     # by forcing an empty placeholder into both the root and localized static dirs.
#     import shutil
#     for target_dir in [app.srcdir, os.path.join(app.srcdir, docset)]:
#         css_dir = os.path.join(target_dir, '_static', 'css')
#         os.makedirs(css_dir, exist_ok=True)
#         with open(os.path.join(css_dir, 'custom.css'), 'w', encoding='utf-8') as f:
#             f.write('/* custom styles to prevent RTD 404 */\n')

#     # Runtime correction for DOCUMENTATION_OPTIONS based on the current docset
#     app.add_js_file(None, body=f'DOCUMENTATION_OPTIONS.LANGUAGE = "{language}";')
#     if docset == "zh":
#         app.add_js_file(None, body='DOCUMENTATION_OPTIONS.URL_ROOT = "../";')
#     elif docset == "en":
#         app.add_js_file(None, body='DOCUMENTATION_OPTIONS.URL_ROOT = "./";')

# # 2. Dynamically assign search engine language to avoid indexing mismatch
# if docset == "zh":
#     html_search_language = 'zh'
# else:
#     html_search_language = 'en'