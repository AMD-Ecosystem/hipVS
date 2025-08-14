
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

version_number = "1.0.0" # TODO: Parse this from a centralized location.
left_nav_title = f"hipVS {version_number} documentation"

# for PDF output on Read the Docs
project = "hipVS"
author = "Advanced Micro Devices, Inc."
copyright = "Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved."
version = version_number
release = version_number
cpp_maximum_signature_line_length = 10
setting_all_article_info = True
all_article_info_os = ["linux"]
all_article_info_author = ""

external_projects_current_project = "hipVS"

html_theme = "rocm_docs_theme"
html_theme_options = {"flavor": "rocm-ds", "repository_url": "https://github.com/AMD-AIOSS/hipVS/"}

extensions = [
    "rocm_docs",
    "breathe",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx_copybutton",
]

myst_heading_anchors = 4
autosectionlabel_prefix_document = True

# Breathe configuration for Doxygen
breathe_projects = {"cuvs": "./doxygen/xml"}  # Ensure Doxygen XML is in ./xml
breathe_default_project = "cuvs"

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "special-members": "__init__, __getitem__",
    "inherited-members": True,
    "show-inheritance": True,
    "imported-members": False,
    "member-order": "bysource",  # bysource: seems unfortunately not to work for Cython modules
}

source_suffix = {
    ".rst": "restructuredtext",
}

external_toc_path = "./sphinx/_toc.yml"
doxygen_root = "doxygen"
doxysphinx_enabled = True
doxygen_project = {
    "name": "doxygen",
    "path": "doxygen/xml",
}
