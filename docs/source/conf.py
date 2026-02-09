# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'COREFL'
copyright = '2026, Xinliang Guo'
author = 'Xinliang Guo'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
  "myst_parser",                    # Markdown support
  "sphinx.ext.mathjax",             # Math rendering
  "sphinx.ext.autosectionlabel",  # Automatic section labels
]
source_suffix = {
  ".md": "markdown",
  ".rst": "restructuredtext",
}
root_doc = "index"

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
  "collapse_navigation": False,
  "navigation_depth": 4,
}
html_context = {
  "display_github": True,
  "github_user": "Liangerty",
  "github_repo": "COREFL-CPC",
  "github_version": "main",
  "conf_py_path": "/docs/source/",
}
html_static_path = ['_static']
