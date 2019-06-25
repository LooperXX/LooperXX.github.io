# MkDocs_demo

## Demo yml

```yml
# Project information
site_name: Looper's wiki
site_description: Looper's personal notes
site_author: 'Looper - Xiao Xu'
site_url: 'https://looperxx.github.io/My_wiki'

# Repository
repo_name: 'looperxx/My_wiki'
repo_url: 'https://github.com/looperxx/My_wiki'

# Copyright
copyright: 'Copyright &copy; 2019 - 2020 Looper Xiao Xu'

nav:
  - Home: index.md
  - NLP:
      - 'Machine Reading Comprehension': 'Neural Reading Comprehension and beyond.md'
  - For MkDocs:
      - 'Demo' : 'MkDocs_demo.md'
      - 'Material Theme Tutorial' : 'Material Theme Tutorial.md'

theme:
  name: 'material' 
  # name: 'readthedocs'
  language: 'zh'
  # palette:
  #   primary: 'indigo'
  #   accent: 'indigo'
  # font:
  #   text: 'Roboto'
  #   code: 'Roboto Mono'
  feature:
      tabs: true
  # logo:
  #   icon: 'cloud'

# Customization
extra:
  social:
    - type: 'github'
      link: 'https://github.com/LooperXX'
    - type: 'linkedin'
      link: 'https://www.linkedin.com/in/%E5%95%B8-%E5%BE%90-012456163/'
  disqus: 'https-looperxx-github-io-my-wiki'
  # search:   
  #   language: 'en, zh'

# Google Analytics
google_analytics:
  - 'UA-XXXXXXXX-X'
  - 'auto'

# Extensions
extra_javascript:
  -  'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-MML-AM_CHTML'
markdown_extensions:
  - admonition
  - codehilite:
      guess_lang: false
      linenums: true
  - footnotes
  - meta
  - toc:
      permalink: true
  - pymdownx.arithmatex
  - pymdownx.betterem:
      smart_enable: all
  - pymdownx.caret
  - pymdownx.critic
  - pymdownx.details
  - pymdownx.emoji:
      emoji_generator: !!python/name:pymdownx.emoji.to_svg
  - pymdownx.inlinehilite
  - pymdownx.magiclink
  - pymdownx.mark
  - pymdownx.smartsymbols
  - pymdownx.superfences
  - pymdownx.tasklist:
      custom_checkbox: true
  - pymdownx.tilde

```

## Requirements

```shell
pip install mkdocs
pip install mkdocs-material
pip install pygments
pip install pymdown-extensions
```

