// https://docs.mathjax.org/en/latest/input/tex/extensions/autoload.html

window.MathJax = {
    tex: {
      inlineMath: [["\\(", "\\)"]],
      displayMath: [["\\[", "\\]"]],
      processEscapes: true,
      processEnvironments: true,
      autoload: expandable({
        action: ['toggle', 'mathtip', 'texttip'],
        amscd: [[], ['CD']],
        bbox: ['bbox'],
        boldsymbol: ['boldsymbol'],
        braket: ['bra', 'ket', 'braket', 'set', 'Bra', 'Ket', 'Braket', 'Set', 'ketbra', 'Ketbra'],
        cancel: ['cancel', 'bcancel', 'xcancel', 'cancelto'],
        color: ['color', 'definecolor', 'textcolor', 'colorbox', 'fcolorbox'],
        enclose: ['enclose'],
        extpfeil: ['xtwoheadrightarrow', 'xtwoheadleftarrow', 'xmapsto',
                   'xlongequal', 'xtofrom', 'Newextarrow'],
        html: ['href', 'class', 'style', 'cssId'],
        mhchem: ['ce', 'pu'],
        newcommand: ['newcommand', 'renewcommand', 'newenvironment', 'renewenvironment', 'def', 'let'],
        unicode: ['unicode'],
        verb: ['verb']
      }),
    },
    options: {
      ignoreHtmlClass: ".*|",
      processHtmlClass: "arithmatex"
    }
  };
  
  document$.subscribe(() => { 
    MathJax.typesetPromise()
  })