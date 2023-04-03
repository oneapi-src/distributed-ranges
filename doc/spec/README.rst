.. SPDX-FileCopyrightText: Intel Corporation
..
.. SPDX-License-Identifier: BSD-3-Clause

==================
 Editing the Spec
==================

Build the spec::

  make -C doc/spec html

Open in your browser: ``doc/spec/build/html/index.html``

Doxygen html is at: ``doc/spec/build/doxygen-html/index.html``

There are pre-commit checks for spelling and broken links. To run it manually::

  make -C doc/spec linkcheck
  make -C doc/spec spelling SPHINXOPTS=-q

The `SPHINXOPTS` is necessary to disable warning as errors, so you can
see all the spelling errors instead of the first one. Add spelling
exceptions to `spelling_wordlist.txt`. Do not add variable, class,
function, etc to the exceptions. Spellcheck ignores them if they are
properly delimited in the source doc.
