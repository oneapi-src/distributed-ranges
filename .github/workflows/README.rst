.. SPDX-FileCopyrightText: Intel Corporation
..
.. SPDX-License-Identifier: BSD-3-Clause

===================
 GitHub Actions CI
===================

DevCloud Runners
================

We have 2 self-hosted runners on devcloud. We run in an emacs daemon
so it will continue to run after disconnecting. ``ssh`` to DevCloud
and start an emacs daemon::

  emacs --daemon

Inside an emacs shell::

  cd github/runner-1
  ./run.sh

Rename the shell to ``devcloud-1`` and start another emacs shell::

  cd github/runner-2
  ./run.sh

If you disconnect emacs and the runners will continue to run. To check
on a runner ``ssh`` to DevCloud and do::

  emacsclient -nw

And visit ``devcloud-1`` and ``devcloud-2`` buffers.

``tmux`` is an alternative. I do not use it because devcloud allows 4
logins via ``ssh``. Using ``tmux`` with 2 runners consumes 3 even
while it is detached. You may create a situation where you cannot log
in because ``tmux`` is consuming all the logins.
