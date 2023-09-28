.. SPDX-FileCopyrightText: Intel Corporation
..
.. SPDX-License-Identifier: BSD-3-Clause

===================
 GitHub Actions CI
===================

DevCloud Runners
================

We have 2 self-hosted runners on devcloud. We run in tmux so it will
continue to run after disconnecting. ``ssh`` to DevCloud::

  tmux
  cd github/runner-1
  ./run.sh

For the second runner, split the window with ``control-b "`` and start
the runner::

  cd github/runner-2
  ./run.sh

To check on a runner ``ssh`` to DevCloud and do::

  tmux ls

To see the sessions. Usually there is just one name ``0``. Attach to
the session::

  tmux attach -t 0
