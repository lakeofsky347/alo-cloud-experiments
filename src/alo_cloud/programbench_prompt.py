"""Official ProgramBench submission contract, added to every ProgramBench prompt.

The scorer at the pinned ProgramBench commit runs
``chmod +x ./compile.sh && ./compile.sh`` and then moves ``./executable``.
The task_cleanroom_v6 image and the no-network rule are the ones described
for that harness. SWE-bench prompts do not use this text.
"""

PROGRAMBENCH_SUBMISSION_CONTRACT = """\
ProgramBench submission contract.

The reference program is ./executable in the workspace root. It is execute-only.
Run it to observe command-line behavior, and read the bundled documentation.
Do not decompile ./executable and do not use strace, ltrace, objdump, or a disassembler on it.
Do not wrap, shim, or delegate to ./executable or to an installed copy of the same program.
This container has no network.

The sandbox image is task_cleanroom_v6 (Ubuntu 22.04) with these toolchains already installed:
Rust 1.92.0 (rustc, cargo), Go 1.21.0, Python 3.10.12, build-essential, cmake, git, and tmux.
No task-specific packages are preinstalled. The official compile step also blocks the network,
so compile.sh cannot download dependencies.

Before deep exploration, write a minimal ./compile.sh at the workspace root that builds ./executable, and run it once. Then iterate on the implementation. Do not spend the request budget only on reading files and probing the reference binary.

Make sure that you have a ./compile.sh file that produces an executable ./executable in the workspace root.
compile.sh should be executable.
The official scorer runs: chmod +x ./compile.sh && ./compile.sh
If compile.sh fails to compile on a fresh checkout, the submission has failed.
A Makefile or cargo project alone is not enough: compile.sh has to produce ./executable.
Do not ship a prebuilt ./executable. The scorer deletes ./executable before it runs compile.sh, then requires ./executable at the workspace root.

The image's ./executable is the reference. It may be owned by root and is execute-only. Do not overwrite it by hand and do not read its bytes.
compile.sh must create ./executable at the workspace root. You may compile under ./build/ and install your own build output onto ./executable, for example: cp build/executable executable
That install copies your build, not the reference. Copying or wrapping the reference is not allowed.
Use python3 when you invoke Python; the image may not have a python command.
"""

PROGRAMBENCH_BUDGET_NOTICE = (
    "Budget notice: 3 or fewer model requests remain in this unit. "
    "Stop exploring. Finalize compile.sh and ./executable now. "
    "compile.sh must sit at the workspace root and must build ./executable."
)
