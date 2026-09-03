# Source this before anything that touches gsplat: `source scripts/env.sh`.
#
# gsplat compiles its CUDA kernels on first use, and three things have to be
# true for that to work. None of them are true when the environment's python is
# invoked by absolute path, which is the obvious way to run it and the way that
# fails:
#
#   PATH      the env's bin must come first, or torch cannot find `ninja` and
#             refuses to build at all - even though ninja is installed.
#   CUDA_HOME points torch's extension builder at the toolkit inside the env.
#   CPATH     conda puts the CUDA headers under targets/x86_64-linux/include,
#             which nvcc does not search by default, so the build dies on
#             `cuda_runtime_api.h: No such file or directory`.
#
# The build takes about a minute and is cached afterwards, keyed on the absolute
# paths - so a different environment prefix means a fresh build, not a reuse.

TWINWORLD_ENV="${TWINWORLD_ENV:-$HOME/micromamba/envs/twinworld}"

if [ ! -x "$TWINWORLD_ENV/bin/python" ]; then
    echo "twinworld env not found at $TWINWORLD_ENV" >&2
    return 1 2>/dev/null || exit 1
fi

export PATH="$TWINWORLD_ENV/bin:$PATH"
export CUDA_HOME="$TWINWORLD_ENV"
export CPATH="$TWINWORLD_ENV/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$TWINWORLD_ENV/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"

# The card here is an RTX 6000 Ada (sm_89). Naming it skips the "all archs for
# visible cards" warning and keeps the build to the one architecture we use.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
