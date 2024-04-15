New-Item -ItemType Directory -Force -Path .\build
clang -c -o build/CoreSystem.o CoreSystem.c
clang -o build/CoreSystem.lib build/CoreSystem.o -fuse-ld=llvm-lib