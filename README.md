# (Quick) Programming Language
## Not named Quick, just a working title (because the original version was implemented in an hour or so).

### Goals:
- Uniformity
- Simplicity
- Performance
- Fun
    - It's a hobby language.

### TODOs

#### IMPORTANT
- Current solution to compiling multiple files requires every typer to type check all the things they depend on. This could lead to a lot of extra work.
  Can we make it so that type checking outside of the current file only skips function bodies specifically and only checks types and function signatures?
  This should be possible, at least for non-generics as a first pass.
  This seems IMPORTANT for performance!

#### Type Checking
- Check for duplicate names of variables, functions, types, fields, etc.

#### Project Structure
- Namespaces.
    - Each file is it's own namespace, named after the file. (eg. a function `Bar()` defined in `Foo.quick` is called with `Bar();` in `Foo.quick` and `Foo.Bar();` elsewhere)
- Some form of incremental compilation.
- Maybe parallel compilation too?

#### Features
- Default parameters.
- Variadic arguments.
- `for elem in array {}`
- Possibly generic type inference, so specifiers aren't required?
- Bring back methods, implemented better than they were originally.
    - Define methods as normal free functions, but with their owning type at the front (eg. `func PlayerStruct.JumpMethod() Void {}` called as `playerInstance.JumpMethod();`)
- Enum methods.
- Top level const declarations.

#### Missing Things and Bug Fixes
- Struct equality comparisons.
- Allow `sizeof` on complex types at compile time.
- Some form of debug info: https://learn.microsoft.com/en-us/cpp/preprocessor/hash-line-directive-c-cpp
- Stack traces.

#### Standard Library
- `func alloc<T>(value T) *T` alternative to malloc for simple situations.

#### Cleanup
- Code that generates and uses union `__WithTag/__CheckTag` needs factoring.
- Refer to TODOs in source files.

#### Notes
- Generics are handled like "templates", they get type checked once for each usage. This is nice for flexibility, eg. a generic function that contains `genericFoo.bar()` will only cause an error if that generic function is called with a type parameter that doesn't have a `.bar()` method. However, it means that a generic function has to get used for it to get type checked.