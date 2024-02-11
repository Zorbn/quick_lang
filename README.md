# (Quick) Programming Language
## Not named Quick, just a working title (because the original version was implemented in an hour or so).

### Goals:
- Uniformity
- Simplicity
- Performance
- Fun
    - It's a hobby language.

### TODOs

#### Type Checking
- Distinguish between variable (var) and constant (val, const, fun) bindings.
- Check function call arguments.
- Check struct literal fields.
- Make sure all non-void functions return, and that all returns have the correct type.
- Check for duplicate names of variables, functions, types, fields, etc.

#### Project Structure
- Namespaces.
- Some form of incremental compilation.
- Maybe parallel compilation too?

#### Features
- Default parameters.
- Variadic arguments.
- `for elem in array {}`
- Bitwise operations (with more intuitive precedence than C).
- Some way to represent pointers to immutable data, eg. `*var Int` vs `*val/const Int`
- Possibly generic type inference, so specifiers aren't required?
- Enum methods.

#### Missing Things and Bug Fixes
- Struct equality comparisons.
- Allow `sizeof` on complex types at compile time.
- Don't print duplicate type errors for generics, probably best to stop checking them if one variant had an error, or maybe it would be easier to stop handling generic usages after an error.
- Some for of debug info: https://learn.microsoft.com/en-us/cpp/preprocessor/hash-line-directive-c-cpp

#### Standard Library
- `func alloc<T>(value T) *T` alternative to malloc for simple situations.

#### Cleanup
- Code that generates and uses union `__WithTag/__CheckTag` needs factoring.
- Refer to TODOs in source files.