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
- Check for duplicate names of fields and environment variables.

#### Project Structure
- Some form of incremental compilation.
- Maybe parallel compilation too?

#### Features
- Default parameters.
- `for elem in array {}`
- Possibly generic type inference, so specifiers aren't required?
- Maybe new/scope/delete keywords like in Beef?

#### Missing Things and Bug Fixes
- Allow `sizeof` on complex types at compile time.
- Some form of debug info: https://learn.microsoft.com/en-us/cpp/preprocessor/hash-line-directive-c-cpp
- Stack traces.

#### Cleanup
- Refer to TODOs in source files.

#### Notes
- Generics are handled like "templates", they get type checked once for each usage. This is nice for flexibility, eg. a generic function that contains `genericFoo.bar()` will only cause an error if that generic function is called with a type parameter that doesn't have a `.bar()` method. However, it means that a generic function has to get used for it to get type checked.